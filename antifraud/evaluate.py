"""Оценка качества на синтетике с известной разметкой и проверка устойчивости.

Сценарии сдвигают параметры мошенников относительно описания кейса. Для каждого
считаются precision/recall правил, отдельно по схемам, ложные срабатывания на
легальных ловушках, а также качество оценки (score) и ML-модели, обученной на
реальных данных (перенос на чужое распределение).
"""
from __future__ import annotations

import json
import os
import time

import polars as pl

from .config import DEFAULT_OUT, load_config
from .detectors import run_detectors
from .features import build_features
from .ingest import parse
from .scoring import score_numbers
from .synth import Synth

SCENARIOS = {
    "base": {"scale": 1.0, "dur_shift": 1.0, "desc": "параметры как в описании кейса"},
    "quiet_x0.5": {"scale": 0.5, "dur_shift": 1.0, "desc": "мошенники вдвое «тише» (меньше вызовов)"},
    "loud_x1.5": {"scale": 1.5, "dur_shift": 1.0, "desc": "мошенники в 1,5 раза активнее"},
    "dur_x1.3": {"scale": 1.0, "dur_shift": 1.3, "desc": "длительности вызовов +30%"},
    "noisy": {"scale": 1.0, "dur_shift": 1.0, "error_rate": 0.01, "desc": "1% строк с ошибками формата"},
    # стресс-тесты: на них пороги НЕ настраивались
    "stress_x0.3": {"scale": 0.3, "dur_shift": 1.0, "seed": 101, "desc": "стресс: мошенники втрое тише"},
    "stress_mix": {"scale": 0.7, "dur_shift": 0.8, "error_rate": 0.003, "seed": 202,
                   "desc": "стресс: −30% вызовов, −20% длительности, 0,3% ошибок, другое зерно"},
}


def _prf(pred: set, true: set):
    tp = len(pred & true)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(true) if true else 1.0
    return round(p, 4), round(r, 4), tp


def evaluate_scenario(name: str, params: dict, cfg: dict, model=None, seed: int = 7) -> dict:
    s = Synth(seed=seed, scale=params.get("scale", 1.0), dur_shift=params.get("dur_shift", 1.0),
              error_rate=params.get("error_rate", 0.0003)).build()
    df, rep = parse(s.sources(), cfg)
    f = build_features(df, cfg)
    hits, rel, legit = run_detectors(df, f, cfg)
    scores = score_numbers(f, hits, rel, legit.join(hits.select("num"), on="num", how="anti"), cfg)
    true = set(s.truth); pred = set(hits["num"])
    p, r, tp = _prf(pred, true)
    legit_true = set(s.legit)
    res = {"scenario": name, "desc": params["desc"], "rows": df.height, "quarantined": len(rep.quarantine),
           "truth_numbers": len(true), "predicted": len(pred), "precision": p, "recall": r,
           "legit_hit_as_fraud": len(pred & legit_true), "legit_total": len(legit_true),
           "legit_recognized": round(len(set(legit["num"]) & legit_true) / len(legit_true), 3)}
    per = {}
    for sch in sorted({v[0] for v in s.truth.values()}):
        t = {n for n, v in s.truth.items() if v[0] == sch}
        pr = set(hits.filter(pl.col("scheme") == sch)["num"])
        pp, rr, _ = _prf(pr, t)
        per[sch] = {"true": len(t), "pred": len(pr), "precision": pp, "recall": rr}
    res["per_scheme"] = per
    # оценка как классификатор: порог 50 (пометить) и 80 (проверка)
    sc = dict(zip(scores["num"], scores["score"]))
    for thr in (50, 80):
        pr = {n for n, v in sc.items() if v >= thr}
        pp, rr, _ = _prf(pr, true)
        res[f"score>={thr}"] = {"precision": pp, "recall": rr, "legit_flagged": len(pr & legit_true)}
    if model is not None:
        from .ml import _mat
        prob = model.predict_proba(_mat(f))[:, 1]
        mp = {n for n, v in zip(f["num"].to_list(), prob) if v >= 0.5}
        pp, rr, _ = _prf(mp, true)
        res["ml_trained_on_real"] = {"precision": pp, "recall": rr, "legit_flagged": len(mp & legit_true),
                                     "missed_by_rules_found_by_ml": len((mp & true) - pred)}
    return res


def run_all(out: str = DEFAULT_OUT, cfg: dict | None = None, scenarios: dict | None = None) -> dict:
    cfg = cfg or load_config()
    from .ml import load_model
    model = load_model(out)
    results = []
    for name, params in (scenarios or SCENARIOS).items():
        t = time.time()
        r = evaluate_scenario(name, params, cfg, model, seed=params.get("seed", 7))
        r["seconds"] = round(time.time() - t, 1)
        results.append(r)
        print(f"  {name:12s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"ловушки как фрод: {r['legit_hit_as_fraud']}", flush=True)
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as fo:
        json.dump(rep, fo, ensure_ascii=False, indent=2)
    return rep
