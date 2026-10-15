"""CatBoost, обученный на размеченной синтетике, и его проверка на реальных CDR.

Зачем. Правила дают ответ, но «правильного ответа» по реальным данным нет. Модель здесь
обучается без единого вывода правил:

  * метки — истинная разметка синтетики (7 схем, 5 типов легальных ловушек, обычные абоненты);
  * синтетика генерируется с разбросом параметров (тише/громче, короче/длиннее, ошибки
    формата) — модель не должна выучить один набор чисел;
  * вариант «синтетика + реальный фон»: к синтетике добавляются случайные номера из реальных
    CDR с меткой «обычный» (мошенников среди них ~0,2%). Это учит модель, как выглядит
    настоящий фон, и по-прежнему не использует ответы правил. Реальные номера делятся на две
    половины по последней цифре: модель, обученная на фоне одной половины, оценивает другую.

Затем ответ модели на реальных данных сравнивается с ответом правил. Совпадение двух
независимых методов — проверка, которой нет при обучении только на синтетике.
Вариант «только синтетика» считается для сравнения: так обычно делают, и он показывает,
насколько точность на синтетике переносится на реальные данные.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import polars as pl

from .features import ML_FEATURES, build_features
from .ingest import parse
from .synth import Synth

try:
    from catboost import CatBoostClassifier, Pool
    HAVE_CATBOOST = True
except ImportError:                                  # pragma: no cover
    HAVE_CATBOOST = False

CLASSES = ["normal", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "D1", "D2", "D3", "D4", "D5"]
FRAUD = CLASSES[1:8]
LEGIT = CLASSES[8:]
CLASS_RU = {"normal": "обычный абонент", "S1": "Wangiri", "S2": "массовый обзвон (робот)", "S3": "SIM-box",
            "S4": "клон SIM", "S5": "цепочка переадресаций", "S6": "карусель", "S7": "вишинг",
            "D1": "колл-центр", "D2": "банковские оповещения", "D3": "ночные диспетчеры", "D4": "горячая линия",
            "D5": "оповещение ЖКХ/МЧС"}

# отложенные сценарии синтетики: на них модель не обучается (другие зёрна генератора)
HOLDOUT = {
    "base": {"seed": 7, "scale": 1.0, "dur_shift": 1.0, "error_rate": 0.0003, "desc": "параметры как в описании кейса"},
    "stress_x0.3": {"seed": 101, "scale": 0.3, "dur_shift": 1.0, "error_rate": 0.0003, "desc": "мошенники втрое тише"},
    "stress_mix": {"seed": 202, "scale": 0.7, "dur_shift": 0.8, "error_rate": 0.003,
                   "desc": "−30% вызовов, −20% длительности, 0,3% ошибок"},
}


def fold_of(num: str) -> int:
    """Половина реальных номеров (по последней цифре) — одинаково в пайплайне и в сервисе."""
    return int(num[-1]) % 2 if num and num[-1].isdigit() else 0


def _mat(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64).fill_null(np.nan) for c in ML_FEATURES]).to_numpy()


def labelled_synth(cfg: dict, seed: int, scale: float = 1.0, dur_shift: float = 1.0, error_rate: float = 0.0003) -> pl.DataFrame:
    s = Synth(seed=seed, scale=scale, dur_shift=dur_shift, error_rate=error_rate).build()
    df, _ = parse(s.sources(), cfg)
    f = build_features(df, cfg)
    lab = {n: v[0] for n, v in s.truth.items()} | dict(s.legit)
    return f.with_columns(pl.col("num").replace_strict(lab, default="normal").alias("cls"))


def training_set(cfg: dict, n_sets: int = 6, seed: int = 1000, normal_sample: int = 30000) -> pl.DataFrame:
    """Синтетика с разбросом параметров (domain randomization). Зёрна не пересекаются с HOLDOUT."""
    rng = random.Random(seed)
    parts = []
    for i in range(n_sets):
        parts.append(labelled_synth(cfg, seed + i, scale=rng.choice([0.4, 0.6, 0.8, 1.0, 1.2, 1.5]),
                                    dur_shift=rng.choice([0.8, 0.9, 1.0, 1.15, 1.3]), error_rate=0.002))
    # порядок строк после группировок не гарантирован — сортируем, чтобы выборка и модель воспроизводились
    t = pl.concat([p.sort("num") for p in parts], how="diagonal_relaxed")
    normal = t.filter(pl.col("cls") == "normal")
    return pl.concat([t.filter(pl.col("cls") != "normal"),
                      normal.sample(min(normal_sample, normal.height), seed=seed)], how="diagonal_relaxed")


def _fit(X, y, seed: int, iterations: int):
    m = CatBoostClassifier(iterations=iterations, depth=6, learning_rate=0.15, loss_function="MultiClass",
                           auto_class_weights="Balanced", random_seed=seed, verbose=False, thread_count=-1,
                           allow_writing_files=False, class_names=list(range(len(CLASSES))))
    m.fit(X, y)
    return m


def _y(df: pl.DataFrame) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx[c] for c in df["cls"].to_list()])


def predict(model, f: pl.DataFrame) -> pl.DataFrame:
    """num, synth_prob (сумма вероятностей схем S1–S7), synth_class (самый вероятный класс)."""
    if f.height == 0:
        return pl.DataFrame(schema={"num": pl.Utf8, "synth_prob": pl.Float64, "synth_class": pl.Utf8})
    P = model.predict_proba(_mat(f))
    return f.select("num").with_columns(pl.Series("synth_prob", P[:, 1:8].sum(1)),
                                        pl.Series("synth_class", np.array(CLASSES)[P.argmax(1)]))


def explain_one(model, frow: pl.DataFrame, top: int = 6) -> dict:
    """Вероятности классов и SHAP-вклады для самого вероятного класса (одна строка признаков)."""
    from .ml import FEATURE_RU
    X = _mat(frow)
    P = model.predict_proba(X)[0]
    k = int(P.argmax())
    sv = model.get_feature_importance(Pool(X), type="ShapValues")[0, k, :-1]
    idx = np.argsort(-np.abs(sv))[:top]
    probs = sorted(({"cls": c, "ru": CLASS_RU[c], "p": float(p)} for c, p in zip(CLASSES, P)), key=lambda d: -d["p"])
    return {"fraud_prob": float(P[1:8].sum()), "cls": CLASSES[k], "cls_ru": CLASS_RU[CLASSES[k]], "top": probs[:3],
            "factors": [{"feature": ML_FEATURES[j], "ru": FEATURE_RU.get(ML_FEATURES[j], ML_FEATURES[j]),
                         "value": None if np.isnan(X[0, j]) else float(X[0, j]), "shap": float(sv[j])} for j in idx]}


def _holdout_metrics(model, holdout: dict[str, pl.DataFrame], thr: float) -> list[dict]:
    res = []
    for name, f in holdout.items():
        p = predict(model, f).join(f.select("num", "cls"), on="num")
        isf = p["cls"].is_in(FRAUD).to_numpy(); isl = p["cls"].is_in(LEGIT).to_numpy()
        flag = (p["synth_prob"] >= thr).to_numpy()
        tp = int((flag & isf).sum())
        same = (p["synth_class"] == p["cls"]).to_numpy()
        res.append({"scenario": name, "desc": HOLDOUT[name]["desc"], "fraud": int(isf.sum()),
                    "precision": round(tp / flag.sum(), 4) if flag.sum() else 1.0,
                    "recall": round(tp / isf.sum(), 4) if isf.sum() else 1.0,
                    "legit_flagged": int((flag & isl).sum()), "legit_total": int(isl.sum()),
                    "scheme_accuracy": round(float(same[isf].mean()), 4) if isf.any() else None})
    return res


def _real_metrics(pred: pl.DataFrame, hits: pl.DataFrame, legit: pl.DataFrame, thr: float) -> dict:
    r = (pred.join(hits.select("num", "scheme", "role"), on="num", how="left")
         .join(legit.select("num", "legit"), on="num", how="left"))
    fl = r.filter(pl.col("synth_prob") >= thr)
    in_rules = fl.filter(pl.col("scheme").is_not_null())
    lg = fl.filter(pl.col("legit").is_not_null())
    other = fl.filter(pl.col("scheme").is_null() & pl.col("legit").is_null())
    h = r.filter(pl.col("scheme").is_not_null())
    per_role = (h.group_by("role").agg(pl.len().alias("n"), (pl.col("synth_prob") >= thr).sum().alias("found"),
                                       (pl.col("synth_class") == pl.col("scheme")).sum().alias("same_scheme"))
                .sort("role").to_dicts())
    return {"threshold": thr, "flagged": fl.height,
            "rules_total": hits.height, "rules_confirmed": in_rules.height,
            "rules_confirmed_share": round(in_rules.height / hits.height, 4) if hits.height else None,
            "same_scheme_share": round(float((in_rules["synth_class"] == in_rules["scheme"]).mean()), 4) if in_rules.height else None,
            "legit_total": legit.height, "legit_flagged": lg.height,
            "legit_flagged_by_class": lg.group_by("legit").len().sort("legit").to_dicts(),
            "other_flagged": other.height,
            "other_by_class": other.group_by("synth_class").len().sort("len", descending=True).to_dicts(),
            "precision_vs_rules": round(in_rules.height / fl.height, 4) if fl.height else None,
            "per_role": per_role}


def run_synth_model(f_real: pl.DataFrame, hits: pl.DataFrame, legit: pl.DataFrame, cfg: dict, out: str,
                    log=print) -> tuple[pl.DataFrame, dict]:
    """Обучает обе версии, сохраняет модели, возвращает оценки по реальным номерам и отчёт."""
    if not HAVE_CATBOOST:
        raise ValueError("CatBoost не установлен — модель на синтетике пропущена")
    m = cfg.get("ml_synth", {})
    seed, iters = m.get("seed", 42), m.get("iterations", 300)
    thr, thr_hi = m.get("threshold", 0.5), m.get("threshold_high", 0.8)
    T = time.time()
    tr = training_set(cfg, m.get("train_sets", 6), m.get("train_seed", 1000), m.get("synth_normal_sample", 30000))
    holdout = {k: labelled_synth(cfg, v["seed"], v["scale"], v["dur_shift"], v["error_rate"]) for k, v in HOLDOUT.items()}
    log(f"      синтетика: {tr.height:,} номеров для обучения, из них схем {tr.filter(pl.col('cls').is_in(FRAUD)).height}, "
        f"ловушек {tr.filter(pl.col('cls').is_in(LEGIT)).height} ({time.time() - T:.0f} с)")

    # вариант 1: только синтетика
    t = time.time()
    m_syn = _fit(_mat(tr), _y(tr), seed, iters)
    pred_syn = predict(m_syn, f_real)
    rep_syn = {"holdout": _holdout_metrics(m_syn, holdout, thr),
               "real": [_real_metrics(pred_syn, hits, legit, x) for x in (thr, thr_hi)]}
    log(f"      только синтетика: на реальных данных отмечено {rep_syn['real'][0]['flagged']}, "
        f"из них легальных {rep_syn['real'][0]['legit_flagged']} ({time.time() - t:.0f} с)")

    # вариант 2: синтетика + реальный фон, две половины номеров
    t = time.time()
    n_bg = m.get("real_background_sample", 60000)
    real = f_real.with_columns(pl.col("num").map_elements(fold_of, return_dtype=pl.Int8).alias("_fold"))
    preds, models = [], []
    for k in (0, 1):
        neg = real.filter(pl.col("_fold") != k).sort("num")
        neg = neg.sample(min(n_bg, neg.height), seed=seed + k).drop("_fold").with_columns(pl.lit("normal").alias("cls"))
        d = pl.concat([tr, neg], how="diagonal_relaxed")
        mk = _fit(_mat(d), _y(d), seed, iters)
        mk.save_model(os.path.join(out, f"model_synth_{k}.cbm"))
        models.append(mk)
        preds.append(predict(mk, real.filter(pl.col("_fold") == k).drop("_fold")))
    pred = pl.concat(preds)
    rep = {"holdout": _holdout_metrics(models[0], holdout, thr),
           "real": [_real_metrics(pred, hits, legit, x) for x in (thr, thr_hi)]}
    log(f"      синтетика + реальный фон: подтверждено {rep['real'][0]['rules_confirmed']} из {hits.height} номеров правил, "
        f"легальных отмечено {rep['real'][0]['legit_flagged']} ({time.time() - t:.0f} с)")

    imp = sorted(zip(ML_FEATURES, models[0].get_feature_importance()), key=lambda x: -x[1])
    from .ml import FEATURE_RU
    report = {
        "model": "CatBoost MultiClass", "classes": CLASSES, "classes_ru": CLASS_RU,
        "train_rows": tr.height, "train_by_class": tr.group_by("cls").len().sort("cls").to_dicts(),
        "train_sets": m.get("train_sets", 6), "real_background_sample": n_bg,
        "threshold": thr, "threshold_high": thr_hi,
        "synth_only": rep_syn, "synth_plus_real_bg": rep,
        "feature_importance": [{"feature": k, "ru": FEATURE_RU.get(k, k), "importance": round(float(v), 2)} for k, v in imp[:12]],
        "seconds": round(time.time() - T, 1),
    }
    with open(os.path.join(out, "ml_synth.json"), "w", encoding="utf-8") as fo:
        json.dump(report, fo, ensure_ascii=False, indent=2)
    return pred, report


def load_synth_models(out: str):
    """Модели обеих половин (или None), для оценки в момент запроса."""
    if not HAVE_CATBOOST:
        return None
    ps = [os.path.join(out, f"model_synth_{k}.cbm") for k in (0, 1)]
    if not all(os.path.exists(p) for p in ps):
        return None
    ms = []
    for p in ps:
        m = CatBoostClassifier(); m.load_model(p); ms.append(m)
    return ms
