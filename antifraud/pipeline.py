"""Полный прогон: CDR -> очистка -> признаки -> детекторы -> оценка -> ML -> файлы ответа.

Все артефакты пишутся в папку out/ и затем читаются сервисом (app/main.py).
"""
from __future__ import annotations

import json
import os
import time

import polars as pl

from .config import DEFAULT_OUT, load_config
from .detectors import LEGIT_RU, ROLE_RU, SCHEME_RU, run_detectors
from .features import build_features
from .ingest import load_source, quarantine_frame
from .scoring import FACTOR_KEYS, score_numbers

AUX_ROLES = {"premium_C", "spoofed_city"}   # инфраструктура схемы, а не сами звонящие


def log(msg):
    print(msg, flush=True)


def _synth_brief(rep: dict | None) -> dict | None:
    """Главные цифры модели на синтетике для сводки."""
    if not rep:
        return None
    a, b = rep["synth_only"], rep["synth_plus_real_bg"]
    hold = {h["scenario"]: {k: h[k] for k in ("precision", "recall", "legit_flagged")} for h in b["holdout"]}
    return {"threshold": rep["threshold"], "holdout": hold,
            "real_synth_only": {k: a["real"][0][k] for k in ("flagged", "rules_confirmed", "legit_flagged", "other_flagged")},
            "real": {k: b["real"][0][k] for k in ("flagged", "rules_confirmed", "rules_confirmed_share", "same_scheme_share",
                                                    "legit_flagged", "other_flagged", "precision_vs_rules")},
            "real_high": {k: b["real"][1][k] for k in ("threshold", "flagged", "rules_confirmed", "legit_flagged", "other_flagged")},
            "review_count": rep.get("review_count")}


def build_groups(hits: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    g = hits.join(scores.select("num", "score"), on="num", how="left")
    return (g.group_by("group_id", "scheme").agg(
        pl.len().alias("size"), pl.col("num").sort().alias("members"), pl.col("role").unique().sort().alias("roles"),
        pl.col("score").max().alias("max_score"), pl.col("evidence").first().alias("example"))
        .with_columns(pl.col("scheme").replace_strict(SCHEME_RU, default="").alias("scheme_ru"))
        .sort("scheme", "group_id"))


def write_answers(hits: pl.DataFrame, out: str) -> dict:
    h = hits.sort("scheme", "role", "num")
    bad = h.filter(~pl.col("num").str.contains(r"^\d{10}$"))
    assert bad.height == 0, f"некорректные номера в ответе: {bad['num'].to_list()[:5]}"

    def write(path, df, title):
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(f"# {title}\n# номеров: {df.height}\n")
            for s, n in df.group_by("scheme").len().sort("scheme").iter_rows():
                fo.write(f"# {s} {SCHEME_RU[s]}: {n}\n")
            for num, s in df.select("num", "scheme").iter_rows():
                fo.write(f"{num};{s}\n")

    strict = h.filter(~pl.col("role").is_in(list(AUX_ROLES)))
    write(os.path.join(out, "answer.txt"), h, "Антифрод CDR — полный ответ (все номера схем)")
    write(os.path.join(out, "answer_strict.txt"), strict,
          "Антифрод CDR — строгий ответ (без платных C Wangiri и подменённых городских номеров вишинга)")
    h.with_columns(pl.col("role").replace_strict(ROLE_RU, default="").alias("role_ru")).write_csv(
        os.path.join(out, "evidence.csv"), separator=";")
    return {"full": h.height, "strict": strict.height}


def run(src: str, out: str = DEFAULT_OUT, cfg: dict | None = None, with_ml: bool | None = None,
        with_realtime: bool = True) -> dict:
    cfg = cfg or load_config()
    os.makedirs(out, exist_ok=True)
    # убираем результаты прошлого прогона, которые этот прогон может не перезаписать
    for stale in ("ml_metrics.json", "shap.parquet", "model.cbm", "model.pkl", "realtime.json", "review.parquet",
                  "ml_synth.json", "model_synth_0.cbm", "model_synth_1.cbm", "synth_pred.parquet"):
        p = os.path.join(out, stale)
        if os.path.exists(p):
            os.remove(p)
    T = time.time(); timings = {}

    t = time.time(); log("[1/7] Загрузка и очистка CDR…")
    df, rep = load_source(src, cfg)
    df.write_parquet(os.path.join(out, "clean.parquet"))
    quarantine_frame(rep).write_csv(os.path.join(out, "quarantine.csv"), separator=";")
    ingest = rep.as_dict() | {"rows_clean": df.height}
    if df.height == 0:
        raise ValueError(f"в источнике нет ни одной пригодной записи CDR (строк: {rep.lines}, все в карантине). "
                         f"Причины — в {os.path.join(out, 'quarantine.csv')}")
    timings["ingest"] = time.time() - t
    log(f"      файлов {rep.files}, строк {rep.lines:,}, в работе {df.height:,}, карантин {len(rep.quarantine)}")

    t = time.time(); log("[2/7] Признаки по номерам…")
    f = build_features(df, cfg)
    f.write_parquet(os.path.join(out, "features.parquet"))
    timings["features"] = time.time() - t
    log(f"      номеров: {f.height:,}")

    t = time.time(); log("[3/7] Детекторы схем S1–S7 и легальных источников D1–D5…")
    hits, rel, legit = run_detectors(df, f, cfg)
    conflicts = hits.join(legit, on="num")
    legit = legit.join(hits.select("num"), on="num", how="anti")
    hits.write_parquet(os.path.join(out, "hits.parquet")); rel.write_parquet(os.path.join(out, "rel.parquet"))
    legit.write_parquet(os.path.join(out, "legit.parquet"))
    timings["detectors"] = time.time() - t
    log(f"      схемы: {hits.height}, легальные: {legit.height}, конфликтов: {conflicts.height}")

    t = time.time(); log("[4/7] Оценка 0–100 и политика действий…")
    scores = score_numbers(f, hits, rel, legit, cfg)
    timings["scoring"] = time.time() - t

    ml_metrics, review = None, pl.DataFrame(schema={"num": pl.Utf8, "ml_prob": pl.Float64})
    use_ml = cfg.get("ml", {}).get("enabled", True) if with_ml is None else with_ml
    if use_ml:
        from .ml import HAVE_CATBOOST, review_candidates, save_model, shap_for, train_and_explain
        t = time.time(); log("[5/7] " + ("CatBoost + SHAP" if HAVE_CATBOOST else
                                          "Логистическая регрессия (CatBoost не установлен)") + " на ответах правил — второе мнение…")
        m = cfg["ml"]
        try:
            model, allp, ml_metrics = train_and_explain(f, hits, legit, seed=m["seed"], n_bg=m["background_sample"],
                                                        iterations=m["iterations"], folds=m["folds"])
        except ValueError as e:
            log(f"      модель пропущена: {e}")
            use_ml = False
    if use_ml:
        save_model(model, out)
        review = review_candidates(allp, f, hits, legit, m["review_min_prob"], m["review_min_activity"])
        scores = scores.join(allp, on="num", how="left")
        show = set(hits["num"]) | set(review["num"]) | set(legit["num"])
        shap_for(model, f, sorted(show)).write_parquet(os.path.join(out, "shap.parquet"))
        ml_metrics["review_count"] = review.height
        agr = allp.join(hits.select("num"), on="num")["ml_prob"]
        ml_metrics["agreement_with_rules"] = float((agr >= 0.5).mean()) if agr.len() else None
        with open(os.path.join(out, "ml_metrics.json"), "w", encoding="utf-8") as fo:
            json.dump(ml_metrics, fo, ensure_ascii=False, indent=2)
        timings["ml"] = time.time() - t
        log(f"      ROC-AUC (out-of-fold) {ml_metrics['roc_auc_oof']:.4f}, на проверку аналитику: {review.height}")
    else:
        scores = scores.with_columns(pl.lit(None, dtype=pl.Float64).alias("ml_prob"))
    review = review.select("num", "ml_prob").with_columns(pl.lit("модель на ответах правил").alias("review_source"))

    # независимая модель: обучена на размеченной синтетике, ответов правил не видела
    synth_report = None
    use_synth = cfg.get("ml_synth", {}).get("enabled", True) if with_ml is None else with_ml
    if use_synth:
        from .ml_synth import run_synth_model
        t = time.time(); log("[6/7] CatBoost на размеченной синтетике — независимая проверка ответа на реальных данных…")
        try:
            spred, synth_report = run_synth_model(f, hits, legit, cfg, out, log=log)
        except ValueError as e:
            log(f"      модель на синтетике пропущена: {e}")
            spred = None
        if spred is not None:
            spred.write_parquet(os.path.join(out, "synth_pred.parquet"))
            scores = scores.join(spred, on="num", how="left")
            ms = cfg["ml_synth"]
            extra = (spred.filter(pl.col("synth_prob") >= ms["review_min_prob"])
                     .join(hits.select("num"), on="num", how="anti").join(legit.select("num"), on="num", how="anti")
                     .join(f.select("num", "n_out", "n_in"), on="num")
                     .filter((pl.col("n_out") + pl.col("n_in")) >= ms["review_min_activity"])
                     .join(review.select("num"), on="num", how="anti"))
            review = pl.concat([review, extra.select("num").join(scores.select("num", "ml_prob"), on="num", how="left")
                                .with_columns(pl.lit("модель на синтетике").alias("review_source"))], how="vertical_relaxed")
            synth_report["review_count"] = extra.height
            with open(os.path.join(out, "ml_synth.json"), "w", encoding="utf-8") as fo:
                json.dump(synth_report, fo, ensure_ascii=False, indent=2)
            timings["ml_synth"] = time.time() - t
            r0 = synth_report["synth_plus_real_bg"]["real"][0]
            log(f"      подтверждено {r0['rules_confirmed']} из {r0['rules_total']} номеров ответа, "
                f"легальных отмечено {r0['legit_flagged']}, на проверку аналитику: {extra.height}")
    if "synth_prob" not in scores.columns:
        scores = scores.with_columns(pl.lit(None, dtype=pl.Float64).alias("synth_prob"),
                                     pl.lit(None, dtype=pl.Utf8).alias("synth_class"))
    if use_ml and review.height:
        # SHAP модели на правилах — и для кандидатов от модели на синтетике
        shap_for(model, f, sorted(set(hits["num"]) | set(review["num"]) | set(legit["num"]))).write_parquet(
            os.path.join(out, "shap.parquet"))
    review = review.join(scores.select("num", "synth_prob", "synth_class"), on="num", how="left")
    review.write_parquet(os.path.join(out, "review.parquet"))

    # сохраняем оценки: полные — для «интересных» номеров, краткие — для всех
    flagged = scores.filter((pl.col("score") > 0) | pl.col("role").is_not_null() | pl.col("legit").is_not_null()
                            | pl.col("num").is_in(review["num"].implode()))
    keep = (["num", "op", "score", "score_raw", "n_strong", "scheme", "scheme_ru", "role", "role_ru", "group_id", "evidence",
             "legit", "legit_ru", "legit_why", "recommendation", "autoblock", "autoblock_reason", "whitelisted", "ml_prob",
             "synth_prob", "synth_class"]
            + [f"pts_{k}" for k in FACTOR_KEYS] + [f"txt_{k}" for k in FACTOR_KEYS])
    flagged = flagged.select(keep)
    flagged.write_parquet(os.path.join(out, "scores_flagged.parquet"))
    scores.select("num", "score", "ml_prob", "synth_prob", "synth_class").write_parquet(os.path.join(out, "scores_all.parquet"))

    groups = build_groups(hits, scores)
    groups.write_parquet(os.path.join(out, "groups.parquet"))
    answers = write_answers(hits, out)

    rt = None
    if with_realtime:
        t = time.time(); log("[7/7] Прогон «в реальном времени»: время до обнаружения…")
        from .realtime import time_to_detect
        rt = time_to_detect(df, hits, cfg)
        if rt.get("economics"):
            log("      " + rt["economics"]["headline"])
        with open(os.path.join(out, "realtime.json"), "w", encoding="utf-8") as fo:
            json.dump(rt, fo, ensure_ascii=False, indent=2, default=str)
        timings["realtime"] = time.time() - t

    cls = scores.with_columns(pl.coalesce(pl.col("role"), pl.col("legit"), pl.lit("background")).alias("cls"))
    dist = (cls.group_by("cls").agg(pl.len().alias("n"), pl.col("score").min().alias("min"),
                                    pl.col("score").median().alias("median"), pl.col("score").max().alias("max"),
                                    pl.col("autoblock").sum().alias("autoblock"))
            .sort("cls").to_dicts())
    summary = {
        "source": os.path.abspath(src), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "period": [str(df["ts"].min()), str(df["ts"].max())],
        "ingest": ingest, "numbers": f.height,
        "schemes": hits.group_by("scheme", "role").agg(pl.len().alias("n"), pl.col("group_id").n_unique().alias("groups"))
        .sort("scheme", "role").with_columns(pl.col("scheme").replace_strict(SCHEME_RU, default="").alias("scheme_ru"),
                                             pl.col("role").replace_strict(ROLE_RU, default="").alias("role_ru")).to_dicts(),
        "legit": legit.group_by("legit").len().sort("legit").with_columns(
            pl.col("legit").replace_strict(LEGIT_RU, default="").alias("legit_ru")).to_dicts(),
        "conflicts_rules_vs_legit": conflicts.height,
        "answers": answers, "score_distribution": dist,
        "autoblock_total": int(scores["autoblock"].sum()),
        "ml": {k: v for k, v in (ml_metrics or {}).items() if k != "feature_importance"} or None,
        "ml_synth": _synth_brief(synth_report),
        "review": review.select("num", "ml_prob", "synth_prob", "synth_class", "review_source").to_dicts(),
        "economics": {k: v for k, v in (rt or {}).get("economics", {}).items() if k != "rows"} or None,
        "realtime_headline": (rt or {}).get("headline"),
        "timings_sec": {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.time() - T, 1)},
        "thresholds": {k: v for k, v in cfg.items() if k not in ("data",)},
    }
    # операционная база: PostgreSQL (DATABASE_URL) или SQLite-файл в out/
    from .db import db_url, describe, save_run
    summary["database"] = describe(db_url(out))
    try:
        summary["run_id"] = save_run(out, summary, flagged, groups, review)
        log(f"      результаты записаны в базу: {summary['database']}, прогон №{summary['run_id']}")
    except Exception as e:                                      # база недоступна — файлы ответа всё равно есть
        summary["database_error"] = str(e).splitlines()[0][:300]
        log(f"      база недоступна ({summary['database']}): {summary['database_error']}")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=2, default=str)
    log(f"Готово за {time.time() - T:.0f} с. Ответ: {answers['full']} номеров (строгий {answers['strict']}). Папка: {out}")
    return summary
