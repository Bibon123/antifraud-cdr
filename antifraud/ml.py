"""CatBoost поверх правил: независимое «второе мнение» и поиск пропущенных номеров.

Обучение на псевдо-метках: 1 — номера, найденные детекторами; 0 — легальные
массовые источники (трудные негативы) + выборка обычных абонентов, в том числе
все активные. Модель видит только поведенческие признаки (без выводов детекторов),
поэтому её согласие с правилами — независимая проверка. Вклады — SHAP (встроены в CatBoost).

Если CatBoost недоступен (например, для новой версии Python ещё нет сборки), используется
логистическая регрессия scikit-learn: её вклады точные — коэффициент × нормированное значение признака.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import ML_FEATURES

try:
    from catboost import CatBoostClassifier, Pool
    HAVE_CATBOOST = True
except ImportError:                                  # pragma: no cover - зависит от окружения
    HAVE_CATBOOST = False


class LinearModel:
    """Запасная модель: логистическая регрессия на log1p-признаках с точными вкладами."""
    name = "LogisticRegression"

    def __init__(self, seed=42):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def _prep(self, X, fit=False):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Z = np.sign(X) * np.log1p(np.abs(X))
        if fit:
            self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        return (Z - self.mu) / self.sd

    def fit(self, X, y):
        self.lr.fit(self._prep(X, fit=True), y); return self

    def predict_proba(self, X):
        return self.lr.predict_proba(self._prep(X))

    def contributions(self, X):
        return self._prep(X) * self.lr.coef_[0]

    def importance(self):
        a = np.abs(self.lr.coef_[0]); return a / a.sum() * 100

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)


def make_model(params: dict):
    if HAVE_CATBOOST:
        m = CatBoostClassifier(**params); m.name = "CatBoost"
        return m
    return LinearModel(params.get("random_seed", 42))


def model_importance(model):
    return model.get_feature_importance() if HAVE_CATBOOST and isinstance(model, CatBoostClassifier) else model.importance()


def model_contributions(model, X):
    if HAVE_CATBOOST and isinstance(model, CatBoostClassifier):
        return model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]
    return model.contributions(X)


def save_model(model, out_dir: str) -> str:
    import os
    if HAVE_CATBOOST and isinstance(model, CatBoostClassifier):
        p = os.path.join(out_dir, "model.cbm"); model.save_model(p)
    else:
        p = os.path.join(out_dir, "model.pkl"); model.save(p)
    return p


def load_model(out_dir: str):
    """Модель из папки результатов (или None)."""
    import os
    import pickle
    cbm, pkl = os.path.join(out_dir, "model.cbm"), os.path.join(out_dir, "model.pkl")
    if os.path.exists(cbm) and HAVE_CATBOOST:
        m = CatBoostClassifier(); m.load_model(cbm); return m
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return None

FEATURE_RU = {
    "n_out": "исходящих вызовов", "uniq_share": "доля уникальных собеседников", "sh0": "доля нулевых вызовов",
    "sh02": "доля сбросов 0–2 с", "sh4_25": "доля вызовов 4–25 с", "sh8_45": "доля вызовов 8–45 с",
    "sh_long": "доля звонков ≥7 мин", "dur_med": "медианная длительность", "dur_mean": "средняя длительность",
    "span_min": "период активности, мин", "n_hours": "часов суток с активностью", "n_days": "дней активности",
    "sh_night": "доля ночных вызовов", "sh_work": "доля вызовов в рабочие часы", "overlap_share": "доля одновременных вызовов",
    "sh_backtoback": "доля вызовов «встык»", "gap_med": "медианная пауза между вызовами", "n_imsi": "число IMSI",
    "fwd_share_a": "доля переадресаций (как A)", "sh_repeat_b": "доля постоянных собеседников",
    "sh_seq": "доля вызовов на соседние номера", "peak_1h": "пик вызовов за час", "peak_5m": "пик вызовов за 5 мин",
    "n_dest_ops": "число операторов собеседников", "top3_hour_share": "концентрация в 3 часах суток",
    "n_in": "входящих вызовов", "ua_in": "уникальных звонивших", "in_out_ratio": "входящие / исходящие",
    "in_sh02": "доля коротких входящих", "n_fwd_as_b": "переадресаций через номер", "n_as_c": "раз как номер C",
}


def _xy(f: pl.DataFrame, hits: pl.DataFrame, legit: pl.DataFrame, seed: int, n_bg: int):
    pos = f.join(hits.select("num"), on="num", how="semi").with_columns(pl.lit(1).alias("y"), pl.lit("fraud").alias("src"))
    neg_l = (f.join(legit.select("num"), on="num", how="semi").join(hits.select("num"), on="num", how="anti")
             .with_columns(pl.lit(0).alias("y"), pl.lit("legit").alias("src")))
    bg = f.join(hits.select("num"), on="num", how="anti").join(legit.select("num"), on="num", how="anti")
    active = bg.filter(pl.col("n_out") >= 20)
    rest = bg.filter(pl.col("n_out") < 20)
    rest = rest.sort("num").sample(n=min(n_bg, rest.height), seed=seed)
    neg_b = pl.concat([active, rest]).with_columns(pl.lit(0).alias("y"), pl.lit("background").alias("src"))
    # порядок строк после группировок не гарантирован — сортируем, чтобы фолды и модель воспроизводились
    return pl.concat([pos, neg_l, neg_b], how="diagonal_relaxed").sort("num")


def _mat(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float64).fill_null(np.nan) for c in ML_FEATURES]).to_numpy()


def train_and_explain(f: pl.DataFrame, hits: pl.DataFrame, legit: pl.DataFrame, seed: int = 42,
                      n_bg: int = 80000, iterations: int = 300, folds: int = 5):
    data = _xy(f, hits, legit, seed, n_bg)
    X, y = _mat(data), data["y"].to_numpy()
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos < folds or n_neg < folds:
        raise ValueError(f"мало данных для обучения: {n_pos} примеров фрода и {n_neg} обычных (нужно хотя бы по {folds})")
    params = dict(iterations=iterations, depth=6, learning_rate=0.1, loss_function="Logloss",
                  random_seed=seed, verbose=False, auto_class_weights="Balanced", thread_count=-1,
                  allow_writing_files=False)
    # out-of-fold оценка качества
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = make_model(params); m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    src = data["src"].to_numpy()
    metrics = {
        "train_rows": int(len(y)), "positives": int(y.sum()),
        "roc_auc_oof": float(roc_auc_score(y, oof)), "pr_auc_oof": float(average_precision_score(y, oof)),
        "recall_at_0.5": float((oof[y == 1] >= 0.5).mean()),
        "fp_legit_at_0.5": int(((oof >= 0.5) & (src == "legit")).sum()),
        "fp_background_at_0.5": int(((oof >= 0.5) & (src == "background")).sum()),
        "legit_max_prob": float(oof[src == "legit"].max()) if (src == "legit").any() else None,
    }
    model = make_model(params); model.fit(X, y)
    metrics["model"] = "CatBoost" if HAVE_CATBOOST else "LogisticRegression (CatBoost недоступен)"
    # вероятность для всех номеров
    Xa = _mat(f)
    prob = model.predict_proba(Xa)[:, 1]
    allp = f.select("num").with_columns(pl.Series("ml_prob", prob))
    # OOF-вероятности для размеченных — честнее, чем предсказание на обучении
    allp = allp.join(data.select("num", pl.Series("oof", oof)), on="num", how="left").with_columns(
        pl.coalesce("oof", "ml_prob").alias("ml_prob")).drop("oof")
    imp = sorted(zip(ML_FEATURES, model_importance(model)), key=lambda t: -t[1])
    metrics["feature_importance"] = [{"feature": k, "ru": FEATURE_RU.get(k, k), "importance": round(float(v), 2)} for k, v in imp]
    return model, allp, metrics


def shap_for(model, f: pl.DataFrame, nums: list[str], top: int = 6) -> pl.DataFrame:
    """SHAP-вклады (в лог-шансах) для набора номеров: num, ml_factors (список словарей)."""
    sub = f.filter(pl.col("num").is_in(nums))
    if sub.height == 0:
        return pl.DataFrame(schema={"num": pl.Utf8, "ml_factors": pl.List(pl.Struct({"feature": pl.Utf8, "ru": pl.Utf8,
                                                                                          "value": pl.Float64, "shap": pl.Float64}))})
    X = _mat(sub)
    sv = model_contributions(model, X)
    out = []
    for i, num in enumerate(sub["num"].to_list()):
        idx = np.argsort(-np.abs(sv[i]))[:top]
        out.append({"num": num, "ml_factors": [
            {"feature": ML_FEATURES[j], "ru": FEATURE_RU.get(ML_FEATURES[j], ML_FEATURES[j]),
             "value": None if np.isnan(X[i, j]) else float(X[i, j]), "shap": float(sv[i, j])} for j in idx]})
    return pl.DataFrame(out)


def review_candidates(allp: pl.DataFrame, f: pl.DataFrame, hits: pl.DataFrame, legit: pl.DataFrame,
                      min_prob: float = 0.9, min_activity: int = 10) -> pl.DataFrame:
    """Номера, которые модель считает похожими на фрод, а правила — нет. Только на ручную проверку."""
    return (allp.filter(pl.col("ml_prob") >= min_prob)
            .join(hits.select("num"), on="num", how="anti").join(legit.select("num"), on="num", how="anti")
            .join(f.select("num", "n_out", "n_in"), on="num")
            .filter((pl.col("n_out") + pl.col("n_in")) >= min_activity)
            .sort("ml_prob", descending=True))
