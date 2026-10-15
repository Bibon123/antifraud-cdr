"""Веб-сервис антифрода: API + страница консоли аналитика.

Читает результаты пайплайна из папки out/ (переменная ANTIFRAUD_OUT).
Запуск: python -m antifraud serve   (или uvicorn app.main:app)
"""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sqlalchemy import func, select

from antifraud import db
from antifraud.config import DEFAULT_OUT, load_config
from antifraud.detectors import LEGIT_RU, ROLE_RU, SCHEME_RU
from antifraud.ml import FEATURE_RU, _mat, load_model, model_contributions
from antifraud.ml_synth import CLASS_RU, explain_one, fold_of, load_synth_models
from antifraud.scoring import explain, is_emergency, score_numbers

OUT = os.environ.get("ANTIFRAUD_OUT", DEFAULT_OUT)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = load_config()

@asynccontextmanager
async def lifespan(_app):
    try:                      # прогрев: данные и модели загружаются до первого запроса
        S()
    except Exception:         # pragma: no cover - нет результатов: страница подскажет, что запустить
        pass
    yield


app = FastAPI(title="Антифрод по метаданным вызовов", version="1.1", lifespan=lifespan,
              description="Оценка номеров и вызовов 0–100 с объяснением факторов, без содержимого разговоров.")


def _json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pq(name, columns=None):
    p = os.path.join(OUT, name)
    return pl.read_parquet(p, columns=columns) if os.path.exists(p) else None


class Store:
    def __init__(self):
        self.summary = _json("summary.json") or {}
        self.eval = _json("eval.json")
        self.realtime = _json("realtime.json")
        self.ml = _json("ml_metrics.json")
        self.ml_synth = _json("ml_synth.json")
        self.scores = _pq("scores_flagged.parquet")
        self.features = _pq("features.parquet")
        self.feat_idx = {n: i for i, n in enumerate(self.features["num"].to_list())} if self.features is not None else {}
        self.hits = _pq("hits.parquet")
        self.rel = _pq("rel.parquet")
        self.legit = _pq("legit.parquet")
        self.groups = _pq("groups.parquet")
        self.review = _pq("review.parquet")
        shap = _pq("shap.parquet")
        self.shap = {r["num"]: r["ml_factors"] for r in shap.iter_rows(named=True)} if shap is not None else {}
        self.calls = _pq("clean.parquet", ["type", "imsi", "a", "b", "ts", "dur", "c"])
        all_s = _pq("scores_all.parquet")
        self.ml_prob = dict(zip(all_s["num"], all_s["ml_prob"])) if all_s is not None else {}
        self.by_num = {r["num"]: r for r in self.scores.iter_rows(named=True)} if self.scores is not None else {}
        # модели для оценки в момент запроса
        self.model_rules = load_model(OUT)
        self.synth_models = load_synth_models(OUT)
        # операционная база; если недоступна — работаем по файлам out/
        self.db_name = db.describe(db.db_url(OUT))
        try:
            self.run = db.last_run(OUT)
            self.db_ok = self.run is not None
            self.db_error = None if self.db_ok else "в базе нет прогонов"
        except Exception as e:                                   # pragma: no cover - зависит от окружения
            self.run, self.db_ok, self.db_error = None, False, str(e).splitlines()[0][:200]


_STORE: dict = {}


def S() -> Store:
    """Результаты из out/; перечитываются, если пайплайн пересчитал их при работающем сервере."""
    p = os.path.join(OUT, "summary.json")
    mtime = os.path.getmtime(p) if os.path.exists(p) else 0
    if _STORE.get("mtime") != mtime:
        _STORE["store"], _STORE["mtime"] = Store(), mtime
    return _STORE["store"]


# ---------------- помощники ----------------
def mask(num: str, reveal: bool = False) -> str:
    if not num:
        return ""
    if reveal or len(num) != 10:
        return f"+7 {num[:3]} {num[3:6]}-{num[6:8]}-{num[8:]}" if len(num) == 10 else num
    return f"+7 {num[:3]} ***-**-{num[8:]}"


def op_name(num: str) -> str:
    if is_emergency(num, CFG):
        return "экстренная служба"
    codes = CFG["data"]["operator_codes"]
    return {"CITY": "городской", "ALFA": "Альфа", "BARB": "Барбетта", "BELL": "Bell-ka"}.get(codes.get(num[:3], ""), "неизвестный код")


def feat(num: str) -> pl.DataFrame:
    """Строка признаков номера (по индексу — быстрее, чем фильтр по 375 тыс. строк)."""
    s = S()
    i = s.feat_idx.get(num)
    return s.features.slice(i, 1) if i is not None else s.features.head(0)


def number_row(num: str) -> dict:
    """Строка оценки номера: из сохранённых или на лету (для любого номера)."""
    s = S()
    if s.db_ok:
        try:
            with db.engine(OUT).connect() as cx:
                d = cx.execute(select(db.numbers.c.data).where(db.numbers.c.num == num)).scalar_one_or_none()
            if d is not None:
                return d
        except Exception:                                        # pragma: no cover
            pass
    if num in s.by_num:
        return s.by_num[num]
    if s.features is None:
        raise HTTPException(404, "нет данных")
    f = feat(num)
    if f.height == 0:
        raise HTTPException(404, f"номер {num} не встречается в CDR")
    r = score_numbers(f, s.hits.filter(pl.col("num") == num), s.rel.filter(pl.col("num") == num),
                      s.legit.filter(pl.col("num") == num), CFG).row(0, named=True)
    r["ml_prob"] = s.ml_prob.get(num)
    return r


def group_row(gid: str) -> dict | None:
    s = S()
    if s.db_ok:
        try:
            with db.engine(OUT).connect() as cx:
                g = cx.execute(select(db.groups).where(db.groups.c.group_id == gid)).mappings().first()
            return dict(g) if g else None
        except Exception:                                        # pragma: no cover
            pass
    if s.groups is None:
        return None
    g = s.groups.filter(pl.col("group_id") == gid)
    return g.row(0, named=True) if g.height else None


def member_info(nums: list[str]) -> dict:
    s = S()
    if s.db_ok:
        try:
            with db.engine(OUT).connect() as cx:
                rows = cx.execute(select(db.numbers.c.num, db.numbers.c.role, db.numbers.c.score)
                                  .where(db.numbers.c.num.in_(nums))).mappings().all()
            return {r["num"]: dict(r) for r in rows}
        except Exception:                                        # pragma: no cover
            pass
    return {m: s.by_num.get(m, {}) for m in nums}


def live_ml(num: str, stored_prob: float | None = None) -> dict | None:
    """Обе модели считаются в момент запроса по признакам номера: вероятность, класс схемы и вклады SHAP."""
    s = S()
    if s.features is None or (s.model_rules is None and not s.synth_models):
        return None
    t0 = time.perf_counter()
    fr = feat(num)
    if fr.height == 0:
        return None
    res = {}
    if s.model_rules is not None:
        X = _mat(fr)
        p_live = float(s.model_rules.predict_proba(X)[0, 1])
        sv = model_contributions(s.model_rules, X)[0]
        idx = sorted(range(len(sv)), key=lambda j: -abs(sv[j]))[:6]
        from antifraud.features import ML_FEATURES
        res["rules_model"] = {
            # для номеров из обучения показываем оценку на отложенном фолде (честнее, чем на обучении)
            "prob": stored_prob if stored_prob is not None else p_live, "prob_live": p_live,
            "factors": [{"feature": ML_FEATURES[j], "ru": FEATURE_RU.get(ML_FEATURES[j], ML_FEATURES[j]),
                         "value": None if X[0, j] != X[0, j] else float(X[0, j]), "shap": float(sv[j])} for j in idx]}
    if s.synth_models:
        res["synth_model"] = explain_one(s.synth_models[fold_of(num)], fr)
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


def number_card(num: str, reveal: bool = False) -> dict:
    s = S()
    r = number_row(num)
    feats = feat(num)
    fr = feats.row(0, named=True) if feats.height else {}
    group = None
    if r.get("group_id"):
        g = group_row(r["group_id"])
        if g:
            info = member_info(g["members"][:40])
            group = {"id": g["group_id"], "scheme_ru": g["scheme_ru"], "size": g["size"],
                     "members": [{"num": m, "masked": mask(m, reveal), "role_ru": ROLE_RU.get(info.get(m, {}).get("role"), ""),
                                  "score": info.get(m, {}).get("score")} for m in g["members"][:40]]}
    ml = live_ml(num, r.get("ml_prob"))
    return {
        "num": num, "masked": mask(num, reveal), "operator": op_name(num),
        "score": int(r["score"]), "scheme": r.get("scheme"), "scheme_ru": r.get("scheme_ru"),
        "role": r.get("role"), "role_ru": r.get("role_ru"), "evidence": r.get("evidence"),
        "legit": r.get("legit"), "legit_ru": r.get("legit_ru"), "legit_why": r.get("legit_why"),
        "factors": explain(r), "recommendation": r["recommendation"], "autoblock": bool(r["autoblock"]),
        "autoblock_reason": r["autoblock_reason"], "whitelisted": bool(r.get("whitelisted")),
        "ml_prob": r.get("ml_prob"),
        "ml_factors": (ml or {}).get("rules_model", {}).get("factors") or s.shap.get(num, []),
        "synth": (ml or {}).get("synth_model"), "ml_latency_ms": (ml or {}).get("latency_ms"), "group": group,
        "stats": {k: fr.get(k) for k in ("n_out", "n_in", "ub", "dur_med", "n_days", "n_hours", "n_imsi",
                                         "peak_1h", "sh02", "sh_night", "first_seen", "last_seen")},
    }


def calls_of(num: str, limit: int = 30) -> list[dict]:
    c = S().calls.filter((pl.col("a") == num) | (pl.col("b") == num) | (pl.col("c") == num)).sort("ts").head(limit)
    return [{"type": r["type"], "a": r["a"], "b": r["b"], "c": r["c"], "ts": r["ts"].isoformat(), "dur": r["dur"],
             "imsi": r["imsi"]} for r in c.iter_rows(named=True)]


MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def _models_of(card: dict | None) -> dict | None:
    """Мнение моделей о номере, по которому оценён вызов (посчитано в момент запроса)."""
    if not card:
        return None
    return {"num": card["num"], "scheme": card.get("scheme"), "legit": card.get("legit"),
            "rules_prob": card.get("ml_prob"), "synth": card.get("synth"),
            "latency_ms": card.get("ml_latency_ms")}


def call_card(a: str, b: str, ts: datetime, dur: int | None, typ: str = "02", c: str | None = None,
              reveal: bool = False) -> dict:
    """Карточка одного вызова, как на макете заказчика."""
    meta = f"{ts.day} {MONTHS[ts.month - 1]}, {ts:%H:%M:%S} · {dur if dur is not None else '?'} сек · {op_name(a)} → {op_name(b)}"
    if typ == "03" and c:
        meta += f" · переадресация на {mask(c, reveal)}"
    base = {"a": a, "b": b, "c": c, "a_masked": mask(a, reveal), "b_masked": mask(b, reveal), "ts": ts.isoformat(),
            "dur": dur, "type": typ, "meta": meta}
    # экстренные службы: вызов всегда проходит, оценка не применяется
    if is_emergency(b, CFG) or is_emergency(a, CFG):
        # риск звонящего показываем, но вызов на экстренную службу соединяется всегда
        try:
            ca = number_card(a, reveal) if not is_emergency(a, CFG) else None
        except HTTPException:
            ca = None
        fs = ([{"key": "emergency", "points": 0, "text": "вызов на экстренную службу — соединяется независимо от оценки"}]
              + (ca["factors"][:5] if ca else []))
        return base | {"score": ca["score"] if ca else None, "unknown": ca is None, "factors": fs, "models": _models_of(ca),
                       "recommendation": "соединить: экстренные вызовы не ограничиваются никогда",
                       "autoblock": False,
                       "autoblock_reason": "вызовы на экстренные и социальные службы не блокируются ни при какой оценке (fail-open); "
                                           "даже заблокированная SIM может звонить на 112",
                       "subject": {"num": ca["num"], "role_ru": ca["role_ru"], "scheme_ru": ca["scheme_ru"],
                                   "legit_ru": ca["legit_ru"]} if ca else None, "emergency": True}
    try:
        ca = number_card(a, reveal)
    except HTTPException:
        ca = None
    try:
        cb = number_card(b, reveal)
    except HTTPException:
        cb = None
    # риск вызова — по звонящему; если опасен вызываемый (например, перезвон на Wangiri) — отдельный фактор
    subj = ca
    factors = list(ca["factors"]) if ca else []
    score = ca["score"] if ca else 0
    if cb and cb["role"] in ("wangiri_source", "premium_C", "spoofed_city", "forward_chain_node") and cb["score"] > score:
        subj, score = cb, cb["score"]
        factors = [{"key": "callee", "points": cb["score"], "text": f"вызов НА номер: {cb['role_ru']}"}] + cb["factors"][:3]
    if subj is None:
        # номера нет в записях: поведения не видно — это «нет данных», а не «риск 0»
        return base | {"score": None, "unknown": True, "factors": [], "models": None,
                       "recommendation": "соединить как обычно — данных для оценки нет",
                       "autoblock": False,
                       "autoblock_reason": "номер не встречается в записях о звонках. Оценка строится по поведению номера "
                                           "(сколько, кому и как он звонит), поэтому появится, когда наберётся история; "
                                           "по одному звонку схему не определить",
                       "subject": None}
    if dur is not None and dur <= 2 and ca and ca.get("scheme") == "S1":
        factors.insert(0, {"key": "drop", "points": 0, "text": f"этот вызов — сброс ({dur} с)"})
    return base | {"score": score, "factors": factors[:8], "models": _models_of(subj),
                   "recommendation": subj["recommendation"] if subj else "нет действий",
                   "autoblock": subj["autoblock"] if subj else False,
                   "autoblock_reason": subj["autoblock_reason"] if subj else "номера нет в данных",
                   "subject": {"num": subj["num"], "role_ru": subj["role_ru"], "scheme_ru": subj["scheme_ru"],
                               "legit_ru": subj["legit_ru"]} if subj else None}


# ---------------- API ----------------
@app.get("/api/summary")
def api_summary():
    s = S()
    return {"summary": s.summary, "realtime": s.realtime, "eval": s.eval,
            "ml": {k: v for k, v in (s.ml or {}).items()}, "ml_synth": s.ml_synth, "classes_ru": CLASS_RU,
            "schemes": SCHEME_RU, "legit_names": LEGIT_RU,
            "database": {"name": s.db_name, "ok": s.db_ok, "error": s.db_error, "run_id": (s.run or {}).get("id")}}


@app.get("/api/decisions")
def api_decisions(limit: int = Query(30, le=500), reveal: bool = False):
    """Журнал решений по вызовам (что запрашивали через API и что система ответила)."""
    try:
        total, rows = db.recent_decisions(OUT, limit)
    except Exception as e:
        raise HTTPException(503, f"база недоступна: {str(e).splitlines()[0][:200]}")
    for r in rows:
        r["a_masked"], r["b_masked"] = mask(r["a"], reveal), mask(r["b"], reveal)
        r["synth_class_ru"] = CLASS_RU.get(r["synth_class"]) if r["synth_class"] else None
    return {"total": total, "items": rows}


def _q_digits(q: str | None) -> str:
    if not q:
        return ""
    qd = "".join(ch for ch in q if ch.isdigit())
    if q.strip().startswith("+7") or (len(qd) == 11 and qd[0] in "78"):
        qd = qd[1:]                                       # код страны в поиске
    return qd[-10:]


NUM_COLS = ("num", "score", "scheme", "role", "group_id", "autoblock", "legit", "ml_prob", "synth_prob", "synth_class",
            "review_source")


def _numbers_db(scheme, q, kind, limit, offset):
    t = db.numbers
    cond = []
    if kind == "fraud":
        cond.append(t.c.role.is_not(None))
    elif kind == "legit":
        cond.append(t.c.legit.is_not(None))
    elif kind == "review":
        cond.append(t.c.review_source.is_not(None))
    if scheme:
        cond.append(t.c.scheme == scheme)
    qd = _q_digits(q)
    if qd:
        cond.append(t.c.num.contains(qd))
    with db.engine(OUT).connect() as cx:
        total = cx.execute(select(func.count()).select_from(t).where(*cond)).scalar_one()
        rows = cx.execute(select(*[t.c[x] for x in NUM_COLS]).where(*cond)
                          .order_by(t.c.score.desc(), t.c.num).offset(offset).limit(limit)).mappings().all()
    out = []
    for r in rows:
        r = dict(r)
        r["scheme_ru"] = SCHEME_RU.get(r["scheme"]) if r["scheme"] else None
        r["role_ru"] = ROLE_RU.get(r["role"]) if r["role"] else None
        r["legit_ru"] = LEGIT_RU.get(r["legit"]) if r["legit"] else None
        out.append(r)
    return total, out


@app.get("/api/numbers")
def api_numbers(scheme: Optional[str] = None, q: Optional[str] = None, kind: str = "fraud",
                limit: int = Query(100, le=1000), offset: int = 0, reveal: bool = False):
    if S().db_ok:
        try:
            total, rows = _numbers_db(scheme, q, kind, limit, offset)
            for r in rows:
                r["masked"] = mask(r["num"], reveal)
            return {"total": total, "items": rows, "source": "db"}
        except Exception:                                        # pragma: no cover
            pass
    s = S().scores
    if kind == "fraud":
        s = s.filter(pl.col("role").is_not_null())
    elif kind == "legit":
        s = s.filter(pl.col("legit").is_not_null())
    elif kind == "review":
        s = s.filter(pl.col("num").is_in(S().review["num"].implode()))
    if scheme:
        s = s.filter(pl.col("scheme") == scheme)
    qd = _q_digits(q)
    if qd:
        s = s.filter(pl.col("num").str.contains(qd))
    s = s.sort(["score", "num"], descending=[True, False])
    total = s.height
    rows = s.slice(offset, limit).select("num", "score", "scheme", "scheme_ru", "role_ru", "group_id", "autoblock",
                                         "legit", "legit_ru", "ml_prob", "synth_prob", "synth_class").to_dicts()
    for r in rows:
        r["masked"] = mask(r["num"], reveal)
    return {"total": total, "items": rows, "source": "files"}


@app.get("/api/number/{num}")
def api_number(num: str, reveal: bool = False):
    card = number_card(num, reveal)
    card["calls"] = calls_of(num)
    return card


@app.get("/api/call/sample")
def api_call_sample(scheme: Optional[str] = None, reveal: bool = False, seed: Optional[int] = None):
    """Случайный реальный вызов из CDR (по умолчанию — из схемы) и его карточка."""
    s = S(); rnd = random.Random(seed)
    if scheme:
        nums = s.hits.filter(pl.col("scheme") == scheme)
        nums = nums.filter(~pl.col("role").is_in(["premium_C"]))["num"].to_list()
    else:
        nums = s.hits["num"].to_list()
    if not nums:
        raise HTTPException(404, "нет номеров")
    rnd.shuffle(nums)
    for n in nums:
        c = s.calls.filter((pl.col("a") == n) | (pl.col("b") == n) | (pl.col("c") == n))
        if c.height:
            break
    else:
        raise HTTPException(404, "у номеров этой схемы нет вызовов в данных")
    r = c.row(rnd.randrange(c.height), named=True)
    return call_card(r["a"], r["b"], r["ts"], r["dur"], r["type"], r["c"], reveal)


class CallIn(BaseModel):
    a: str
    b: str
    ts: datetime
    dur: Optional[int] = None
    type: str = "02"
    c: Optional[str] = None


@app.post("/api/score_call")
def api_score_call(call: CallIn, reveal: bool = False):
    """Оценка произвольного вызова (метаданные одной записи CDR)."""
    def norm(x: str) -> str:
        d = "".join(ch for ch in x if ch.isdigit())
        if is_emergency(d, CFG) or len(d) < 10:
            return d
        return d[-10:]
    a, b = norm(call.a), norm(call.b)
    if not a or not b:
        raise HTTPException(422, "нужны оба номера: A и B")
    if call.dur is not None and call.dur < 0:
        raise HTTPException(422, "длительность не может быть отрицательной")
    c = norm(call.c) if call.c else None
    t0 = time.perf_counter()
    card = call_card(a, b, call.ts.replace(tzinfo=None), call.dur, call.type, c, reveal)
    card["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    m = card.get("models") or {}
    try:                                                         # журнал решений: кто, когда, что решено и почему
        db.log_decision(OUT, a=a, b=b, c=c, call_ts=call.ts.replace(tzinfo=None), dur=call.dur, score=card["score"],
                        recommendation=card["recommendation"], autoblock=bool(card["autoblock"]),
                        emergency=bool(card.get("emergency")), subject=(card.get("subject") or {}).get("num"),
                        ml_prob=m.get("rules_prob"), synth_prob=(m.get("synth") or {}).get("fraud_prob"),
                        synth_class=(m.get("synth") or {}).get("cls"), latency_ms=card["latency_ms"])
        card["logged"] = True
    except Exception:                                            # pragma: no cover - база недоступна: решение всё равно отдаём
        card["logged"] = False
    return card


@app.get("/api/groups")
def api_groups(scheme: Optional[str] = None, reveal: bool = False):
    rows = None
    if S().db_ok:
        try:
            t = db.groups
            with db.engine(OUT).connect() as cx:
                rows = [dict(r) for r in cx.execute(select(t).where(*([t.c.scheme == scheme] if scheme else []))
                                                    .order_by(t.c.scheme, t.c.size.desc(), t.c.group_id)).mappings().all()]
        except Exception:                                        # pragma: no cover
            rows = None
    if rows is None:
        g = S().groups
        if scheme:
            g = g.filter(pl.col("scheme") == scheme)
        rows = list(g.sort("scheme", "size", descending=[False, True]).iter_rows(named=True))
    out = []
    for r in rows:
        out.append({"group_id": r["group_id"], "scheme": r["scheme"], "scheme_ru": r["scheme_ru"], "size": r["size"],
                    "max_score": r["max_score"], "example": r["example"],
                    "members": [{"num": m, "masked": mask(m, reveal)} for m in r["members"][:12]],
                    "roles": [ROLE_RU.get(x, x) for x in r["roles"]]})
    return out


@app.get("/api/answer", response_class=PlainTextResponse)
def api_answer(strict: bool = False):
    p = os.path.join(OUT, "answer_strict.txt" if strict else "answer.txt")
    with open(p, encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    s = S()
    return {"ok": s.scores is not None, "numbers_scored": len(s.by_num), "out": OUT,
            "database": s.db_name, "database_ok": s.db_ok, "models": {"rules": s.model_rules is not None,
                                                                    "synth": bool(s.synth_models)}}


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
