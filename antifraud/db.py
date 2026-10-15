"""Операционная база: результаты прогонов, номера с решениями, группы, журнал решений по вызовам.

PostgreSQL или SQLite через SQLAlchemy — схема одна и та же:
  * DATABASE_URL=postgresql+psycopg://user:pass@host:5432/antifraud  — боевой вариант (docker compose up);
  * без DATABASE_URL — файл SQLite out/antifraud.db, ничего ставить не нужно (для ноутбука и демо).

Признаки всех номеров (сотни тысяч строк) остаются в Parquet (out/features.parquet) — это
хранилище признаков для модели; в базе — то, с чем работают аналитик и сервис.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine,
                        delete, func, insert, select)

meta = MetaData()

runs = Table(
    "runs", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False),
    Column("source", Text),
    Column("rows_clean", Integer),
    Column("numbers", Integer),
    Column("answer_full", Integer),
    Column("answer_strict", Integer),
    Column("summary", JSON),
)

numbers = Table(
    "numbers", meta,
    Column("num", String(16), primary_key=True),
    Column("run_id", Integer, index=True),
    Column("score", Integer, index=True),
    Column("scheme", String(4), index=True),
    Column("role", String(32)),
    Column("group_id", String(32), index=True),
    Column("legit", String(4), index=True),
    Column("recommendation", Text),
    Column("autoblock", Boolean),
    Column("ml_prob", Float),
    Column("synth_prob", Float),
    Column("synth_class", String(8)),
    Column("review_source", String(64)),
    Column("data", JSON),                      # полная строка оценки: факторы, тексты, доказательства
)

groups = Table(
    "number_groups", meta,
    Column("group_id", String(32), primary_key=True),
    Column("run_id", Integer),
    Column("scheme", String(4), index=True),
    Column("scheme_ru", Text),
    Column("size", Integer),
    Column("max_score", Integer),
    Column("members", JSON),
    Column("roles", JSON),
    Column("example", Text),
)

decisions = Table(
    "decisions", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False, index=True),
    Column("a", String(16)),
    Column("b", String(16)),
    Column("c", String(16)),
    Column("call_ts", DateTime),
    Column("dur", Integer),
    Column("score", Integer),
    Column("recommendation", Text),
    Column("autoblock", Boolean),
    Column("emergency", Boolean),
    Column("subject", String(16)),
    Column("ml_prob", Float),
    Column("synth_prob", Float),
    Column("synth_class", String(8)),
    Column("latency_ms", Float),
)


def db_url(out: str) -> str:
    """DATABASE_URL из окружения, иначе [database] url из config.toml, иначе SQLite-файл в out/."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        from .config import load_config
        url = (load_config().get("database") or {}).get("url") or ""
    return url or "sqlite:///" + os.path.abspath(os.path.join(out, "antifraud.db"))


def _ensure_pg_database(url: str) -> None:
    """Создаёт базу PostgreSQL, если её ещё нет (сервер есть, базы нет — частый случай на новом компьютере)."""
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    u = make_url(url)
    admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as cx:
            if not cx.execute(text("select 1 from pg_database where datname = :d"), {"d": u.database}).scalar():
                cx.execute(text(f'create database "{u.database}"'))
    finally:
        admin.dispose()


def describe(url: str) -> str:
    """Название базы без пароля — для логов и интерфейса."""
    if url.startswith("sqlite"):
        return "SQLite (" + os.path.basename(url.split("///", 1)[-1]) + ")"
    tail = url.split("@", 1)[-1]
    return "PostgreSQL (" + tail + ")"


_ENGINES: dict = {}


def engine(out: str):
    url = db_url(out)
    if url not in _ENGINES:
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            _ensure_pg_database(url)
        e = create_engine(url, **kw)
        meta.create_all(e)
        _ENGINES[url] = e
    return _ENGINES[url]


def _clean(v):
    """JSON-совместимое значение (без NaN, даты строкой)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def save_run(out: str, summary: dict, scores, groups_df, review) -> int:
    """Сохраняет прогон: история прогонов копится, номера и группы — снимок последнего прогона."""
    e = engine(out)
    rev = dict(zip(review["num"].to_list(), review["review_source"].to_list())) if review is not None and review.height else {}
    with e.begin() as cx:
        rid = cx.execute(insert(runs).values(
            created_at=datetime.now(), source=summary.get("source"), rows_clean=summary["ingest"].get("rows_clean"),
            numbers=summary.get("numbers"), answer_full=summary["answers"]["full"], answer_strict=summary["answers"]["strict"],
            summary=json.loads(json.dumps(_clean(summary), default=str)))).inserted_primary_key[0]
        cx.execute(delete(numbers)); cx.execute(delete(groups))
        rows = []
        for r in scores.iter_rows(named=True):
            r = _clean(r)
            rows.append({"num": r["num"], "run_id": rid, "score": int(r["score"]), "scheme": r.get("scheme"), "role": r.get("role"),
                         "group_id": r.get("group_id"), "legit": r.get("legit"), "recommendation": r.get("recommendation"),
                         "autoblock": bool(r.get("autoblock")), "ml_prob": r.get("ml_prob"), "synth_prob": r.get("synth_prob"),
                         "synth_class": r.get("synth_class"), "review_source": rev.get(r["num"]), "data": r})
        for i in range(0, len(rows), 2000):
            cx.execute(insert(numbers), rows[i:i + 2000])
        g = [{"group_id": x["group_id"], "run_id": rid, "scheme": x["scheme"], "scheme_ru": x["scheme_ru"], "size": x["size"],
              "max_score": x["max_score"], "members": list(x["members"]), "roles": list(x["roles"]), "example": x["example"]}
             for x in groups_df.iter_rows(named=True)]
        if g:
            cx.execute(insert(groups), g)
    return rid


def last_run(out: str) -> dict | None:
    with engine(out).connect() as cx:
        r = cx.execute(select(runs).order_by(runs.c.id.desc()).limit(1)).mappings().first()
    return dict(r) if r else None


def log_decision(out: str, **kw) -> None:
    with engine(out).begin() as cx:
        cx.execute(insert(decisions).values(created_at=datetime.now(), **kw))


def recent_decisions(out: str, limit: int = 50) -> tuple[int, list[dict]]:
    with engine(out).connect() as cx:
        total = cx.execute(select(func.count()).select_from(decisions)).scalar_one()
        rows = cx.execute(select(decisions).order_by(decisions.c.id.desc()).limit(limit)).mappings().all()
    return total, [dict(r) for r in rows]
