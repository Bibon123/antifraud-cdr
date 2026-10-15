"""Чтение и очистка CDR.

Источник: папка с *.cdr (рекурсивно) или zip-архив. Каждая строка либо
исправляется (с учётом в отчёте fixes), либо уходит в карантин с причиной.
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from collections import Counter
from datetime import datetime
from typing import Iterable, Iterator, TextIO

import polars as pl

from .config import load_config

TYPES = {"01", "02", "03"}
_DIGITS = re.compile(r"\D")
_NUM_FIELD = re.compile(r"\+?\d{10,12}")


def iter_sources(src: str) -> Iterator[tuple[str, TextIO]]:
    """(имя файла, текстовый поток) для папки или zip-архива."""
    if src.lower().endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            for name in sorted(n for n in z.namelist() if n.lower().endswith(".cdr")):
                with z.open(name) as fb:
                    yield os.path.basename(name), io.TextIOWrapper(fb, encoding="utf-8", errors="replace")
    else:
        for path in sorted(glob.glob(os.path.join(src, "**", "*.cdr"), recursive=True)):
            with open(path, encoding="utf-8", errors="replace") as f:
                yield os.path.basename(path), f


def norm_num(s: str):
    """Номер -> 10 цифр, '' для пустого, None если не восстановить. Снимает код страны +7/7/8."""
    s = s.strip()
    if not s:
        return ""
    d = _DIGITS.sub("", s)
    if len(d) == 10:
        return d
    if len(d) == 11 and d[0] in "78":
        return d[1:]
    if len(d) == 12 and d[:2] == "78":
        return d[2:]
    return None


def norm_ts(s: str):
    s = s.strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


class Report:
    def __init__(self):
        self.fixes: Counter = Counter()
        self.quarantine: list[tuple[str, int, str, str]] = []
        self.files = 0
        self.lines = 0

    def q(self, fname, ln, reason, raw):
        self.quarantine.append((fname, ln, reason, raw))

    def as_dict(self) -> dict:
        return {"files": self.files, "lines": self.lines,
                "quarantined": len(self.quarantine),
                "quarantine_reasons": dict(Counter(r for _, _, r, _ in self.quarantine)),
                "fixes": dict(self.fixes)}


def parse(sources: Iterable[tuple[str, TextIO]], cfg: dict | None = None) -> tuple[pl.DataFrame, Report]:
    cfg = cfg or load_config()
    codes = cfg["data"]["operator_codes"]
    t_lo = datetime.fromisoformat(cfg["data"]["period_start"])
    t_hi = datetime.fromisoformat(cfg["data"]["period_end"])
    max_dur = cfg["data"]["max_duration_sec"]
    imsi_len = cfg["data"]["imsi_length"]
    rep = Report()
    cols = {k: [] for k in ("type", "imsi", "a", "b", "ts", "dur", "c", "file", "line")}

    for fname, f in sources:
        rep.files += 1
        for ln, line in enumerate(f, 1):
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            rep.lines += 1
            p = raw.split(";")
            # --- число полей
            if len(p) > 7:
                rep.fixes["extra_fields_nonempty_dropped" if any(x.strip() for x in p[7:]) else "extra_empty_fields"] += 1
                p = p[:7]
            if len(p) == 6:
                last = p[5].strip()
                if p[0].strip() == "03" and _NUM_FIELD.fullmatch(last):
                    p = p[:5] + ["", p[5]]                    # нет длительности, есть номер C
                    rep.fixes["missing_duration"] += 1
                elif last and last.isdigit() and len(last) <= 6:
                    p = p[:5] + [last, ""]                    # нет последнего пустого поля C
                    rep.fixes["missing_trailing_field"] += 1
                else:
                    p = p[:5] + ["", ""]
                    rep.fixes["missing_duration"] += 1
            if len(p) != 7:
                rep.q(fname, ln, "field_count", raw); continue
            t, imsi, a, b, ts, dur, c = (x.strip() for x in p)
            # --- тип
            if t in ("1", "2", "3"):
                t = "0" + t; rep.fixes["type_leading_zero"] += 1
            if t not in TYPES:
                rep.q(fname, ln, "bad_type", raw); continue
            # --- номера
            na, nb, nc = norm_num(a), norm_num(b), norm_num(c)
            if na is None or nb is None or nc is None:
                rep.q(fname, ln, "bad_number", raw); continue
            if not na or not nb:
                rep.q(fname, ln, "empty_number", raw); continue
            if na[:3] not in codes or nb[:3] not in codes or (nc and nc[:3] not in codes):
                rep.q(fname, ln, "unknown_operator_code", raw); continue
            if (na, nb, nc) != (a, b, c):
                rep.fixes["number_country_code"] += 1
            # --- время
            dt = norm_ts(ts)
            if dt is None or not (t_lo <= dt < t_hi):
                rep.q(fname, ln, "bad_datetime", raw); continue
            if len(ts) != 14 or not ts.isdigit():
                rep.fixes["datetime_format"] += 1
            # --- длительность
            if dur == "":
                d = None
            elif dur.isdigit():
                d = int(dur)
                if d > max_dur:
                    d = None; rep.fixes["duration_out_of_range_nulled"] += 1
            else:
                rep.q(fname, ln, "bad_duration", raw); continue
            # --- IMSI
            if imsi and (len(imsi) != imsi_len or not imsi.isdigit()):
                imsi = ""; rep.fixes["truncated_imsi_nulled"] += 1
            for k, v in zip(cols, (t, imsi or None, na, nb, dt, d, nc or None, fname, ln)):
                cols[k].append(v)

    df = pl.DataFrame(cols, schema={"type": pl.Utf8, "imsi": pl.Utf8, "a": pl.Utf8, "b": pl.Utf8,
                                    "ts": pl.Datetime("us"), "dur": pl.Int64, "c": pl.Utf8,
                                    "file": pl.Utf8, "line": pl.Int64})
    n0 = df.height
    df = df.unique(subset=["type", "imsi", "a", "b", "ts", "dur", "c"], keep="first", maintain_order=True)
    if n0 - df.height:
        rep.fixes["exact_duplicates_removed"] += n0 - df.height
    return enrich(df, codes), rep


def enrich(df: pl.DataFrame, codes: dict) -> pl.DataFrame:
    """Производные колонки, общие для всех модулей."""
    return df.with_columns(
        pl.col("a").str.slice(0, 3).replace_strict(codes, default="OTHER").alias("op_a"),
        pl.col("b").str.slice(0, 3).replace_strict(codes, default="OTHER").alias("op_b"),
        (pl.col("ts") + pl.duration(seconds=pl.col("dur").fill_null(0))).alias("te"),
        pl.col("ts").dt.hour().alias("h"),
        pl.col("ts").dt.date().alias("day"),
    ).sort("ts")


def load_source(src: str, cfg: dict | None = None):
    df, rep = parse(iter_sources(src), cfg)
    if rep.files == 0:
        raise FileNotFoundError(f"Не найдено ни одного .cdr в {src}")
    return df, rep


def quarantine_frame(rep: Report) -> pl.DataFrame:
    return pl.DataFrame(rep.quarantine, schema=["file", "line", "reason", "raw"], orient="row")
