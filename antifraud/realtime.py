"""Потоковый режим: записи CDR обрабатываются по одной в порядке времени.

Для каждого номера хранится небольшое состояние (скользящие окна, счётчики).
Онлайн-правило срабатывает в момент, когда данных уже достаточно. Это даёт:
  * время до обнаружения каждой схемы;
  * долю вреда, которую успели бы предотвратить (перезвоны, звонки жертвам, трафик);
  * ложные срабатывания онлайн-правил на всех остальных номерах.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from statistics import median

import polars as pl

ONLINE = {  # пороги онлайн-правил (мягче пакетных: решение нужно раньше)
    "S1": {"window_min": 60, "min_calls": 100, "min_short": 0.9},
    "S2": {"window_min": 30, "min_calls": 50, "min_seq": 0.8},
    "S3": {"min_calls": 60, "max_in": 2, "min_unique": 0.98, "min_hours": 12},
    "S4": {"min_imsi": 3},
    "S5": {"min_links": 3, "tol_sec": 5},
    "S6": {"min_calls": 40, "max_partners": 11, "min_regular_share": 0.98, "min_hours": 12},
    "S7": {"min_long": 3, "long_min": 400, "prelude_min": 360, "min_prelude_numbers": 2, "max_in": 2},
}


class _Num:
    __slots__ = ("w60", "w30", "n", "bs", "hours", "last_end", "overlap", "long", "first", "partners", "pcnt", "phours")

    def __init__(self):
        self.w60 = deque(); self.w30 = deque(); self.n = 0; self.bs = set(); self.hours = set()
        self.last_end = None; self.overlap = 0; self.long = 0; self.first = None
        self.partners = defaultdict(int); self.pcnt = 0; self.phours = set()


def replay(df: pl.DataFrame, eligible: set[str], cfg_online: dict = ONLINE):
    """Возвращает {num: (scheme, t_detect)} — первое срабатывание онлайн-правила."""
    o = cfg_online
    st: dict[str, _Num] = {}
    inc = defaultdict(int)                 # входящие по номеру
    imsis = defaultdict(set)
    city_recent = defaultdict(deque)       # жертва -> (ts, городской номер)
    vish = defaultdict(int)                # звонящий -> длинных звонков с прелюдией
    fwd_recent = defaultdict(deque)        # A -> недавние записи 03 (ts, b, c)
    links = defaultdict(int)
    fired: dict[str, tuple[str, object]] = {}

    def fire(num, scheme, t):
        if num not in fired:
            fired[num] = (scheme, t)

    cols = df.select("type", "imsi", "a", "b", "ts", "dur", "c", "op_a").iter_rows()
    for typ, imsi, a, b, ts, dur, c, op_a in cols:
        d = dur or 0
        inc[b] += 1
        # --- S4: третий IMSI на номере
        if imsi:
            s = imsis[a]
            if imsi not in s:
                s.add(imsi)
                if len(s) >= o["S4"]["min_imsi"]:
                    fire(a, "S4", ts)
        # --- S5: звенья цепочки переадресаций (одинаковый A, C предыдущей записи = B текущей)
        if typ == "03" and c:
            q = fwd_recent[a]
            while q and (ts - q[0][0]).total_seconds() > o["S5"]["tol_sec"]:
                q.popleft()
            for _, pb, pc in q:
                if pc == b or c == pb:
                    for n in (pb, pc, b, c):
                        links[n] += 1
                        if links[n] >= o["S5"]["min_links"] * 2:
                            fire(n, "S5", ts)
            q.append((ts, b, c))
        # --- S7: прелюдия из «городских» у жертвы
        if op_a == "CITY":
            cq = city_recent[b]; cq.append((ts, a))
        if d >= o["S7"]["long_min"] and inc[a] <= o["S7"]["max_in"]:
            cq = city_recent.get(b)
            if cq:
                lim = ts - timedelta(minutes=o["S7"]["prelude_min"])
                while cq and cq[0][0] < lim:
                    cq.popleft()
                if len({x for _, x in cq}) >= o["S7"]["min_prelude_numbers"]:
                    vish[a] += 1
                    if vish[a] >= o["S7"]["min_long"]:
                        fire(a, "S7", ts)
        # --- S6: собеседники (в обе стороны)
        for x, y in ((a, b), (b, a)):
            if x in eligible:
                s6 = st.setdefault(x, _Num())
                s6.partners[y] += 1; s6.pcnt += 1; s6.phours.add(ts.hour)
                if s6.pcnt >= o["S6"]["min_calls"] and len(s6.partners) <= o["S6"]["max_partners"] \
                        and len(s6.phours) >= o["S6"]["min_hours"]:
                    reg = sum(v for v in s6.partners.values() if v >= 3) / s6.pcnt
                    if reg >= o["S6"]["min_regular_share"] and len(s6.partners) >= 3:
                        fire(x, "S6", ts)
        if a not in eligible:
            continue
        s = st.setdefault(a, _Num())
        if s.first is None:
            s.first = ts
        s.n += 1; s.bs.add(b); s.hours.add(ts.hour)
        if s.last_end is not None and ts < s.last_end:
            s.overlap += 1
        end = ts + timedelta(seconds=d)
        s.last_end = end if s.last_end is None or end > s.last_end else s.last_end
        s.long += d >= 420
        try:
            bi = int(b)
        except ValueError:
            bi = 0
        # --- S1: окно 60 мин
        s.w60.append((ts, d <= 2))
        lim = ts - timedelta(minutes=o["S1"]["window_min"])
        while s.w60 and s.w60[0][0] < lim:
            s.w60.popleft()
        if len(s.w60) >= o["S1"]["min_calls"]:
            if sum(x for _, x in s.w60) / len(s.w60) >= o["S1"]["min_short"]:
                fire(a, "S1", ts)
        # --- S2: окно 30 мин, соседние номера B
        prev = s.w30[-1][1] if s.w30 else None
        s.w30.append((ts, bi, prev is not None and 1 <= abs(bi - prev) <= 3))
        lim = ts - timedelta(minutes=o["S2"]["window_min"])
        while s.w30 and s.w30[0][0] < lim:
            s.w30.popleft()
        if len(s.w30) >= o["S2"]["min_calls"]:
            if sum(x[2] for x in s.w30) / len(s.w30) >= o["S2"]["min_seq"]:
                fire(a, "S2", ts)
        # --- S3: SIM-box
        p = o["S3"]
        if s.n >= p["min_calls"] and inc[a] <= p["max_in"] and len(s.bs) / s.n >= p["min_unique"] \
                and len(s.hours) >= p["min_hours"] and s.overlap == 0 and s.long / s.n < 0.5:
            fire(a, "S3", ts)
    return fired


def _fmt_td(sec):
    if sec is None:
        return None
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60} мин"
    return f"{sec / 3600:.1f} ч"


def time_to_detect(df: pl.DataFrame, hits: pl.DataFrame, cfg: dict | None = None) -> dict:
    act = pl.concat([df.select(pl.col("a").alias("num")), df.select(pl.col("b").alias("num"))]).group_by("num").len()
    eligible = set(act.filter(pl.col("len") >= 30)["num"]) | set(hits["num"])
    fired = replay(df, eligible)
    truth = dict(zip(hits["num"], hits["scheme"]))
    roles = dict(zip(hits["num"], hits["role"]))
    first = (pl.concat([df.select(pl.col("a").alias("num"), "ts"), df.select(pl.col("b").alias("num"), "ts"),
                        df.filter(pl.col("c").is_not_null()).select(pl.col("c").alias("num"), "ts")])
             .group_by("num").agg(pl.col("ts").min().alias("t0")))
    t0 = dict(zip(first["num"], first["t0"]))

    # вредные события после обнаружения (что можно было предотвратить)
    harm = {}
    det_time = {n: t for n, (_, t) in fired.items()}
    w = hits.filter(pl.col("role") == "wangiri_source")["num"]
    cb = df.filter(pl.col("b").is_in(w.implode()) & (pl.col("type") == "03")).select("b", "ts")
    harm["S1"] = ("перезвонов жертв на платный номер", [(r[1] > det_time[r[0]]) if r[0] in det_time else False
                                                         for r in cb.iter_rows()])
    v = hits.filter(pl.col("role") == "vishing_caller")["num"]
    vc = df.filter(pl.col("a").is_in(v.implode()) & (pl.col("dur") >= 400)).select("a", "ts")
    harm["S7"] = ("длинных звонков мошенника жертвам", [(r[1] > det_time[r[0]]) if r[0] in det_time else False
                                                        for r in vc.iter_rows()])
    for sch, role in (("S2", "robot_dialer"), ("S3", "simbox_sim")):
        nums = hits.filter(pl.col("role") == role)["num"]
        x = df.filter(pl.col("a").is_in(nums.implode())).select("a", "ts")
        harm[sch] = ("вызовов схемы", [(r[1] > det_time[r[0]]) if r[0] in det_time else False for r in x.iter_rows()])

    per = {}
    primary = {"S1": "wangiri_source", "S2": "robot_dialer", "S3": "simbox_sim", "S4": "cloned_msisdn",
               "S5": "forward_chain_node", "S6": "carousel_member", "S7": "vishing_caller"}
    for sch, role in primary.items():
        nums = [n for n, r in roles.items() if r == role]
        delays = [(fired[n][1] - t0[n]).total_seconds() for n in nums if n in fired]
        fp = [n for n, (s, _) in fired.items() if s == sch and n not in truth]
        row = {"entities": len(nums), "detected_online": len(delays),
               "recall_online": round(len(delays) / len(nums), 3) if nums else None,
               "median_delay": _fmt_td(median(delays)) if delays else None,
               "median_delay_sec": int(median(delays)) if delays else None,
               "false_positive_numbers": len(fp)}
        if sch in harm:
            what, flags = harm[sch]
            row["harm_events"] = len(flags)
            row["prevented_share"] = round(sum(flags) / len(flags), 3) if flags else None
            row["prevented_what"] = what
        per[sch] = row
    total_fp = sum(1 for n in fired if n not in truth)
    s1 = per["S1"]
    if s1["median_delay"]:
        headline = (f"Wangiri обнаруживается в среднем через {s1['median_delay']} после начала обзвона; "
                    f"{(s1.get('prevented_share') or 0) * 100:.1f}% перезвонов жертв на платный номер можно было предотвратить. ")
    else:
        headline = "Wangiri в данных не найден. "
    headline += f"Ложных онлайн-срабатываний на остальных номерах: {total_fp}."
    res = {"per_scheme": per, "false_positive_total": total_fp, "online_thresholds": ONLINE, "headline": headline,
           "fired_sample": [{"num": n, "scheme": s, "t": str(t)} for n, (s, t) in list(fired.items())[:50]]}
    if cfg is not None and cfg.get("economics"):
        from .economics import economic_effect
        # деньги считаем только по номерам схем (ложных онлайн-срабатываний на остальных нет — см. false_positive_total)
        res["economics"] = economic_effect(df, hits, {n: v for n, v in fired.items() if n in truth}, cfg)
    return res
