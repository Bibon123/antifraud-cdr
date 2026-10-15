"""Детекторы схем S1–S7 и легальных массовых источников D1–D5.

Каждый детектор возвращает строки (num, scheme, role, group_id, evidence) и
метрики для скоринга (rel-колонки m_*). Все пороги — из config.toml.
"""
from __future__ import annotations

import networkx as nx
import polars as pl

HIT_SCHEMA = {"num": pl.Utf8, "scheme": pl.Utf8, "role": pl.Utf8, "group_id": pl.Utf8, "evidence": pl.Utf8}
REL_COLS = {"m_callbacks": pl.Int64, "m_premium_hits": pl.Int64, "m_chains": pl.Int64, "m_rings": pl.Int64,
            "m_chain_len": pl.Int64, "m_group_size": pl.Int64, "m_closure": pl.Float64,
            "m_group_dur": pl.Float64, "m_prelude": pl.Float64, "m_spoof_share": pl.Float64,
            "m_victims": pl.Int64}

ROLE_RU = {
    "wangiri_source": "источник Wangiri", "premium_C": "платный номер для перезвонов",
    "robot_dialer": "робот-обзвонщик", "simbox_sim": "SIM в SIM-боксе", "cloned_msisdn": "клонированный номер",
    "forward_chain_node": "узел цепочки переадресаций", "carousel_member": "участник карусели",
    "vishing_caller": "звонящий мошенник (вишинг)", "spoofed_city": "подменённый городской номер",
}
SCHEME_RU = {"S1": "Wangiri", "S2": "Массовый обзвон (робот)", "S3": "SIM-box", "S4": "Клон SIM / смена IMSI",
             "S5": "Цепочка переадресаций", "S6": "Карусель", "S7": "Вишинг со спуфингом"}
LEGIT_RU = {"D1": "Колл-центр", "D2": "Банковские оповещения", "D3": "Ночные диспетчеры / охрана",
            "D4": "Горячая линия", "D5": "Оповещение ЖКХ/МЧС"}


def _hits(rows):
    return pl.DataFrame(rows, schema=HIT_SCHEMA, orient="row")


def _pct(x):
    return f"{x * 100:.0f}%"


# ---------- S1 Wangiri ----------
def s1_wangiri(df, f, cfg):
    c = cfg["s1_wangiri"]
    w = f.filter((pl.col("n_out") >= c["min_calls"]) & (pl.col("sh02") >= c["min_short_share"])
                 & (pl.col("uniq_share") >= c["min_unique_share"]) & (pl.col("span_min") <= c["max_span_min"]))
    cb = (df.filter(pl.col("b").is_in(w["num"].implode()) & (pl.col("type") == "03") & pl.col("c").is_not_null())
          .group_by("b", "c").agg(pl.len().alias("n")))
    cbn = dict(zip(*df.filter(pl.col("b").is_in(w["num"].implode())).group_by("b").len().to_dict().values()))
    rows, rel = [], []
    for r in w.iter_rows(named=True):
        rows.append((r["num"], "S1", "wangiri_source", f"S1-{r['num']}",
                     f"{r['ub']} разных номеров за {r['span_min']} мин, {_pct(r['sh02'])} вызовов 0–2 с, "
                     f"{cbn.get(r['num'], 0)} перезвонов жертв"))
        rel.append({"num": r["num"], "m_callbacks": cbn.get(r["num"], 0)})
    for b, cnum, n in cb.iter_rows():
        rows.append((cnum, "S1", "premium_C", f"S1-{b}",
                     f"на этот номер переадресовано {n} перезвонов жертв Wangiri-номера {b}"))
        rel.append({"num": cnum, "m_premium_hits": n})
    return _hits(rows), rel


# ---------- S2 Робот: последовательный диапазон ----------
def s2_robot(df, f, cfg):
    c = cfg["s2_robot"]
    r = f.filter((pl.col("n_out") >= c["min_calls"]) & (pl.col("sh_seq") >= c["min_seq_share"])
                 & (pl.col("span_min") <= c["max_span_min"]))
    if r.height == 0:
        return _hits([]), []
    rng = (df.filter(pl.col("a").is_in(r["num"].implode()))
           .group_by("a").agg(pl.col("b").cast(pl.Int64).min().alias("lo"), pl.col("b").cast(pl.Int64).max().alias("hi"),
                              pl.col("ts").min().alias("t0"), pl.col("ts").max().alias("t1")))
    # кампания: номера, которые параллельно обзванивают пересекающиеся диапазоны
    G = nx.Graph(); G.add_nodes_from(rng["a"].to_list())
    rr = rng.rows(named=True)
    for i in range(len(rr)):
        for j in range(i + 1, len(rr)):
            x, y = rr[i], rr[j]
            if x["lo"] <= y["hi"] and y["lo"] <= x["hi"] and x["t0"] <= y["t1"] and y["t0"] <= x["t1"]:
                G.add_edge(x["a"], y["a"])
    gid, gsize = {}, {}
    for k, comp in enumerate(sorted(nx.connected_components(G), key=lambda s: min(s))):
        for n in comp:
            gid[n], gsize[n] = f"S2-{k + 1:03d}", len(comp)
    rows, rel = [], []
    for x in r.iter_rows(named=True):
        rows.append((x["num"], "S2", "robot_dialer", gid[x["num"]],
                     f"{x['n_out']} вызовов за {x['span_min']} мин, {_pct(x['sh_seq'])} номеров B подряд, "
                     f"{_pct(x['sh0'])} нулевых; в кампании {gsize[x['num']]} номер(а)"))
        rel.append({"num": x["num"], "m_group_size": gsize[x["num"]]})
    return _hits(rows), rel


# ---------- S3 SIM-box ----------
def s3_simbox(df, f, cfg):
    c = cfg["s3_simbox"]
    s = f.filter((pl.col("n_in") <= c["max_incoming"]) & (pl.col("n_out") >= c["min_calls"]) & (pl.col("uniq_share") >= c["min_unique_share"])
                 & (pl.col("n_hours") >= c["min_active_hours"]) & (pl.col("n_overlap") == 0)
                 & (pl.col("n_imsi") <= 1) & (pl.col("sh_long") < c["max_long_share"]))
    rows = [(x["num"], "S3", "simbox_sim", f"S3-{x['num']}",
             f"только исходящие ({x['n_out']}), входящих {x['n_in']}, все B разные, активен {x['n_hours']} ч суток, "
             f"вызовы не пересекаются") for x in s.iter_rows(named=True)]
    return _hits(rows), []


# ---------- S4 Клон SIM ----------
def s4_clone(df, f, cfg):
    s = f.filter(pl.col("n_imsi") >= cfg["s4_clone"]["min_imsi"])
    rows = [(x["num"], "S4", "cloned_msisdn", f"S4-{x['num']}",
             f"{x['n_imsi']} разных IMSI на одном номере, {x['n_overlap']} одновременных вызовов")
            for x in s.iter_rows(named=True)]
    return _hits(rows), []


# ---------- S5 Цепочки переадресаций ----------
def s5_chains(df, f, cfg):
    c = cfg["s5_chains"]
    fw = df.filter(pl.col("type") == "03").with_row_index("rid").select("rid", "a", "b", "c", "ts", "dur")
    j = fw.join(fw, left_on=["c", "a"], right_on=["b", "a"], suffix="_2").filter(
        (pl.col("ts_2") - pl.col("ts")).dt.total_seconds().abs() <= c["link_tolerance_sec"])
    G = nx.Graph(); G.add_edges_from(zip(j["rid"].to_list(), j["rid_2"].to_list()))
    rec = {r: (b, cc) for r, b, cc in zip(fw["rid"].to_list(), fw["b"].to_list(), fw["c"].to_list())}
    stats, N = {}, nx.Graph()
    for comp in nx.connected_components(G):
        rs = [rec[r] for r in comp]
        bs, cs = {x[0] for x in rs}, {x[1] for x in rs}
        ring = cs <= bs
        nodes = bs | cs
        for n in nodes:
            s = stats.setdefault(n, [0, 0, 0]); s[0] += 1; s[1] += ring; s[2] = max(s[2], len(rs))
        nl = list(nodes); N.add_edges_from(zip(nl, nl[1:]))
    keep = {n for n, v in stats.items() if v[0] >= c["min_chains"]}
    gid = {}
    for k, comp in enumerate(sorted(nx.connected_components(N.subgraph(keep)), key=lambda s: min(s))):
        for n in comp:
            gid[n] = f"S5-{k + 1:03d}"
    rows, rel = [], []
    for n in sorted(keep):
        v = stats[n]
        rows.append((n, "S5", "forward_chain_node", gid[n],
                     f"узел в {v[0]} цепочках переадресаций (до {v[2]} звеньев), {v[1]} из них замкнуты в кольцо"))
        rel.append({"num": n, "m_chains": v[0], "m_rings": v[1], "m_chain_len": v[2]})
    return _hits(rows), rel


# ---------- S6 Карусель ----------
def s6_carousel(df, f, cfg):
    c = cfg["s6_carousel"]
    p = df.group_by(pl.min_horizontal("a", "b").alias("x"), pl.max_horizontal("a", "b").alias("y")).agg(pl.len().alias("n"))
    st = p.filter(pl.col("n") >= c["pair_min_calls"])
    G = nx.Graph(); G.add_edges_from(zip(st["x"].to_list(), st["y"].to_list()))
    rows, rel, k = [], [], 0
    for comp in sorted(nx.connected_components(G), key=lambda s: min(s)):
        if not (c["min_group"] <= len(comp) <= c["max_group"]):
            continue
        cl = list(comp)
        sub = df.filter(pl.col("a").is_in(cl) | pl.col("b").is_in(cl))
        inside = sub.filter(pl.col("a").is_in(cl) & pl.col("b").is_in(cl))
        closure = inside.height / sub.height
        nh = inside["h"].n_unique()
        dmed = inside["dur"].median() or 0
        if closure >= c["min_closure"] and nh >= c["min_hours"] and c["dur_min"] <= dmed <= c["dur_max"]:
            k += 1
            for n in sorted(cl):
                rows.append((n, "S6", "carousel_member", f"S6-{k:03d}",
                             f"замкнутая группа из {len(cl)} номеров: {_pct(closure)} звонков внутри, "
                             f"{nh} ч суток, медиана {dmed:.0f} с"))
                rel.append({"num": n, "m_group_size": len(cl), "m_closure": closure, "m_group_dur": float(dmed)})
    return _hits(rows), rel


# ---------- S7 Вишинг со спуфингом ----------
def s7_vishing(df, f, cfg):
    c = cfg["s7_vishing"]
    fr = f.filter((pl.col("sh_long") >= c["min_long_share"]) & (pl.col("n_out") >= c["min_calls"]) & (pl.col("n_in") <= c["max_incoming"]))
    L = (df.filter(pl.col("a").is_in(fr["num"].implode()) & pl.col("dur").is_between(c["long_min_sec"], c["long_max_sec"]))
         .select(pl.col("a").alias("fr"), pl.col("b").alias("v"), pl.col("ts").alias("tl")))
    C = (df.filter(pl.col("op_a") == "CITY").with_row_index("rid")
         .select("rid", pl.col("b").alias("v"), pl.col("a").alias("city"), "ts"))
    j = L.join(C, on="v").filter((pl.col("tl") - pl.col("ts")).dt.total_minutes().is_between(0, c["prelude_window_min"]))
    pre = (L.join(j.group_by("fr", "v", "tl").agg(pl.col("city").n_unique().alias("nc")), on=["fr", "v", "tl"], how="left")
           .group_by("fr").agg(pl.col("nc").fill_null(0).mean().alias("avg_city"), pl.col("v").n_unique().alias("nv")))
    hit = j.select("rid").unique().with_columns(pl.lit(1).alias("hit"))
    sp = (C.filter(pl.col("city").is_in(j["city"].unique().implode())).join(hit, on="rid", how="left")
          .group_by("city").agg(pl.len().alias("n"), pl.col("hit").fill_null(0).mean().alias("sh_hit"),
                                pl.col("v").n_unique().alias("nv"))
          .filter(pl.col("sh_hit") >= c["spoof_min_hit_share"]))
    frx = fr.join(pre, left_on="num", right_on="fr").filter(pl.col("avg_city") >= c["min_avg_prelude_numbers"])
    # сеть: мошенник — подменённые номера через общих жертв
    G = nx.Graph()
    good_fr, good_sp = set(frx["num"]), set(sp["city"])
    for fr_, city in j.select("fr", "city").unique().iter_rows():
        if fr_ in good_fr and city in good_sp:
            G.add_edge("F" + fr_, "C" + city)
    gid = {}
    for k, comp in enumerate(sorted(nx.connected_components(G), key=lambda s: min(s))):
        for n in comp:
            gid[n[1:]] = f"S7-{k + 1:03d}"
    rows, rel = [], []
    for x in frx.iter_rows(named=True):
        rows.append((x["num"], "S7", "vishing_caller", gid.get(x["num"], f"S7-{x['num']}"),
                     f"{x['n_out']} длинных звонков (медиана {x['dur_med']:.0f} с), входящих {x['n_in']}; перед звонком у жертвы "
                     f"в среднем {x['avg_city']:.1f} «городских» номера со сбросами; {x['nv']} жертв"))
        rel.append({"num": x["num"], "m_prelude": x["avg_city"], "m_victims": x["nv"]})
    for x in sp.iter_rows(named=True):
        rows.append((x["city"], "S7", "spoofed_city", gid.get(x["city"], f"S7-{x['city']}"),
                     f"городской номер: {_pct(x['sh_hit'])} его звонков — будущим жертвам вишинга "
                     f"незадолго до длинного звонка мошенника ({x['nv']} жертв)"))
        rel.append({"num": x["city"], "m_spoof_share": x["sh_hit"], "m_victims": x["nv"]})
    return _hits(rows), rel


DETECTORS = [s1_wangiri, s2_robot, s3_simbox, s4_clone, s5_chains, s6_carousel, s7_vishing]


# ---------- Легальные массовые источники ----------
def legit_sources(df, f, cfg):
    out = []
    d1 = f.filter((pl.col("op") == "CITY") & (pl.col("n_out") >= 100) & (pl.col("sh_work") >= 0.8)
                  & pl.col("sh0").is_between(0.2, 0.5) & (pl.col("n_days") <= 4) & (pl.col("sh02") < 0.6))
    out.append(d1.select("num", pl.lit("D1").alias("legit"),
                         pl.format("рабочие часы ({}%), {} дня, есть разговоры; {} входящих",
                                   (pl.col("sh_work") * 100).round(0).cast(pl.Int64), "n_days", "n_in").alias("why")))
    d2 = f.filter((pl.col("n_out") >= 300) & (pl.col("n_hours") <= 4) & (pl.col("n_days") >= 4)
                  & pl.col("dur_med").is_between(6, 20))
    out.append(d2.select("num", pl.lit("D2").alias("legit"),
                         pl.format("волны в одни и те же {} часа суток, {} дней подряд, вызовы {} с",
                                   "n_hours", "n_days", pl.col("dur_med").cast(pl.Int64)).alias("why")))
    fw = (df.filter(pl.col("type") == "03").group_by("c")
          .agg(pl.len().alias("n"), pl.col("b").n_unique().alias("nb"), pl.col("dur").median().alias("dm")))
    d4 = fw.filter((pl.col("n") >= 200) & pl.col("nb").is_between(8, 40) & (pl.col("dm") >= 30))
    out.append(d4.select(pl.col("c").alias("num"), pl.lit("D4").alias("legit"),
                         pl.format("единый номер C для {} номеров компании, {} переадресаций, медиана {} с",
                                   "nb", "n", pl.col("dm").cast(pl.Int64)).alias("why")))
    d4b = (df.filter((pl.col("type") == "03") & pl.col("c").is_in(d4["c"].implode()))
           .group_by("b").agg(pl.len().alias("n"), pl.col("c").first().alias("c")))
    out.append(d4b.select(pl.col("b").alias("num"), pl.lit("D4").alias("legit"),
                          pl.format("номер компании: {} входящих переадресованы на горячую линию {}", "n", "c").alias("why")))
    d5 = f.filter((pl.col("op") == "CITY") & (pl.col("n_out") >= 200) & (pl.col("span_min") <= 180) & (pl.col("sh_seq") < 0.2))
    out.append(d5.select("num", pl.lit("D5").alias("legit"),
                         pl.format("{} вызовов за {} мин со случайными номерами B (не диапазон)", "n_out", "span_min").alias("why")))
    d3 = f.filter((pl.col("sh_repeat_b") >= 0.9) & (pl.col("n_out") >= 100) & (pl.col("ub") >= 10) & (pl.col("sh_night") >= 0.35))
    out.append(d3.select("num", pl.lit("D3").alias("legit"),
                         pl.format("{}% звонков ночью, повторные контакты ({} постоянных), есть входящие",
                                   (pl.col("sh_night") * 100).round(0).cast(pl.Int64), "ub").alias("why")))
    return pl.concat(out).unique("num", keep="first")


def run_detectors(df: pl.DataFrame, f: pl.DataFrame, cfg: dict):
    parts, rel = [], []
    for d in DETECTORS:
        h, r = d(df, f, cfg)
        parts.append(h); rel.extend(r)
    hits = pl.concat(parts).unique("num", keep="first").sort("scheme", "group_id", "num")
    relf = (pl.DataFrame(rel, schema={"num": pl.Utf8, **REL_COLS}) if rel
            else pl.DataFrame(schema={"num": pl.Utf8, **REL_COLS}))
    relf = relf.group_by("num").agg([pl.col(k).max() for k in REL_COLS])
    legit = legit_sources(df, f, cfg)          # пересечение с hits проверяется в pipeline
    return hits, relf, legit
