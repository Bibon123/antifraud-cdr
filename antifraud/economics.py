"""Экономический эффект: объём вреда по каждой схеме в данных и сколько его можно было остановить.

Объёмы (минуты, вызовы, жертвы) считаются по CDR точно. Деньги = объём × тариф из
config.toml [economics]: тарифы — допущения или публичные ориентиры (источник указан
рядом с параметром), их нужно заменить на тарифы оператора. «Можно было остановить» —
доля вредных событий, случившихся уже после момента онлайн-обнаружения номера
(antifraud/realtime.py). Годовая оценка — линейная экстраполяция периода данных.
"""
from __future__ import annotations

import polars as pl


def _mln(x: float) -> str:
    return f"{x / 1e6:.2f}".replace(".", ",")


def _min(df: pl.DataFrame) -> float:
    return float(df["dur"].fill_null(0).sum()) / 60 if df.height else 0.0


def _after(df: pl.DataFrame, num_col: str, det: dict) -> pl.DataFrame:
    """Записи, случившиеся после обнаружения соответствующего номера онлайн-правилом."""
    if df.height == 0 or not det:
        return df.head(0)
    d = pl.DataFrame({num_col: list(det), "_t": [t for t in det.values()]},
                     schema={num_col: pl.Utf8, "_t": pl.Datetime("us")})
    return df.join(d, on=num_col, how="inner").filter(pl.col("ts") > pl.col("_t")).drop("_t")


def economic_effect(df: pl.DataFrame, hits: pl.DataFrame, fired: dict, cfg: dict) -> dict:
    e = cfg.get("economics", {})
    det = {n: t for n, (_, t) in fired.items()}
    nums = {r: hits.filter(pl.col("role") == r)["num"] for r in hits["role"].unique().to_list()}
    get = lambda r: nums.get(r, pl.Series([], dtype=pl.Utf8)).implode()
    days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400, 1 / 24) if df.height else 1
    rows = []

    def add(scheme, what, unit, total, after, rate, rate_note, money_note):
        rows.append({"scheme": scheme, "what": what, "unit": unit, "volume": round(total, 1),
                     "volume_after_detection": round(after, 1),
                     "prevented_share": round(after / total, 3) if total else None,
                     "rate_rub": rate, "rate_note": rate_note,
                     "loss_rub": round(total * rate) if rate is not None else None,
                     "prevented_rub": round(after * rate) if rate is not None else None, "money_note": money_note})

    # S1: перезвоны жертв на платный номер (переадресация 03 с номера-источника на номер C)
    cb = df.filter((pl.col("type") == "03") & pl.col("b").is_in(get("wangiri_source")))
    cb_after = _after(cb.rename({"b": "src"}), "src", det)
    add("S1", "минут перезвонов жертв на платный номер", "мин", _min(cb), _min(cb_after), e.get("premium_rub_per_min"),
        e.get("premium_note", ""), "деньги абонентов, ушедшие на платный номер")
    # S2: звонки робота — спам без денежной оценки
    rb = df.filter(pl.col("a").is_in(get("robot_dialer")))
    add("S2", "вызовов робота-обзвонщика", "вызовов", rb.height, _after(rb, "a", det).height, None, "",
        "время и нервы абонентов; в рублях не оцениваем")
    # S3: трафик через SIM-box — оператор теряет разницу международной и местной терминации
    sb = df.filter(pl.col("a").is_in(get("simbox_sim")))
    add("S3", "минут трафика через SIM-box", "мин", _min(sb), _min(_after(sb, "a", det)),
        e.get("termination_gap_rub_per_min"), e.get("termination_note", ""), "недополученная выручка оператора за терминацию")
    # S4: вызовы клона — с IMSI, отличного от основного (самого частого) IMSI номера
    cl = df.filter(pl.col("a").is_in(get("cloned_msisdn")) & pl.col("imsi").is_not_null())
    if cl.height:
        main = cl.group_by("a", "imsi").len().sort("len", descending=True).group_by("a").agg(pl.col("imsi").first().alias("main"))
        cl = cl.join(main, on="a").filter(pl.col("imsi") != pl.col("main")).drop("main")
    add("S4", "минут разговоров с чужих SIM за счёт владельца", "мин", _min(cl), _min(_after(cl, "a", det)),
        e.get("retail_rub_per_min"), e.get("retail_note", ""), "списано со счёта законного владельца номера")
    # S5: плечи переадресаций через узлы цепочек — лишний межоператорский трафик
    ch = df.filter((pl.col("type") == "03") & pl.col("b").is_in(get("forward_chain_node")))
    add("S5", "минут переадресаций через узлы цепочек", "мин", _min(ch), _min(_after(ch, "b", det)),
        e.get("interconnect_rub_per_min"), e.get("interconnect_note", ""), "искусственный межоператорский трафик")
    # S6: звонки внутри карусели — накрутка трафика
    c6 = get("carousel_member")
    cc = df.filter(pl.col("a").is_in(c6) & pl.col("b").is_in(c6))
    add("S6", "минут звонков внутри каруселей", "мин", _min(cc), _min(_after(cc, "a", det)),
        e.get("interconnect_rub_per_min"), e.get("interconnect_note", ""), "искусственный межоператорский трафик")
    # S7: жертвы вишинга — длинный звонок после «прелюдии»; ущерб = жертвы × доля успешных × средняя потеря
    vc = df.filter(pl.col("a").is_in(get("vishing_caller")) & (pl.col("dur") >= cfg["s7_vishing"]["long_min_sec"]))
    victims = vc.group_by("b").agg(pl.col("ts").min(), pl.col("a").first())
    rate7 = None
    if e.get("vishing_success_share") is not None and e.get("avg_theft_rub") is not None:
        rate7 = round(e["vishing_success_share"] * e["avg_theft_rub"])
    add("S7", "абонентов, которым дозвонился мошенник (длинный звонок)", "жертв", victims.height,
        _after(victims, "a", det).height, rate7, e.get("vishing_note", ""),
        "ожидаемые потери абонентов: жертвы × доля успешных × средняя сумма хищения")

    loss = sum(r["loss_rub"] or 0 for r in rows)
    prevented = sum(r["prevented_rub"] or 0 for r in rows)
    k = 365 / days
    return {"period_days": round(days, 2), "rows": rows, "loss_rub": loss, "prevented_rub": prevented,
            "prevented_share": round(prevented / loss, 3) if loss else None,
            "annual_factor": round(k, 1), "loss_rub_year": round(loss * k), "prevented_rub_year": round(prevented * k),
            "headline": (f"За {days:.0f} дн. данных схемы нанесли бы ущерб ≈ {_mln(loss)} млн ₽; "
                         f"онлайн-обнаружение останавливает ≈ {_mln(prevented)} млн ₽ "
                         f"({prevented / loss * 100:.0f}%), ≈ {prevented * k / 1e6:.0f} млн ₽ в год при таком же потоке. "
                         "Тарифы — допущения из config.toml.") if loss else "Денежный ущерб не оценён (нет тарифов)."}
