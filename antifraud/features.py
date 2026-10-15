"""Поведенческие признаки по номеру за период (как звонящий A и как вызываемый B/C)."""
from __future__ import annotations

import polars as pl

# признаки, которые идут в ML (числовые, без выводов детекторов)
ML_FEATURES = [
    "n_out", "uniq_share", "sh0", "sh02", "sh4_25", "sh8_45", "sh_long", "dur_med", "dur_mean",
    "span_min", "n_hours", "n_days", "sh_night", "sh_work", "overlap_share", "sh_backtoback",
    "gap_med", "n_imsi", "fwd_share_a", "sh_repeat_b", "sh_seq", "peak_1h", "peak_5m",
    "n_dest_ops", "top3_hour_share", "n_in", "ua_in", "in_out_ratio", "in_sh02", "n_fwd_as_b", "n_as_c",
]


def caller_features(df: pl.DataFrame, short_sec: int = 2, max_step: int = 3) -> pl.DataFrame:
    d = df.sort("a", "ts").with_columns(
        (pl.col("ts") - pl.col("te").shift(1).over("a")).dt.total_seconds().alias("gap"),
        (pl.col("b").cast(pl.Int64) - pl.col("b").cast(pl.Int64).shift(1).over("a")).abs().alias("db"),
    )
    f = d.group_by("a").agg(
        pl.len().alias("n_out"),
        pl.col("b").n_unique().alias("ub"),
        (pl.col("dur") == 0).mean().alias("sh0"),
        (pl.col("dur") <= short_sec).mean().alias("sh02"),
        pl.col("dur").is_between(4, 25).mean().alias("sh4_25"),
        pl.col("dur").is_between(8, 45).mean().alias("sh8_45"),
        (pl.col("dur") >= 420).mean().alias("sh_long"),
        pl.col("dur").median().alias("dur_med"),
        pl.col("dur").mean().alias("dur_mean"),
        pl.col("dur").max().alias("dur_max"),
        pl.col("ts").min().alias("first_seen"),
        pl.col("ts").max().alias("last_seen"),
        (pl.col("ts").max() - pl.col("ts").min()).dt.total_minutes().alias("span_min"),
        pl.col("h").n_unique().alias("n_hours"),
        pl.col("day").n_unique().alias("n_days"),
        (pl.col("h") < 6).mean().alias("sh_night"),
        pl.col("h").is_between(9, 18).mean().alias("sh_work"),
        (pl.col("gap") < 0).sum().alias("n_overlap"),
        pl.col("gap").is_between(0, 60).mean().alias("sh_backtoback"),
        pl.col("gap").median().alias("gap_med"),
        pl.col("imsi").drop_nulls().n_unique().alias("n_imsi"),
        (pl.col("type") == "03").sum().alias("n_fwd_as_a"),
        pl.col("db").is_between(1, max_step).mean().alias("sh_seq"),
        pl.col("op_b").n_unique().alias("n_dest_ops"),
        pl.col("op_a").first().alias("op"),
    ).with_columns(
        (pl.col("ub") / pl.col("n_out")).alias("uniq_share"),
        (pl.col("n_overlap") / pl.col("n_out")).alias("overlap_share"),
        (pl.col("n_fwd_as_a") / pl.col("n_out")).alias("fwd_share_a"),
    )
    rep = df.group_by("a", "b").len().group_by("a").agg((pl.col("len") > 1).mean().alias("sh_repeat_b"))
    p1 = df.group_by("a", pl.col("ts").dt.truncate("1h")).len().group_by("a").agg(pl.col("len").max().alias("peak_1h"))
    p5 = df.group_by("a", pl.col("ts").dt.truncate("5m")).len().group_by("a").agg(pl.col("len").max().alias("peak_5m"))
    hh = (df.group_by("a", "h").len().sort("len", descending=True)
          .group_by("a").agg((pl.col("len").head(3).sum() / pl.col("len").sum()).alias("top3_hour_share")))
    return (f.join(rep, on="a", how="left").join(p1, on="a", how="left").join(p5, on="a", how="left")
            .join(hh, on="a", how="left").rename({"a": "num"}))


def callee_features(df: pl.DataFrame, short_sec: int = 2) -> pl.DataFrame:
    b = df.group_by("b").agg(
        pl.len().alias("n_in"),
        pl.col("a").n_unique().alias("ua_in"),
        (pl.col("dur") <= short_sec).mean().alias("in_sh02"),
        (pl.col("type") == "03").sum().alias("n_fwd_as_b"),
    ).rename({"b": "num"})
    c = df.filter(pl.col("c").is_not_null()).group_by("c").agg(pl.len().alias("n_as_c")).rename({"c": "num"})
    return b.join(c, on="num", how="full", coalesce=True)


def build_features(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    short = cfg["s1_wangiri"]["short_sec"]
    cf = caller_features(df, short, cfg["s2_robot"]["max_step"])
    ce = callee_features(df, short)
    f = cf.join(ce, on="num", how="full", coalesce=True)
    codes = cfg["data"]["operator_codes"]
    zero = ["n_out", "ub", "n_in", "ua_in", "n_fwd_as_b", "n_as_c", "n_overlap", "n_imsi", "n_fwd_as_a", "n_hours", "n_days"]
    f = f.with_columns([pl.col(c).fill_null(0) for c in zero]).with_columns(
        pl.col("op").fill_null(pl.col("num").str.slice(0, 3).replace_strict(codes, default="OTHER")),
        (pl.col("n_in") / (pl.col("n_out") + 1)).alias("in_out_ratio"),
    )
    return f
