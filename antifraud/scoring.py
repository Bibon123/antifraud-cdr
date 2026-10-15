"""Объяснимая оценка 0–100: сумма вкладов понятных факторов + политика действий.

Каждый фактор — правило на поведенческих признаках номера с весом. Вклад
фактора виден оператору («312 вызовов за 4 минуты +38»). Отрицательные факторы
(признаки легального поведения) снижают оценку.
"""
from __future__ import annotations

import polars as pl

from .detectors import LEGIT_RU, ROLE_RU, SCHEME_RU


def _clip01(e):
    return pl.min_horizontal(pl.lit(1.0), pl.max_horizontal(pl.lit(0.0), e))


def _i(c):
    return pl.col(c).fill_null(0).cast(pl.Int64).cast(pl.Utf8)


def _p(c):
    return (pl.col(c).fill_null(0) * 100).round(0).cast(pl.Int64).cast(pl.Utf8)


# (ключ, баллы-выражение, текст-выражение). Баллы 0 => фактор не показывается.
def factor_defs():
    n_out, n_in = pl.col("n_out"), pl.col("n_in")
    mass = n_out >= 30
    # входящие/повторы, которые сами являются частью схемы, не считаются признаком легальности
    fraud_ctx = ((pl.col("m_closure").fill_null(0) >= 0.9) | (pl.col("m_spoof_share").fill_null(0) > 0)
                 | (pl.col("m_callbacks").fill_null(0) > 0))
    return [
        # --- массовость и всплески
        ("burst", pl.when(pl.col("peak_1h") >= 60).then(25 * _clip01((pl.col("peak_1h") - 60) / 240)).otherwise(0),
         pl.concat_str([_i("peak_1h"), pl.lit(" вызовов за час (пик)")])),
        ("burst5", pl.when(pl.col("peak_5m") >= 30).then(10.0).otherwise(0),
         pl.concat_str([_i("peak_5m"), pl.lit(" вызовов за 5 минут")])),
        ("short", pl.when(mass).then(30 * _clip01((pl.col("sh02") - 0.4) / 0.5)).otherwise(0),
         pl.concat_str([_p("sh02"), pl.lit("% вызовов — сброс 0–2 с")])),
        ("seq", pl.when(mass).then(40 * _clip01((pl.col("sh_seq") - 0.3) / 0.5)).otherwise(0),
         pl.concat_str([_p("sh_seq"), pl.lit("% вызовов на номера B подряд (перебор диапазона)")])),
        ("mass_drop", pl.when((n_out >= 200) & (pl.col("sh02") >= 0.9)).then(25.0).otherwise(0),
         pl.concat_str([_i("ub"), pl.lit(" номеров обзвонено сбросами за "), _i("span_min"), pl.lit(" мин")])),
        ("uniq", pl.when((n_out >= 40) & (pl.col("uniq_share") >= 0.98)).then(10.0).otherwise(0),
         pl.concat_str([pl.lit("все "), _i("ub"), pl.lit(" собеседников разные")])),
        ("out_only", pl.when((n_out >= 40) & (n_in <= 2)).then(30.0).otherwise(0),
         pl.concat_str([_i("n_out"), pl.lit(" исходящих, входящих: "), _i("n_in")])),
        ("allday", pl.when((pl.col("n_hours") >= 8) & (n_in <= 2) & (n_out >= 40))
         .then(25 * _clip01(pl.col("n_hours") / 16)).otherwise(0),
         pl.concat_str([pl.lit("без перерыва: активен в "), _i("n_hours"), pl.lit(" разных часах суток")])),
        ("nooverlap", pl.when((n_out >= 40) & (pl.col("n_overlap") == 0) & (n_in <= 2) & (pl.col("uniq_share") >= 0.98))
         .then(20.0).otherwise(0), pl.lit("вызовы идут один за другим без наложений (конвейер)")),
        ("young", pl.when((pl.col("n_days") <= 2) & (n_out >= 40)).then(10.0).otherwise(0),
         pl.concat_str([pl.lit("номер активен "), _i("n_days"), pl.lit(" дн.")])),
        # --- SIM / IMSI
        ("imsi", pl.when(pl.col("n_imsi") >= 3).then(50 + 4 * pl.min_horizontal(pl.lit(5), pl.col("n_imsi") - 3)).otherwise(0),
         pl.concat_str([_i("n_imsi"), pl.lit(" разных IMSI на одном номере")])),
        ("simult", pl.when((pl.col("n_imsi") >= 2) & (pl.col("n_overlap") > 0)).then(25.0).otherwise(0),
         pl.concat_str([_i("n_overlap"), pl.lit(" одновременных вызовов с разных SIM")])),
        # --- вишинг
        ("longonly", pl.when((pl.col("sh_long") >= 0.8) & (n_out >= 5) & (n_in <= 2)).then(35.0).otherwise(0),
         pl.concat_str([_i("n_out"), pl.lit(" звонков по 7–25 мин, входящих: "), _i("n_in")])),
        ("prelude", 45 * _clip01(pl.col("m_prelude").fill_null(0) / 2.5),
         pl.concat_str([pl.lit("перед звонком у жертвы сбросы с «городских» (в среднем "),
                        pl.col("m_prelude").fill_null(0).round(1).cast(pl.Utf8), pl.lit(" номера)")])),
        ("victims", pl.when((pl.col("m_prelude").fill_null(0) > 0) & (pl.col("m_victims").fill_null(0) >= 5)).then(15.0).otherwise(0),
         pl.concat_str([_i("m_victims"), pl.lit(" разных жертв, каждой — один длинный звонок")])),
        ("spoof", 75 * pl.col("m_spoof_share").fill_null(0),
         pl.concat_str([_p("m_spoof_share"), pl.lit("% звонков — будущим жертвам вишинга ("),
                        _i("m_victims"), pl.lit(" жертв)")])),
        ("spoofdrop", pl.when(pl.col("m_spoof_share").fill_null(0) > 0).then(20 * _clip01(pl.col("sh02") * 2)).otherwise(0),
         pl.concat_str([pl.lit("короткие звонки и сбросы, медиана "), _i("dur_med"), pl.lit(" с")])),
        # --- Wangiri
        ("callback", pl.when(pl.col("m_callbacks").fill_null(0) > 0).then(15.0).otherwise(0),
         pl.concat_str([_i("m_callbacks"), pl.lit(" перезвонов жертв уходят на платный номер")])),
        ("premium", pl.when(pl.col("m_premium_hits").fill_null(0) > 0).then(80.0).otherwise(0),
         pl.concat_str([pl.lit("принимает переадресованные перезвоны жертв Wangiri ("), _i("m_premium_hits"), pl.lit(")")])),
        ("premium_hidden", pl.when((pl.col("m_premium_hits").fill_null(0) > 0) & (n_out == 0)).then(20.0).otherwise(0),
         pl.lit("сам не звонит, появляется только как получатель переадресаций")),
        # --- переадресации
        ("chains", pl.when(pl.col("m_chains").fill_null(0) >= 3)
         .then(65 + 20 * _clip01(pl.col("m_chains").fill_null(0) / 50)).otherwise(0),
         pl.concat_str([pl.lit("узел в "), _i("m_chains"), pl.lit(" цепочках переадресаций (до "),
                        _i("m_chain_len"), pl.lit(" звеньев)")])),
        ("rings", pl.when(pl.col("m_rings").fill_null(0) > 0).then(20.0).otherwise(0),
         pl.concat_str([_i("m_rings"), pl.lit(" цепочек замкнуты в кольцо с нулевой длительностью")])),
        ("chain_silent", pl.when((pl.col("m_chains").fill_null(0) >= 3) & (n_out <= 2)).then(10.0).otherwise(0),
         pl.lit("своих звонков почти нет — только транзит")),
        # --- карусель
        ("closed", 70 * _clip01((pl.col("m_closure").fill_null(0) - 0.9) / 0.1),
         pl.concat_str([pl.lit("закрытая группа из "), _i("m_group_size"), pl.lit(" номеров: "),
                        _p("m_closure"), pl.lit("% звонков внутри")])),
        ("closed247", pl.when(pl.col("m_closure").fill_null(0) >= 0.98).then(20.0).otherwise(0),
         pl.concat_str([pl.lit("звонят друг другу круглосуточно, медиана "),
                        pl.col("m_group_dur").fill_null(0).round(0).cast(pl.Int64).cast(pl.Utf8), pl.lit(" с")])),
        # --- кампания роботов
        ("campaign", pl.when((pl.col("m_group_size").fill_null(0) >= 2) & (pl.col("sh_seq").fill_null(0) >= 0.5))
         .then(10.0).otherwise(0),
         pl.concat_str([pl.lit("работает в связке из "), _i("m_group_size"), pl.lit(" номеров по одному диапазону")])),
        # --- признаки легального поведения (минус)
        ("has_in", pl.when(((n_in - pl.col("n_fwd_as_b")) >= 20) & (pl.col("in_out_ratio") >= 0.1)
                           & ~fraud_ctx).then(-15.0).otherwise(0),
         pl.concat_str([pl.lit("есть входящие ("), _i("n_in"), pl.lit(") — с номером общаются")])),
        ("repeat", pl.when((pl.col("sh_repeat_b").fill_null(0) >= 0.3) & ~fraud_ctx)
         .then(-10.0).otherwise(0),
         pl.concat_str([_p("sh_repeat_b"), pl.lit("% собеседников — постоянные")])),
        ("workhours", pl.when((pl.col("sh_work").fill_null(0) >= 0.85) & (n_out >= 30) & (pl.col("n_days") >= 2)
                              & (pl.col("dur_med").fill_null(0) >= 20)).then(-15.0).otherwise(0),
         pl.concat_str([_p("sh_work"), pl.lit("% вызовов в рабочие часы")])),
        ("waves", pl.when((pl.col("top3_hour_share").fill_null(0) >= 0.95) & (pl.col("n_days") >= 4) & (n_out >= 100))
         .then(-25.0).otherwise(0),
         pl.lit("волны в одни и те же часы каждый день (расписание рассылки)")),
        ("legit", pl.when(pl.col("legit").is_not_null()).then(-50.0).otherwise(0),
         pl.concat_str([pl.lit("профиль легального источника: "), pl.col("legit_ru").fill_null(""),
                        pl.lit(" — "), pl.col("legit_why").fill_null("")])),
    ]


FACTOR_KEYS = [k for k, _, _ in factor_defs()]


def score_numbers(f: pl.DataFrame, hits: pl.DataFrame, rel: pl.DataFrame, legit: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Оценка для всех номеров. Возвращает таблицу с pts_*, txt_*, score, scheme, role, рекомендацией."""
    lg = legit.join(hits.select("num"), on="num", how="anti").with_columns(
        pl.col("legit").replace_strict(LEGIT_RU, default="").alias("legit_ru")).rename({"why": "legit_why"})
    x = f.join(rel, on="num", how="left").join(hits, on="num", how="left").join(lg, on="num", how="left")
    defs = factor_defs()
    x = x.with_columns([p.fill_null(0).round(0).cast(pl.Int64).alias(f"pts_{k}") for k, p, _ in defs])
    x = x.with_columns([t.alias(f"txt_{k}") for k, _, t in defs])
    raw = pl.sum_horizontal([pl.col(f"pts_{k}") for k in FACTOR_KEYS])
    sc = cfg["scoring"]
    x = x.with_columns(raw.alias("score_raw"), pl.min_horizontal(pl.lit(100), pl.max_horizontal(pl.lit(0), raw)).alias("score"))
    x = x.with_columns(
        pl.sum_horizontal([(pl.col(f"pts_{k}") >= sc["factor_min_points"]).cast(pl.Int32) for k in FACTOR_KEYS]).alias("n_strong"),
        pl.col("scheme").replace_strict(SCHEME_RU, default=None).alias("scheme_ru"),
        pl.col("role").replace_strict(ROLE_RU, default=None).alias("role_ru"),
    )
    return apply_policy(x, cfg)


# ---------------- политика ----------------
ROLE_ACTION = {
    "wangiri_source": ("заблокировать исходящие; разослать жертвам «не перезванивайте»", True),
    "premium_C": ("закрыть переадресации и перезвоны на номер, приостановить взаиморасчёты", True),
    "robot_dialer": ("ограничить скорость исходящих, пометить абонентам «возможен спам»", False),
    "simbox_sim": ("заблокировать SIM, передать в отдел межоператорского трафика", True),
    "cloned_msisdn": ("заблокировать лишние IMSI, перевыпустить SIM законному владельцу", False),
    "forward_chain_node": ("отключить переадресацию на номере", True),
    "carousel_member": ("заблокировать группу, проверить начисления и взаиморасчёты", True),
    "vishing_caller": ("заблокировать номер, предупредить жертв", True),
    "spoofed_city": ("НЕ блокировать номер: маркировать вызовы с ним из чужих сетей «возможна подмена номера»", False),
}
NO_BLOCK_REASON = {
    "cloned_msisdn": "у номера есть законный владелец — блокируются только чужие IMSI",
    "spoofed_city": "номер подменяют мошенники, а принадлежать он может честной организации",
    "robot_dialer": "массовый обзвон — сначала ограничение и проверка, блок — решением аналитика",
}


def apply_policy(x: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    sc, pol = cfg["scoring"], cfg["policy"]
    white = set(pol.get("social_whitelist", [])) | set(pol.get("emergency_numbers", []))
    act = {k: v[0] for k, v in ROLE_ACTION.items()}
    can = [k for k, v in ROLE_ACTION.items() if v[1]]
    s = pl.col("score")
    x = x.with_columns(pl.col("num").is_in(list(white)).alias("whitelisted"))
    rec = (pl.when(pl.col("whitelisted")).then(pl.lit("белый список: никаких ограничений, вызовы всегда проходят"))
           .when(s >= sc["review_threshold"]).then(
               pl.concat_str([pl.lit("пометить абоненту + проверка аналитиком"),
                              pl.when(pl.col("role").is_not_null())
                              .then(pl.concat_str([pl.lit("; "), pl.col("role").replace_strict(act, default="")]))
                              .otherwise(pl.lit(""))]))
           .when(s >= sc["mark_threshold"]).then(pl.lit("пометить абоненту «возможно мошенничество»"))
           .otherwise(pl.lit("нет действий")))
    auto = ((~pl.col("whitelisted")) & (s >= sc["strong_threshold"]) & (pl.col("n_strong") >= sc["autoblock_min_factors"])
            & pl.col("role").is_in(can))
    reason = (pl.when(pl.col("whitelisted")).then(pl.lit("номер в белом списке (экстренные/социальные службы)"))
              .when(auto).then(pl.concat_str([pl.lit("оценка "), s.cast(pl.Utf8), pl.lit(", независимых факторов: "),
                                              pl.col("n_strong").cast(pl.Utf8), pl.lit("; журнал и отмена в 1 клик")]))
              .when(pl.col("role").is_in(list(NO_BLOCK_REASON))).then(pl.col("role").replace_strict(NO_BLOCK_REASON, default=""))
              .when(s >= sc["strong_threshold"]).then(pl.lit("высокая оценка, но схема не подтверждена детектором — нужна проверка"))
              .otherwise(pl.concat_str([pl.lit("оценка "), s.cast(pl.Utf8), pl.lit(" ниже порога "),
                                        pl.lit(str(sc["strong_threshold"]))])))
    return x.with_columns(rec.alias("recommendation"), auto.alias("autoblock"), reason.alias("autoblock_reason"))


def explain(row: dict, top: int | None = None) -> list[dict]:
    """Список факторов номера с вкладом, по убыванию модуля вклада."""
    fs = [{"key": k, "points": int(row[f"pts_{k}"]), "text": row[f"txt_{k}"]}
          for k in FACTOR_KEYS if row.get(f"pts_{k}")]
    fs.sort(key=lambda d: -abs(d["points"]))
    return fs[:top] if top else fs


def is_emergency(num: str, cfg: dict) -> bool:
    return num in set(cfg["policy"].get("emergency_numbers", [])) | set(cfg["policy"].get("social_whitelist", []))
