"""Тесты: очистка, детекторы на синтетике с разметкой, политика, сервис.

Запуск: python -m pytest -q
"""
import io
import os
from datetime import datetime

import polars as pl
import pytest

from antifraud.config import load_config
from antifraud.detectors import run_detectors
from antifraud.features import build_features
from antifraud.ingest import parse
from antifraud.scoring import score_numbers
from antifraud.synth import Synth

CFG = load_config()


# ---------------- очистка ----------------
def _parse(text):
    return parse([("t.cdr", io.StringIO(text))], CFG)


def test_ingest_fixes_and_quarantine():
    lines = "\n".join([
        "02;250021404312271;8102243665;4991844747;20261002120037;14;",      # норма
        "02;250021404312271;+78102243665;4991844747;20261002120137;14;",    # код страны
        "01;;4991490520;0265675038;2026-10-01 06:55:41;65;",                 # другой формат даты
        "2;250021404312271;8102243665;4991844747;20261002120237;5;",         # тип без нуля
        "03;;8428913184;8995769997;20261001053930;8994878312",               # пропущена длительность у 03
        "02;250021404312271;8102243665;4991844747;20261002120337;7;;X123",   # лишние поля
        "AA;;8926136351;8105997735;20261001072744;0;",                       # битый тип
        "02;250027489216671;8999596816;;20261001034901;636;",                # пустой B
        "02;250010200173884;0267681872;8928293141;20261301055036;83;",       # 13-й месяц
        "02;250027549269733;8343067253;8104553060;20261001012502;abc;",      # длительность не число
        "01;;7774548221;8922990301;20261001013217;86;",                      # неизвестный код 777
        "02;25001;8102243665;4991844747;20261002120437;14;",                 # обрезанный IMSI
        "02;250021404312271;8102243665;4991844747;20261002120037;14;",       # точный дубль
    ])
    df, rep = _parse(lines)
    assert df.height == 7
    q = rep.as_dict()["quarantine_reasons"]
    assert q == {"bad_type": 1, "empty_number": 1, "bad_datetime": 1, "bad_duration": 1, "unknown_operator_code": 1}
    fx = rep.fixes
    assert fx["number_country_code"] == 1 and fx["datetime_format"] == 1 and fx["type_leading_zero"] == 1
    assert fx["missing_duration"] == 1 and fx["exact_duplicates_removed"] == 1 and fx["truncated_imsi_nulled"] == 1
    fwd = df.filter(pl.col("type") == "03").row(0, named=True)
    assert fwd["c"] == "8994878312" and fwd["dur"] is None
    assert df["a"].str.len_chars().eq(10).all()          # ведущие нули и длина сохранены


def test_leading_zero_numbers_kept():
    df, _ = _parse("02;250021404312271;0261234567;0251234567;20261002120037;14;")
    assert df["a"][0] == "0261234567" and df["b"][0] == "0251234567"


# ---------------- детекторы на синтетике ----------------
@pytest.fixture(scope="module")
def synth_run():
    s = Synth(seed=11, n_users=2500, n_city=500).build()
    df, _ = parse(s.sources(), CFG)
    f = build_features(df, CFG)
    hits, rel, legit = run_detectors(df, f, CFG)
    legit = legit.join(hits.select("num"), on="num", how="anti")
    return s, df, f, hits, rel, legit


def test_every_scheme_found_without_false_positives(synth_run):
    s, _, _, hits, _, _ = synth_run
    pred = dict(zip(hits["num"], hits["scheme"]))
    for num, (scheme, _) in s.truth.items():
        assert pred.get(num) == scheme, f"{scheme}: пропущен {num}"
    fp = set(pred) - set(s.truth)
    assert not fp, f"ложные срабатывания: {sorted(fp)[:5]}"


def test_legit_traps_untouched(synth_run):
    s, _, _, hits, _, _ = synth_run
    assert not (set(hits["num"]) & set(s.legit))


def test_scores_separate_fraud_from_rest(synth_run):
    s, _, f, hits, rel, legit = synth_run
    sc = score_numbers(f, hits, rel, legit, CFG)
    d = dict(zip(sc["num"], sc["score"]))
    assert min(d[n] for n in s.truth) >= 50
    assert max(v for n, v in d.items() if n not in s.truth) < 50


# ---------------- политика ----------------
def test_policy_never_autoblocks_innocent_owners(synth_run):
    s, _, f, hits, rel, legit = synth_run
    sc = score_numbers(f, hits, rel, legit, CFG)
    for role in ("spoofed_city", "cloned_msisdn", "robot_dialer"):
        assert not sc.filter(pl.col("role") == role)["autoblock"].any(), role
    assert not sc.filter(pl.col("legit").is_not_null())["autoblock"].any()


def test_whitelist_blocks_nothing(synth_run):
    s, _, f, hits, rel, legit = synth_run
    num = next(n for n, (sch, r) in s.truth.items() if r == "wangiri_source")
    cfg = {**CFG, "policy": {**CFG["policy"], "social_whitelist": [num]}}
    row = score_numbers(f, hits, rel, legit, cfg).filter(pl.col("num") == num).row(0, named=True)
    assert row["whitelisted"] and not row["autoblock"]


# ---------------- сервис ----------------
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    src = tmp_path_factory.mktemp("cdr")
    s = Synth(seed=5, n_users=1500, n_city=300).build()
    for name, buf in s.sources():
        (src / name).write_text(buf.getvalue(), encoding="utf-8")
    from antifraud.pipeline import run
    run(str(src), str(out), with_ml=False, with_realtime=True)
    os.environ["ANTIFRAUD_OUT"] = str(out)
    import importlib
    import app.main as m
    importlib.reload(m)
    from fastapi.testclient import TestClient
    return TestClient(m.app), s


def test_api_summary_and_answer(client):
    c, s = client
    d = c.get("/api/summary").json()
    assert d["summary"]["answers"]["full"] == len(s.truth)
    txt = c.get("/api/answer").text
    nums = [l.split(";")[0] for l in txt.splitlines() if l and not l.startswith("#")]
    assert set(nums) == set(s.truth)


def test_api_number_card_explains(client):
    c, s = client
    num = next(n for n, (_, r) in s.truth.items() if r == "simbox_sim")
    card = c.get(f"/api/number/{num}").json()
    assert card["score"] >= 80 and card["factors"] and card["scheme"] == "S3"
    assert all({"points", "text"} <= set(f) for f in card["factors"])


def test_api_emergency_call_always_connects(client):
    c, s = client
    fraud = next(n for n, (_, r) in s.truth.items() if r == "wangiri_source")
    r = c.post("/api/score_call", json={"a": fraud, "b": "112", "ts": "2026-10-01T10:00:00", "dur": 30}).json()
    assert r["autoblock"] is False and "соединить" in r["recommendation"]
    assert r["score"] >= 80                     # риск звонящего виден, но вызов проходит


def test_api_unknown_number_404(client):
    c, _ = client
    assert c.get("/api/number/0000000000").status_code == 404


# ---------------- регрессии найденных ошибок ----------------
def test_six_fields_with_duration_but_no_c():
    df, rep = _parse("02;250021404312271;8102243665;4991844747;20261002120037;14")
    assert df["dur"][0] == 14 and df["c"][0] is None and rep.fixes["missing_trailing_field"] == 1


def test_pipeline_tiny_and_broken_inputs(tmp_path):
    from antifraud.pipeline import run
    tiny = tmp_path / "tiny"; tiny.mkdir()
    (tiny / "a.cdr").write_text("02;250021404312271;8102243665;4991844747;20261002120037;14;\n", encoding="utf-8")
    s = run(str(tiny), str(tmp_path / "o1"))                 # модель пропускается, прогон не падает
    assert s["answers"]["full"] == 0 and s["ml"] is None
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "a.cdr").write_text("garbage\nAA;;1;2;3;4;5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нет ни одной пригодной"):
        run(str(bad), str(tmp_path / "o2"))
    with pytest.raises(FileNotFoundError):
        run(str(tmp_path / "empty_dir_does_not_exist"), str(tmp_path / "o3"))


def test_api_sample_never_500_and_search_with_country_code(client):
    c, s = client
    for sch in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        for seed in range(8):
            assert c.get(f"/api/call/sample?scheme={sch}&seed={seed}").status_code == 200, sch
    num = next(n for n, (_, r) in s.truth.items() if r == "simbox_sim")
    for q in (num, "+7" + num, "8" + num, f"+7 {num[:3]} {num[3:6]}-{num[6:8]}-{num[8:]}"):
        assert c.get("/api/numbers", params={"q": q}).json()["total"] == 1, q


def test_api_score_call_validation(client):
    c, _ = client
    assert c.post("/api/score_call", json={"a": "", "b": "112", "ts": "2026-10-01T10:00:00"}).status_code == 422
    assert c.post("/api/score_call", json={"a": "8105266916", "b": "1", "ts": "2026-10-01T10:00:00", "dur": -1}).status_code == 422
    r = c.post("/api/score_call", json={"a": "+7 (999) 000-00-00", "b": "8 999 000 00 01", "ts": "2026-10-01T10:00:00Z"})
    assert r.status_code == 200 and r.json()["a"] == "9990000000"


# ---------------- база, журнал, экономика, модель на синтетике ----------------
def test_db_holds_run_and_journal(client):
    c, s = client
    h = c.get("/api/health").json()
    assert h["database_ok"] and h["database"].startswith("SQLite")
    d = c.get("/api/numbers", params={"kind": "fraud", "limit": 5}).json()
    assert d["source"] == "db" and d["total"] == len(s.truth)
    before = c.get("/api/decisions").json()["total"]
    fraud = next(n for n, (_, r) in s.truth.items() if r == "wangiri_source")
    r = c.post("/api/score_call", json={"a": fraud, "b": "112", "ts": "2026-10-01T10:00:00", "dur": 30}).json()
    assert r["logged"] and r["latency_ms"] >= 0
    j = c.get("/api/decisions").json()
    assert j["total"] == before + 1 and j["items"][0]["emergency"] and j["items"][0]["a"] == fraud
    g = c.get("/api/groups").json()
    assert g and all("members" in x for x in g)


def test_economics_counts_harm_and_prevention(client):
    c, _ = client
    rt = c.get("/api/summary").json()["realtime"]
    e = rt["economics"]
    by = {r["scheme"]: r for r in e["rows"]}
    assert set(by) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    assert by["S1"]["volume"] > 0 and 0 < by["S1"]["prevented_share"] <= 1
    assert e["prevented_rub"] <= e["loss_rub"] and e["loss_rub_year"] > e["loss_rub"]


def test_synth_model_is_independent_check(tmp_path):
    from antifraud.ml_synth import HAVE_CATBOOST
    if not HAVE_CATBOOST:
        pytest.skip("CatBoost не установлен")
    from antifraud.config import deep_update
    from antifraud.ml_synth import labelled_synth, run_synth_model
    cfg = deep_update(CFG, {"ml_synth": {"train_sets": 1, "iterations": 60, "synth_normal_sample": 4000,
                                         "real_background_sample": 3000}})
    real = labelled_synth(cfg, seed=55)                      # «реальные» данные — другой набор синтетики
    hits = real.filter(pl.col("cls").str.starts_with("S")).select("num", pl.col("cls").alias("scheme"),
                                                                  pl.lit("x").alias("role"))
    legit = real.filter(pl.col("cls").str.starts_with("D")).select("num", pl.col("cls").alias("legit"))
    pred, rep = run_synth_model(real.drop("cls"), hits, legit, cfg, str(tmp_path))
    assert pred.height == real.height and (tmp_path / "model_synth_0.cbm").exists()
    r = rep["synth_plus_real_bg"]["real"][0]
    assert r["rules_confirmed_share"] > 0.8 and r["legit_flagged"] <= 5


def test_unknown_number_is_no_data_not_zero(client):
    c, _ = client
    r = c.post("/api/score_call", json={"a": "9990001122", "b": "9991234567", "ts": "2026-10-01T10:00:00", "dur": 30}).json()
    assert r["score"] is None and r["unknown"] and r["autoblock"] is False and r["logged"]
