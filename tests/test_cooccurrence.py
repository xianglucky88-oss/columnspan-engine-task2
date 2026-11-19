"""Tests for revisable label co-occurrence analysis.

The hard invariants exercised here:

* the object universe, observed/positive sets and test family are fixed;
* positive withdrawals change counts, eligibility withdrawals also change the
  denominator (joint eligible set);
* corrections are idempotent and baseline-checked, never double-accumulated;
* only affected-label counts are recomputed, but the whole fixed family is
  re-corrected and every published version is cross-checked by an independent
  full recomputation;
* unmodified pairs can flip significance purely through family re-ranking;
* blank/constant/sparse pairs keep a reason, never become zero-association
  results and never leave the family denominator;
* interrupted corrections leave the old version queryable and continuation is
  bound to the original universe and evidence identity.
"""
import math
import random
from fractions import Fraction

import pytest

from columnspan import bitmap, cooccurrence as co
from columnspan.cooccurrence import (
    LabelStudy, StaleBaseline, DuplicateCorrection, CorrectionInterrupted,
    IdentityMismatch, CooccurrenceError, family_correction,
    fisher_two_sided, REASON_BLANK, REASON_CONSTANT, REASON_SPARSE,
)


# --------------------------------------------------------------- statistics
def exact_fisher(a, b, c, d):
    n = a + b + c + d
    row1, col1 = a + b, a + c
    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)

    def comb(nn, kk):
        return Fraction(math.comb(nn, kk)) if 0 <= kk <= nn else Fraction(0)

    def pr(x):
        return comb(col1, x) * comb(n - col1, row1 - x) / comb(n, row1)

    observed = pr(a)
    return float(sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= observed))


def test_fisher_matches_exact_rational_enumeration():
    rng = random.Random(11)
    # canonical tea-tasting table
    assert fisher_two_sided(4, 0, 0, 4) == pytest.approx(2 / 70)
    for _ in range(1500):
        n = rng.randint(1, 60)
        a = rng.randint(0, n)
        b = rng.randint(0, n - a)
        c = rng.randint(0, n - a - b)
        d = n - a - b - c
        assert fisher_two_sided(a, b, c, d) == pytest.approx(
            exact_fisher(a, b, c, d), abs=1e-12)
    # degenerate fixed-margin table is exactly 1.0
    assert fisher_two_sided(14, 3, 5, 0) == pytest.approx(1.0)


def test_holm_and_bh_hand_calculations():
    holm = family_correction([0.01, 0.02, 0.03], "holm", 0.05)
    assert [round(x["p_adj"], 6) for x in holm] == [0.03, 0.04, 0.04]
    assert [x["rank"] for x in holm] == [1, 2, 3]
    bh = family_correction([0.01, 0.02, 0.03], "bh", 0.05)
    assert [round(x["p_adj"], 6) for x in bh] == [0.03, 0.03, 0.03]
    # step-up monotonicity: raw .03,.075,.05 becomes .03,.05,.05
    bh2 = family_correction([0.01, 0.05, 0.05], "bh", 0.05)
    assert [round(x["p_adj"], 6) for x in bh2] == [0.03, 0.05, 0.05]


def test_non_estimable_pairs_remain_in_family_denominator():
    # family size m=3 even though only two entries have a p-value; the gap
    # still consumes one of Holm's multipliers and one of BH's m/rank factors
    holm = family_correction([0.01, None, 0.03], "holm", 0.05)
    assert holm[1] is None
    assert holm[0]["p_adj"] == pytest.approx(0.03)   # 3 * 0.01
    assert holm[2]["p_adj"] == pytest.approx(0.06)   # 2 * 0.03
    assert [holm[0]["rank"], holm[2]["rank"]] == [1, 2]
    bh = family_correction([0.01, None, 0.03], "bh", 0.05)
    assert bh[0]["p_adj"] == pytest.approx(0.03)
    assert bh[2]["p_adj"] == pytest.approx(0.045)
    assert bh[1] is None


# ------------------------------------------------------------------ building
def _study_three():
    # Deterministic engineered scenario (n=25, every object eligible).
    a = [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]
    b = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    c = [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    pos = lambda seq: [i for i, v in enumerate(seq) if v]
    return LabelStudy(
        universe=list(range(25)),
        labels=[
            {"name": "A", "eligible": list(range(25)), "positive": pos(a)},
            {"name": "B", "eligible": list(range(25)), "positive": pos(b)},
            {"name": "C", "eligible": list(range(25)), "positive": pos(c)},
        ])


def test_universe_and_family_are_fixed():
    study = _study_three()
    assert study.family["family_size"] == 3
    assert study.family["pairs"] == [["A", "B"], ["A", "C"], ["B", "C"]]
    fingerprint = study.universe_fingerprint
    study.apply_corrections({"correction_id": "w", "base_version": 0, "changes": [
        {"kind": "positive_withdraw", "label": "A", "object": 0}]})
    assert study.universe_fingerprint == fingerprint
    assert study.family["family_size"] == 3
    with pytest.raises(CooccurrenceError):
        study.affected_scope([{"label": "NOPE", "kind": "x", "object": 0}])
    with pytest.raises(CooccurrenceError):
        study.apply_corrections({"correction_id": "oob", "base_version": 1,
                                 "changes": [{"kind": "positive_withdraw",
                                              "label": "A", "object": 99}]})
    with pytest.raises(CooccurrenceError):
        LabelStudy(universe=[0, 1], labels=[
            {"name": "x", "eligible": [0], "positive": [1]}])


def test_genesis_pair_delivers_samples_effect_and_family_evidence():
    study = _study_three()
    row = study.pair_result("B", "C", version=0)
    assert row["estimable"] and row["n_valid"] == 25
    assert sum(row["table"].values()) == 25
    assert row["p_value"] == pytest.approx(0.03300579, abs=1e-6)
    assert row["adjusted_p"] is not None and row["rank"] == 3
    assert row["family_size"] == 3
    assert row["odds_ratio"] > 0 and row["log_odds_ratio"] == pytest.approx(
        math.log(row["odds_ratio"]))
    assert row["direction"] in {"positive", "negative", "none"}


def test_positive_withdraw_recomputes_only_affected_but_rechecks_whole_family():
    study = _study_three()
    before = study.result(0)
    scope = study.affected_scope([
        {"kind": "positive_withdraw", "label": "A", "object": 0}])
    assert scope["affected_labels"] == ["A"]
    assert scope["affected_pairs"] == [["A", "B"], ["A", "C"]]
    version = study.apply_corrections(
        {"correction_id": "w0", "base_version": 0,
         "changes": [{"kind": "positive_withdraw", "label": "A", "object": 0}]})
    assert version == 1
    after = study.result(1)
    assert after["affected_pairs"] == [["A", "B"], ["A", "C"]]
    # untouched pair keeps identical counts ...
    bc0 = next(r for r in before["results"] if r["pair"] == ["B", "C"])
    bc1 = next(r for r in after["results"] if r["pair"] == ["B", "C"])
    assert bc0["table"] == bc1["table"] and bc0["n_valid"] == bc1["n_valid"]
    # ... yet its whole-family correction changes and its conclusion flips
    assert bc0["significant"] is True and bc1["significant"] is False
    assert bc0["adjusted_p"] != bc1["adjusted_p"]
    diff = study.diff_versions(0, 1)
    bc_change = next(c for c in diff["pair_changes"] if c["pair"] == ["B", "C"])
    assert bc_change["directly_modified"] is False
    assert "re-ranked" in bc_change["explanation"]
    ab_change = next(c for c in diff["pair_changes"] if c["pair"] == ["A", "B"])
    assert ab_change["directly_modified"] is True
    # independent full recomputation confirms the published version
    assert study.verify_version(1)["ok"]


def test_eligibility_withdraw_changes_denominator_and_backfill_restores():
    study = LabelStudy(universe=list(range(8)), labels=[
        {"name": "X", "eligible": [0, 1, 2, 3, 4, 5], "positive": [0, 1, 2]},
        {"name": "Y", "eligible": [1, 2, 3, 4, 5, 6], "positive": [1, 2, 6]}])
    before = study.pair_result("X", "Y")
    assert before["n_valid"] == 5 and before["table"]["both_positive"] == 2
    study.apply_corrections({"correction_id": "e1", "base_version": 0, "changes": [
        {"kind": "eligibility_withdraw", "label": "X", "object": 1}]})
    after = study.pair_result("X", "Y", version=1)
    assert after["n_valid"] == 4
    assert after["table"]["both_positive"] == 1
    # the withdrawn object also leaves X's own positive set
    assert 1 not in study._labels["X"].positive.values()
    study.apply_corrections({"correction_id": "e2", "base_version": 1, "changes": [
        {"kind": "eligibility_add", "label": "X", "object": 1}]})
    study.apply_corrections({"correction_id": "e3", "base_version": 2, "changes": [
        {"kind": "positive_add", "label": "X", "object": 1}]})
    restored = study.pair_result("X", "Y", version=3)
    assert restored["table"] == before["table"]
    assert restored["n_valid"] == before["n_valid"]
    for v in range(4):
        assert study.verify_version(v)["ok"]
    # positive back-fill is refused while the object is not eligible
    study.apply_corrections({"correction_id": "e4", "base_version": 3, "changes": [
        {"kind": "eligibility_withdraw", "label": "X", "object": 0}]})
    with pytest.raises(CooccurrenceError):
        study.apply_corrections({"correction_id": "bad", "base_version": 4, "changes": [
            {"kind": "positive_add", "label": "X", "object": 0}]})
    assert study.current_version == 4


def test_duplicate_and_baseline_rules_never_double_accumulate():
    study = _study_three()
    batch = {"correction_id": "w0", "base_version": 0,
             "changes": [{"kind": "positive_withdraw", "label": "A", "object": 0}]}
    assert study.apply_corrections(batch) == 1
    # identical retry (even carrying the stale original base) returns v1
    assert study.apply_corrections(dict(batch)) == 1
    assert len(study._versions) == 2
    snap = study.result(1)
    # same id, different content is rejected
    with pytest.raises(DuplicateCorrection):
        study.apply_corrections({"correction_id": "w0", "base_version": 1,
                                 "changes": [{"kind": "positive_withdraw",
                                              "label": "A", "object": 1}]})
    # stale baseline is rejected without mutating state
    with pytest.raises(StaleBaseline):
        study.apply_corrections({"correction_id": "x", "base_version": 0,
                                 "changes": [{"kind": "positive_withdraw",
                                              "label": "A", "object": 1}]})
    assert study.current_version == 1
    assert study.result(1)["evidence_hash"] == snap["evidence_hash"]


def test_blank_constant_and_sparse_keep_reasons_and_stay_in_family():
    study = LabelStudy(
        universe=list(range(6)),
        labels=[
            {"name": "   ", "eligible": [0, 1], "positive": [0]},
            {"name": "const", "eligible": [0, 1, 2, 3], "positive": [0, 1, 2, 3]},
            {"name": "sparse", "eligible": [4], "positive": [4]},
            {"name": "ok", "eligible": [0, 1, 2, 3, 4], "positive": [0, 2, 4]},
        ], min_joint_observations=2)
    envelope = study.result(0)
    assert envelope["family_size"] == 6
    assert all(row["family_size"] == 6 for row in envelope["results"])
    reasons = {tuple(row["pair"]): row["reason"] for row in envelope["results"]}
    assert reasons[("   ", "const")] == REASON_BLANK
    assert reasons[("const", "ok")] == REASON_CONSTANT
    assert reasons[("sparse", "ok")] == REASON_SPARSE
    for row in envelope["results"]:
        if not row["estimable"]:
            assert row["p_value"] is None and row["adjusted_p"] is None
            assert row["significant"] is False
    # a correction cannot promote a non-estimable pair into a fabricated zero
    study.apply_corrections({"correction_id": "c", "base_version": 0, "changes": [
        {"kind": "positive_withdraw", "label": "ok", "object": 4}]})
    assert study.verify_version(1)["ok"]
    assert study.result(1)["family_size"] == 6


def test_interrupted_correction_keeps_old_version_and_requires_bound_resume():
    study = _study_three()
    token = study.identity_token()
    batch = {"correction_id": "i1", "base_version": 0,
             "changes": [{"kind": "positive_withdraw", "label": "A", "object": 0}]}
    tx = study.correction_transaction(batch)
    with pytest.raises(CorrectionInterrupted):
        with tx:
            tx.run()
            tx.interrupt()
    assert study.current_version == 0
    old = study.result(0)
    assert len(old["results"]) == 3  # old version fully queryable
    bad_token = dict(token, evidence_head="0" * 64)
    with pytest.raises(IdentityMismatch):
        study.apply_corrections(
            {"correction_id": "i1-resume", "base_version": 0,
             "changes": batch["changes"]}, identity_token=bad_token)
    assert study.current_version == 0
    resume_token = study.identity_token()
    version = study.apply_corrections(
        {"correction_id": "i1-resume", "base_version": 0,
         "changes": batch["changes"]}, identity_token=resume_token)
    assert version == 1 and study.verify_version(1)["ok"]


def test_serialization_roundtrip_and_tamper_detection():
    study = _study_three()
    study.apply_corrections({"correction_id": "w0", "base_version": 0, "changes": [
        {"kind": "positive_withdraw", "label": "A", "object": 0}]})
    payload = study.to_dict()
    restored = LabelStudy.from_dict(payload)
    for v in range(len(study._versions)):
        assert co._canonical(study.result(v)) == co._canonical(restored.result(v))
    assert restored.verify_version()["ok"]
    tampered = study.to_dict()
    tampered["evidence"][-1]["hash"] = "0" * 64
    with pytest.raises(IdentityMismatch):
        LabelStudy.from_dict(tampered)
    tampered_fp = study.to_dict()
    tampered_fp["versions"][-1]["universe_fingerprint"] = "ff"
    with pytest.raises(IdentityMismatch):
        LabelStudy.from_dict(tampered_fp)


def test_scoped_path_matches_independent_full_recomputation_randomly():
    rng = random.Random(2026)
    for trial in range(60):
        n = rng.randint(8, 45)
        labels = []
        for li in range(rng.randint(2, 6)):
            eligible = sorted(rng.sample(range(n), rng.randint(max(2, n // 3), n)))
            positive = sorted(rng.sample(eligible, rng.randint(0, len(eligible))))
            labels.append({"name": f"L{li}", "eligible": eligible,
                           "positive": positive})
        study = LabelStudy(
            universe=list(range(n)), labels=labels,
            method=rng.choice(["holm", "bh"]),
            min_joint_observations=rng.choice([1, 2, 3]))
        study.verify_version(0)
        for b in range(rng.randint(1, 5)):
            changes = [{
                "kind": rng.choice(["positive_add", "positive_withdraw",
                                    "eligibility_add", "eligibility_withdraw"]),
                "label": rng.choice(labels)["name"],
                "object": rng.randrange(n)} for _ in range(rng.randint(1, 3))]
            before = study.current_version
            try:
                version = study.apply_corrections(
                    {"correction_id": f"{trial}-{b}",
                     "base_version": study.current_version, "changes": changes})
            except CooccurrenceError:
                assert study.current_version == before
                continue
            # every version: scoped publication must equal full recomputation
            assert study.verify_version(version)["ok"]


def test_verify_detects_spliced_count_and_old_correction():
    study = _study_three()
    study.apply_corrections({"correction_id": "w0", "base_version": 0, "changes": [
        {"kind": "positive_withdraw", "label": "A", "object": 0}]})
    assert study.verify_version(1)["ok"]
    # Tamper with a stored count without rebuilding the correction: a spliced
    # result must be caught by the independent full recomputation.
    tampered = study.to_dict()
    row = tampered["versions"][1]["results"][0]
    row["table"]["both_positive"] += 1
    restored = LabelStudy.from_dict(tampered)
    with pytest.raises(CooccurrenceError):
        restored.verify_version(1)
    # Tampering only with an adjusted p-value (old correction kept on new
    # counts) is equally detectable.
    tampered2 = study.to_dict()
    tampered2["versions"][1]["results"][1]["adjusted_p"] = 0.0001
    restored2 = LabelStudy.from_dict(tampered2)
    with pytest.raises(CooccurrenceError):
        restored2.verify_version(1)


def test_bitmap_api_is_reused_for_counts():
    study = LabelStudy(universe=list(range(10)), labels=[
        {"name": "A", "eligible": list(range(10)), "positive": [0, 1, 2, 3]},
        {"name": "B", "eligible": list(range(10)), "positive": [2, 3, 4, 5]}])
    # internal state really is the existing Bitmap implementation
    assert isinstance(study._labels["A"].eligible, bitmap.Bitmap)
    a, b, c, d, n_valid = study._pair_counts("A", "B")
    assert (a, b, c, d, n_valid) == (2, 2, 2, 4, 10)


def test_api_cooccurrence_endpoint_state_machine():
    from columnspan import api
    created = api.cooccurrence_call({
        "action": "create", "universe": [0, 1, 2, 3, 4],
        "labels": [
            {"name": "A", "eligible": [0, 1, 2, 3, 4], "positive": [0, 1, 2]},
            {"name": "B", "eligible": [0, 1, 2, 3, 4], "positive": [0, 1, 3]}]})
    sid = created["study_id"]
    scope = api.cooccurrence_call({"study_id": sid, "action": "scope",
                                   "changes": [{"kind": "eligibility_withdraw",
                                                "label": "A", "object": 0}]})
    assert scope["affected_pairs"] == [["A", "B"]]
    corrected = api.cooccurrence_call({
        "study_id": sid, "action": "correct",
        "batch": {"correction_id": "w", "base_version": 0,
                  "changes": [{"kind": "eligibility_withdraw",
                               "label": "A", "object": 0}]}})
    assert corrected["version"]["results"][0]["n_valid"] == 4
    verify = api.cooccurrence_call({"study_id": sid, "action": "verify"})
    assert verify["ok"]
    old = api.cooccurrence_call({"study_id": sid, "action": "result", "version": 0})
    assert old["version"]["results"][0]["n_valid"] == 5
    payload = api.cooccurrence_call({"study_id": sid, "action": "export"})
    restored = api.cooccurrence_call({"action": "restore", "study": payload})
    assert restored["identity"]["version"] == 1
    assert api.cooccurrence_call(
        {"study_id": restored["study_id"], "action": "verify"})["ok"]
    with pytest.raises(CooccurrenceError):
        api.cooccurrence_call({"study_id": "missing", "action": "verify"})
    with pytest.raises(CooccurrenceError):
        api.cooccurrence_call({"action": "frobnicate"})
