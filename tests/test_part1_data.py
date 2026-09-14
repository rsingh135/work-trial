"""Part 1 critical-path tests: split hygiene, train-only vocab fitting, target construction,
censoring masks, model interface contracts. Synthetic cases keep these fast and data-free;
one optional test touches the real bundle if present."""

from pathlib import Path

import numpy as np
import pytest

from bpm.data.cases import Case, split_chronological, split_random
from bpm.data.encoding import EOS, PAD, UNK, Encoder
from bpm.evaluate import evaluate
from bpm.metrics import dl_similarity, ece
from bpm.models.markov import MarkovModel


def make_case(cid, acts, start=0.0, step=3600.0, complete=True, role="EMPLOYEE"):
    T = len(acts)
    ts = start + np.arange(T) * step
    return Case(cid, list(acts), ts, np.zeros(T, int) + 9, np.zeros(T, int), {"org:role": [role] * T}, {"unit": "U1"}, {"Amount": 10.0}, complete, "bpi2020")


@pytest.fixture
def cases():
    rng = np.random.default_rng(0)
    out = []
    for i in range(60):
        acts = ["Declaration SUBMITTED by EMPLOYEE", "Declaration APPROVED by ADMINISTRATION", "Declaration FINAL_APPROVED by SUPERVISOR", "Request Payment", "Payment Handled"]
        if rng.random() < 0.3:
            acts = acts[:1] + ["Declaration REJECTED by ADMINISTRATION"] + acts
        out.append(make_case(f"c{i}", acts, start=i * 86400.0, complete=rng.random() > 0.1))
    return out


def test_splits_are_case_disjoint_and_deterministic(cases):
    s1, s2 = split_random(cases, seed=1), split_random(cases, seed=1)
    assert s1.fingerprint() == s2.fingerprint()
    assert set(s1.train).isdisjoint(s1.test) and set(s1.train).isdisjoint(s1.val) and set(s1.val).isdisjoint(s1.test)
    assert len(s1.train) + len(s1.val) + len(s1.test) == len(cases)
    assert split_random(cases, seed=2).fingerprint() != s1.fingerprint()


def test_chronological_split_orders_by_case_start(cases):
    s = split_chronological(cases)
    by = {c.case_id: c for c in cases}
    assert max(by[i].start for i in s.train) <= min(by[i].start for i in s.test)


def test_encoder_fits_on_train_only_and_maps_unseen_to_unk(cases):
    train = cases[:40]
    enc = Encoder("bpi2020", ["org:role"], ["unit"], ["Amount"], min_count_cat=1).fit(train)
    assert enc.act_vocab["<PAD>"] == PAD and enc.act_vocab["<UNK>"] == UNK and enc.act_vocab["<EOS>"] == EOS
    novel = make_case("x", ["Declaration SUBMITTED by EMPLOYEE", "Totally New Activity"], role="ALIEN")
    e = enc.transform(novel)
    assert e.act[1] == UNK and e.cat[1, 0] == UNK
    # decomposed components still carry signal for a *partially* novel label
    partial = make_case("y", ["Request For Payment APPROVED by ADMINISTRATION"])
    ep = enc.transform(partial)
    assert ep.act[0] == UNK
    assert ep.comps[0, 1] == enc.comp_vocab["act=APPROVED"] and ep.comps[0, 2] == enc.comp_vocab["actor=ADMINISTRATION"]
    assert "Totally New Activity" not in enc.act_vocab


def test_targets_and_censoring_masks():
    enc = Encoder("bpi2020", ["org:role"], ["unit"], ["Amount"], min_count_cat=1)
    full = make_case("a", ["A x by B", "C x by D", "E x by F"], complete=True)
    part = make_case("b", ["A x by B", "C x by D", "E x by F"], complete=False)
    enc.fit([full, part])
    ef, ep = enc.transform(full), enc.transform(part)
    assert ef.next_act[-1] == EOS and ep.next_act[-1] == -1
    assert list(ef.next_act[:-1]) == list(ef.act[1:])
    assert np.isnan(ef.next_dt[-1]) and np.isfinite(ef.next_dt[:-1]).all()
    assert np.isclose(np.expm1(ef.next_dt[0]), 3600.0)
    assert np.isfinite(ef.remaining).all() and np.isnan(ep.remaining).all()
    assert np.isclose(ef.remaining[0], 7200.0) and ef.remaining[-1] == 0.0


def test_markov_contract_and_evaluate(cases):
    train, val, test = cases[:40], cases[40:50], cases[50:]
    enc = Encoder("bpi2020", ["org:role"], ["unit"], ["Amount"], min_count_cat=1).fit(train)
    etr, eva, ete = enc.transform_all(train), enc.transform_all(val), enc.transform_all(test)
    m = MarkovModel(order=2)
    m.fit(etr, eva, enc)
    pr = m.predict_case(ete[0])
    assert pr.next_probs.shape == (len(ete[0]), enc.n_activities)
    assert np.allclose(pr.next_probs.sum(1), 1.0)
    suf = m.rollout(ete[0], 0, 20)[0]
    assert len(suf) <= 20 and EOS not in suf
    res = evaluate(m, ete, eva, enc, {enc.act_id("Payment Handled")}, n_suffix_prefixes=20, max_suffix_len=20, n_samples=2, n_boot=20)
    for k in ("next_activity", "next_dt_hours", "remaining_hours", "suffix", "uncertainty"):
        assert k in res
    assert 0 <= res["next_activity"]["accuracy"]["mean"] <= 1
    assert res["next_activity"]["nll"]["ci95"][0] <= res["next_activity"]["nll"]["mean"] <= res["next_activity"]["nll"]["ci95"][1]


def test_metric_sanity():
    assert dl_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert dl_similarity([], [1, 2]) == 0.0
    assert 0 < dl_similarity([1, 3, 2], [1, 2, 3]) < 1
    p = np.array([[0.9, 0.1]] * 10)
    y = np.array([0] * 9 + [1])
    e, _ = ece(p, y, n_bins=10)
    assert e < 0.01  # 90% confident, 90% correct


@pytest.mark.skipif(not Path("researcher_work_trial_bundle/data/bpi2020/DomesticDeclarations.xes.gz").exists(), reason="bundle not present")
def test_real_log_smoke():
    from bpm.data.cases import load_cases
    from bpm.ingest.registry import get_schema
    cs = load_cases(get_schema("bpi2020_domestic"), max_cases=30)
    assert len(cs) == 30 and all(np.all(np.diff(c.timestamps) >= 0) for c in cs)
    assert all(c.timestamps[0] > 1.4e9 for c in cs)  # epoch seconds, not µs/1e9
