"""Failure analysis for a finished run: error compounding, internal coherence, long-history
information loss, plus concrete failure cases.

    uv run python -m bpm.analyze_failures results/international_random.json --n-cases 3

Reads the results JSON (breakdowns computed at eval time), reloads the saved GRU
(``results/<name>/gru_seed0.pt``) + encoder + split, and writes
``results/<name>/failure_analysis.md``.

Analyses
  1. Compounding: per-step error rate along greedy suffixes; DL-similarity by prefix length; how
     often the first wrong step is followed by further wrong steps ("derailment").
  2. Coherence: post-terminal continuation rate, EOS precision/recall, predicted vs true suffix
     length, non-positive Δt rate, Δt interval coverage, fraction of rollouts that repeat the
     same activity ≥3× consecutively (loops).
  3. Long-history loss: next-activity accuracy conditional on whether an "informative" event
     (REJECTED / Queued / Wait) occurred earlier in the prefix, bucketed by how many events ago;
     and whether the recurrent state still separates those histories (probe: does the model's
     predicted distribution differ, measured by mean KL vs the counterfactual prefix with that
     event removed).
  4. Concrete cases: the n worst greedy suffixes (lowest DL-sim among long true suffixes), with
     prefix, true suffix, predicted suffix, top-3 next-activity probabilities at the divergence
     point, and the Δt interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpm import metrics as M
from bpm.data.cases import Split, load_cases
from bpm.data.encoding import EOS, Encoder
from bpm.ingest.registry import get_schema
from bpm.models.recurrent import RecurrentModel
from bpm.run import truncate

INFORMATIVE = ("REJECTED", "Queued", "Wait", "Send Reminder", "SAVED")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--n-prefixes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gru_multihead", help="model name whose <name>_seed0.pt to analyse")
    a = ap.parse_args(argv)
    r = json.loads(Path(a.results_json).read_text())
    name = r["config"]["name"]
    d = Path("results") / name
    schema = get_schema(r["log"])
    encoder = Encoder.from_json(d / "encoder.json")
    split = Split.from_json(d / "split.json")
    cases = truncate(load_cases(schema, max_cases=r["config"].get("max_cases")), r["config"].get("max_len"))
    by = {c.case_id: c for c in cases}
    test = encoder.transform_all([by[i] for i in split.test])
    if r["config"].get("assume_complete"):
        for c in cases:
            c.complete = True
    model = RecurrentModel.load(d / f"{a.model}_seed0.pt", encoder)
    id2act = encoder.id2act
    term_ids = {encoder.act_id(t) for t in schema.terminal}
    gru = r["models"][a.model]["runs"][0]["metrics"]
    L = [f"# Failure analysis — {name} ({a.model} seed 0)", ""]

    # ---------------- 1. compounding
    L += ["## 1. Do errors compound in multi-step prediction?", ""]
    se = gru.get("suffix", {}).get("greedy_step_error_rate", {})
    if se:
        L.append("Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):")
        L.append("")
        L.append("| step | " + " | ".join(se.keys()) + " |")
        L.append("|---|" + "---|" * len(se))
        L.append("| error | " + " | ".join(f"{v['err']:.2f}" for v in se.values()) + " |")
        L.append("| n | " + " | ".join(str(v["n"]) for v in se.values()) + " |")
        L.append("")
    rng = np.random.default_rng(a.seed)
    cand = [(i, t) for i, c in enumerate(test) if c.complete for t in range(len(c) - 1)]
    pick = rng.choice(len(cand), size=min(a.n_prefixes, len(cand)), replace=False)
    items = [(test[cand[j][0]], cand[j][1]) for j in pick]
    greedy = model.rollout_many(items, gru["suffix"]["max_len"], mode="greedy")
    first_wrong_then_wrong, first_wrong_total, derail = 0, 0, []
    loops, len_pred, len_true, sims, records = 0, [], [], [], []
    for (c, t), g in zip(items, greedy):
        g = g[0]
        true = c.act[t + 1:].tolist()
        sims.append(M.dl_similarity(g, true)); len_pred.append(len(g)); len_true.append(len(true))
        if any(g[i] == g[i + 1] == g[i + 2] for i in range(len(g) - 2)):
            loops += 1
        fw = next((i for i, (x, y) in enumerate(zip(g, true)) if x != y), None)
        if fw is not None and fw + 1 < min(len(g), len(true)):
            first_wrong_total += 1
            rest_err = np.mean([x != y for x, y in zip(g[fw + 1:], true[fw + 1:])])
            derail.append(rest_err)
            first_wrong_then_wrong += rest_err > 0.5
        records.append((sims[-1], c, t, g, true))
    L.append(f"- Of {first_wrong_total} suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is "
             f"**{np.mean(derail):.2f}** (vs overall per-step error {np.mean([v['err'] for v in se.values()]) if se else float('nan'):.2f}); "
             f"{first_wrong_then_wrong / max(first_wrong_total, 1):.0%} of them are mostly wrong afterwards (derailment).")
    bk = gru["suffix"].get("dl_by_prefix_len", {})
    L.append("- DL-similarity by prefix length: " + ", ".join(f"k={k}: {v['mean']:.3f} (n={v['n']})" for k, v in bk.items()))
    L.append("")

    # ---------------- 2. coherence
    L += ["## 2. Are predictions internally coherent?", ""]
    na = gru["next_activity"]
    L.append(f"- Post-terminal continuation rate (rollout emits another activity after a terminal one): **{gru['suffix']['post_terminal_continuation_rate']:.3f}**")
    L.append(f"- EOS precision / recall (next-event head): {na['eos_precision']:.3f} / {na['eos_recall']:.3f}")
    L.append(f"- Mean predicted vs true suffix length: {np.mean(len_pred):.2f} vs {np.mean(len_true):.2f}; valid-termination rate {gru['suffix']['valid_termination_rate']:.3f}")
    L.append(f"- Rollouts with the same activity ≥3× consecutively (loops): {loops / len(items):.3f}")
    dt = gru["next_dt_hours"]
    L.append(f"- Δt: MAE {dt['mae']['mean']:.1f} h, medAE {dt['medae']:.1f} h, 80% interval coverage {dt.get('q10_q90_coverage', float('nan')):.2f} "
             f"(mean width {dt.get('q10_q90_mean_width_hours', float('nan')):.0f} h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).")
    # remaining-time monotonicity: within a case, predicted remaining time should decrease with t
    mono = []
    for c in test[:400]:
        if c.complete and len(c) > 2:
            pr = model.predict_case(c)
            mono.append(np.mean(np.diff(pr.remaining_s) <= 0))
    L.append(f"- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **{np.mean(mono):.2f}**")
    L.append("")

    # ---------------- 3. long-history information loss
    L += ["## 3. What is lost from long histories?", ""]
    bk2 = gru["next_activity_by_prefix_len"]
    L.append("- Next-activity accuracy by prefix length: " + ", ".join(f"{k}: {v['accuracy']:.3f} (n={v['n']})" for k, v in bk2.items()))
    buckets = {"1-2": [], "3-5": [], "6-10": [], "11+": []}
    kls = {"1-2": [], "3-5": [], "6-10": [], "11+": []}
    base_acc = []
    for c in test:
        pr = model.predict_case(c)
        info_pos = [i for i, act in enumerate(c.activities) if any(s in act for s in INFORMATIVE)]
        for t in range(len(c) - 1):
            y = c.next_act[t]
            if y < 0:
                continue
            correct = float(pr.next_probs[t].argmax() == y)
            earlier = [t - i for i in info_pos if i < t]
            if not earlier:
                base_acc.append(correct)
                continue
            dist = min(earlier)
            b = "1-2" if dist <= 2 else "3-5" if dist <= 5 else "6-10" if dist <= 10 else "11+"
            buckets[b].append(correct)
    L.append(f"- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: {np.mean(base_acc) if base_acc else float('nan'):.3f} (n={len(base_acc)})")
    L.append("- Accuracy by distance (events) since the most recent informative event: " + ", ".join(f"{k}: {np.mean(v):.3f} (n={len(v)})" for k, v in buckets.items() if v))
    # state-separation probe: KL between p(next | prefix) and p(next | prefix with the informative event deleted)
    probe = {"1-2": [], "3-5": [], "6-10": [], "11+": []}
    n_probe = 0
    for c in test:
        info_pos = [i for i, act in enumerate(c.activities) if any(s in act for s in INFORMATIVE)]
        if not info_pos or len(c) < 4:
            continue
        i0 = info_pos[0]
        if i0 == 0 or i0 >= len(c) - 2:
            continue
        pr = model.predict_case(c)
        # counterfactual: drop event i0
        keep = [j for j in range(len(c)) if j != i0]
        cf = _subcase(c, keep)
        prc = model.predict_case(cf)
        for t in range(i0 + 1, len(c) - 1):
            p, q = pr.next_probs[t], prc.next_probs[t - 1]
            kl = float(np.sum(p * (np.log(p + 1e-9) - np.log(q + 1e-9))))
            dist = t - i0
            b = "1-2" if dist <= 2 else "3-5" if dist <= 5 else "6-10" if dist <= 10 else "11+"
            probe[b].append(kl)
        n_probe += 1
        if n_probe >= 300:
            break
    L.append("- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: "
             + ", ".join(f"{k}: {np.mean(v):.3f} (n={len(v)})" for k, v in probe.items() if v)
             + ". A KL that decays toward 0 with distance means the state has forgotten the event.")
    L.append("")

    # ---------------- 4. concrete cases
    L += [f"## 4. Concrete failure cases ({a.n_cases} worst greedy suffixes among true suffixes of length ≥ 3)", ""]
    worst = sorted([rec for rec in records if len(rec[4]) >= 3], key=lambda x: x[0])[: a.n_cases]
    for sim, c, t, g, true in worst:
        pr = model.predict_case(c)
        fw = next((i for i, (x, y) in enumerate(zip(g, true)) if x != y), min(len(g), len(true)))
        L.append(f"### case `{c.case_id}` — prefix length {t + 1}, DL-sim {sim:.2f}")
        L.append("- prefix: " + " → ".join(c.activities[: t + 1]))
        L.append("- true suffix: " + " → ".join(id2act[i] for i in true))
        L.append("- greedy suffix: " + " → ".join(id2act[i] for i in g) + (" → <EOS>" if len(g) < gru["suffix"]["max_len"] else " (truncated)"))
        top = np.argsort(-pr.next_probs[t])[:3]
        L.append(f"- at the prefix end, top-3 next: " + ", ".join(f"`{id2act[i]}` {pr.next_probs[t][i]:.2f}" for i in top) + f"; true next `{id2act[true[0]]}`")
        L.append(f"- first divergence at suffix step {fw}; Δt 80% interval at prefix end: [{pr.next_dt_q[t][0] / 3600:.1f}, {pr.next_dt_q[t][1] / 3600:.1f}] h, true {np.expm1(c.next_dt[t]) / 3600:.1f} h")
        L.append("")
    fn = "failure_analysis.md" if a.model == "gru_multihead" else f"failure_analysis_{a.model}.md"
    (d / fn).write_text("\n".join(L))
    print(f"wrote {d / fn}")


def _subcase(c, keep):
    from bpm.data.encoding import EncodedCase
    k = np.array(keep)
    return EncodedCase(c.case_id, c.act[k], c.comps[k], c.cat[k], c.time[k], c.case_cat, c.case_num, c.next_act[k], c.next_dt[k], c.remaining[k],
                       c.timestamps[k], [c.activities[i] for i in keep], c.complete)


if __name__ == "__main__":
    main()
