"""Aggregate the 5-fold published-protocol runs and put them next to the transcribed published
BPI2013 numbers (researcher_work_trial_bundle/results/PUBLISHED_BPI2013_RESULTS.md).

    uv run python -m bpm.published_compare   → results/PUBLISHED_COMPARISON.md

Units follow the paper: accuracy in %, suffix DL as normalised similarity (0-1, higher better),
remaining-time MAE in **days**.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path("results")
PUBLISHED = {
    "closed": {"acc": {"Tax": 64.01, "Hinkka": 63.47, "Theis (no attrs)": 59.48, "Evermann": 58.83, "Camargo": 54.67, "Theis (attrs)": 54.65},
               "dl": {"Camargo argmax": 0.6641, "Evermann": 0.6416, "Tax": 0.5824, "Camargo random": 0.5357, "Francescomarino": 0.5276},
               "rem_days": {"Navarin": 159.164, "Tax": 172.849, "Francescomarino": 191.100, "Camargo argmax": 257.086}},
    "incidents": {"acc": {"Hinkka": 74.69, "Tax": 70.09, "Evermann": 66.78, "Camargo": 66.68, "Theis (no attrs)": 59.41, "Theis (attrs)": 51.50},
                  "dl": {"Camargo random": 0.5294, "Evermann": 0.4730, "Francescomarino": 0.3607, "Tax": 0.3336, "Camargo argmax": 0.2607},
                  "rem_days": {"Navarin": 12.366, "Camargo argmax": 28.132, "Tax": 30.082, "Francescomarino": 35.405}},
}


def main():
    lines = ["# BPI 2013 — reproduction of the published protocol vs. published results", "",
             "Protocol (Rama-Maneiro et al., arXiv:2009.13251 v4, as transcribed in the bundle): 5-fold CV over cases, 80/20 train/val within the "
             "training folds, end-of-case token appended to **every** trace, trace-level attributes excluded, metrics averaged over the five test folds. "
             "Every prefix (length 1 … T, the last one predicting the end token) is scored. Remaining-time MAE in days, suffix = normalised "
             "Damerau–Levenshtein similarity (greedy decoding; `sampled` = mean over 5 ancestral samples where available). "
             "Remaining difference to the paper: my models use *event*-level attributes (role, group, impact, product, resource) — the closest published "
             "row is therefore *Theis, with attributes*; the Markov baseline uses activity labels only.", ""]
    for tag in ("closed", "incidents"):
        files = sorted(RES.glob(f"published_{tag}_fold*.json"))
        if not files:
            continue
        runs = [json.loads(f.read_text()) for f in files]
        lines.append(f"## {'Closed Problems' if tag == 'closed' else 'Incidents'} ({len(runs)} folds)")
        lines.append("")
        lines.append("| model | next-activity acc % (mean ± sd over folds) | suffix DL sim | remaining MAE days | NLL |")
        lines.append("|---|---|---|---|---|")
        for m in runs[0]["models"]:
            acc = [r["models"][m]["runs"][0]["metrics"]["next_activity"]["accuracy"]["mean"] * 100 for r in runs]
            dl = [r["models"][m]["runs"][0]["metrics"]["suffix"]["dl_similarity"]["mean"] for r in runs]
            rem = [r["models"][m]["runs"][0]["metrics"]["remaining_hours"]["mae"]["mean"] / 24 for r in runs]
            nll = [r["models"][m]["runs"][0]["metrics"]["next_activity"]["nll"]["mean"] for r in runs]
            lines.append(f"| **{m} (this work)** | {np.mean(acc):.2f} ± {np.std(acc):.2f} | {np.mean(dl):.4f} ± {np.std(dl):.4f} | {np.mean(rem):.2f} ± {np.std(rem):.2f} | {np.mean(nll):.3f} |")
        pub = PUBLISHED[tag]
        for name, v in pub["acc"].items():
            lines.append(f"| {name} (published) | {v:.2f} | {pub['dl'].get(name, pub['dl'].get(name + ' argmax', '–')) if name in pub['dl'] or name + ' argmax' in pub['dl'] else '–'} | {pub['rem_days'].get(name, pub['rem_days'].get(name + ' argmax', '–'))} | – |")
        for name, v in pub["dl"].items():
            if name not in pub["acc"] and name.replace(" argmax", "") not in pub["acc"] and name.replace(" random", "") not in pub["acc"]:
                lines.append(f"| {name} (published) | – | {v:.4f} | {pub['rem_days'].get(name, '–')} | – |")
        lines.append("")
        lines.append(f"- folds: " + ", ".join(f"fold{r['split']['fold']} test n={r['split']['n_test']}" for r in runs) + f"; git {runs[0]['git_sha']}")
        lines.append("")
    lines.append("Published numbers are transcriptions from the bundle; see `researcher_work_trial_bundle/results/PUBLISHED_BPI2013_RESULTS.md` for caveats "
                 "(Table 7 caption ambiguity, supplementary next-timestamp results not transcribed).")
    (RES / "PUBLISHED_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("wrote results/PUBLISHED_COMPARISON.md")


if __name__ == "__main__":
    main()
