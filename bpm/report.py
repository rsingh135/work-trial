"""Aggregate results/*.json into results/SUMMARY.md (+ reliability plots and per-prefix-length
degradation plots as PNGs under results/figures/).

    uv run python -m bpm.report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results")
FIG = RES / "figures"


def _g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _fmt(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _ci(d, nd=3):
    if not isinstance(d, dict):
        return "–"
    lo, hi = d.get("ci95", [None, None])
    return f"{d['mean']:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if lo is not None else f"{d['mean']:.{nd}f}"


def model_rows(name: str, r: dict) -> list[str]:
    rows = []
    for mname, m in r["models"].items():
        runs = m["runs"]
        ss = m.get("seed_summary")
        met = runs[0]["metrics"]
        na, dt, rm, sf, un = met["next_activity"], met["next_dt_hours"], met.get("remaining_hours", {}), met.get("suffix", {}), met["uncertainty"]
        seeds = f" (±{ss['nll']['std']:.3f}, n={ss['nll']['n']})" if ss else ""
        rows.append(
            f"| {name} | {mname} | {_ci(na['nll'])}{seeds} | {_ci(na['accuracy'])} | {na['macro_f1']:.3f} | "
            f"{_ci(dt['mae'], 1)} | {dt['medae']:.1f} | {_ci(rm.get('mae'), 1) if rm.get('mae') else '–'} | "
            f"{_ci(sf.get('dl_similarity'), 3) if sf.get('dl_similarity') else '–'} | {_fmt(sf.get('exact_match'))} | {_fmt(sf.get('valid_termination_rate'))} | "
            f"{un['ece_raw']:.3f}/{un['ece_calibrated']:.3f} | {un['brier_raw']:.3f} | {_fmt(dt.get('q10_q90_coverage'))} | {runs[0]['fit_seconds']:.0f}s |"
        )
    return rows


def shift_rows(results: dict) -> list[str]:
    """Relative degradation random → chronological for the same log/model."""
    rows = []
    for base in [k for k in results if k.endswith("_random")]:
        shifted = base.replace("_random", "_chrono")
        if shifted not in results:
            continue
        for mname in results[base]["models"]:
            if mname not in results[shifted]["models"]:
                continue
            a, b = results[base]["models"][mname]["runs"][0]["metrics"], results[shifted]["models"][mname]["runs"][0]["metrics"]
            def rel(x, y):
                return f"{x:.3f} → {y:.3f} ({(y - x) / abs(x) * 100:+.0f}%)" if x else "–"
            rows.append(f"| {base.replace('_random','')} | {mname} | {rel(a['next_activity']['nll']['mean'], b['next_activity']['nll']['mean'])} | "
                        f"{rel(a['next_activity']['accuracy']['mean'], b['next_activity']['accuracy']['mean'])} | {rel(a['next_dt_hours']['mae']['mean'], b['next_dt_hours']['mae']['mean'])} | "
                        f"{rel(_g(a,'suffix','dl_similarity','mean', default=0.0), _g(b,'suffix','dl_similarity','mean', default=0.0))} | {rel(a['uncertainty']['ece_raw'], b['uncertainty']['ece_raw'])} |")
    return rows


def transfer_rows(results: dict) -> list[str]:
    rows = []
    for name, r in results.items():
        for mname, t in r.get("transfer", {}).items():
            for mode in ("zero_shot", "few_shot_finetune", "few_shot_scratch"):
                m = t[mode]
                rows.append(f"| {name} → {t['target_log']} | {mname} / {mode} | {_ci(m['next_activity']['nll'])} | {_ci(m['next_activity']['accuracy'])} | "
                            f"{_ci(m['next_dt_hours']['mae'], 1)} | {_ci(_g(m,'suffix','dl_similarity'), 3) if _g(m,'suffix','dl_similarity') else '–'} | {t['unseen_label_rate_in_target']:.2f} | {t['n_finetune_cases']} |")
    return rows


def plots(results: dict):
    FIG.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        # reliability
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
        for mname, m in r["models"].items():
            rel = m["runs"][0]["metrics"]["uncertainty"]["reliability_raw"]
            ax.plot(rel["confidence"], rel["accuracy"], "o-", label=f"{mname} (ECE {m['runs'][0]['metrics']['uncertainty']['ece_raw']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("confidence (top-1)"); ax.set_ylabel("accuracy"); ax.set_title(f"reliability — {name}"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIG / f"reliability_{name}.png", dpi=120); plt.close(fig)
        # degradation with prefix length + suffix step error
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for mname, m in r["models"].items():
            bk = m["runs"][0]["metrics"]["next_activity_by_prefix_len"]
            axes[0].plot(list(bk.keys()), [v["accuracy"] for v in bk.values()], "o-", label=mname)
            se = m["runs"][0]["metrics"].get("suffix", {}).get("greedy_step_error_rate", {})
            if se:
                axes[1].plot([int(k) for k in se], [v["err"] for v in se.values()], "o-", label=mname)
        axes[0].set_xlabel("prefix length"); axes[0].set_ylabel("next-activity accuracy"); axes[0].legend(fontsize=7)
        axes[1].set_xlabel("steps into greedy suffix"); axes[1].set_ylabel("per-step error rate"); axes[1].legend(fontsize=7)
        fig.suptitle(name); fig.tight_layout(); fig.savefig(FIG / f"degradation_{name}.png", dpi=120); plt.close(fig)


def main():
    results = {p.stem: json.loads(p.read_text()) for p in sorted(RES.glob("*.json")) if not p.stem.startswith("_")}
    lines = ["# Part 1 results summary", "", "Generated by `bpm/report.py` from `results/*.json`. Time unit: hours. CIs: 95% case-level bootstrap on the test split; "
             "(±) = std over seeds for the primary metric. ECE shown raw/temperature-scaled.", ""]
    lines.append("| run | model | NLL | top-1 acc | macro-F1 | Δt MAE h | Δt medAE h | remaining MAE h | suffix DL-sim | exact | valid-term | ECE raw/cal | Brier | Δt 80% cov | fit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines += model_rows(name, r)
    sr = shift_rows(results)
    if sr:
        lines += ["", "## Distribution shift: random → chronological split (same log, same model)", "",
                  "| log | model | NLL | acc | Δt MAE h | suffix DL | ECE |", "|---|---|---|---|---|---|---|"] + sr
    tr = transfer_rows(results)
    if tr:
        lines += ["", "## Cross-log transfer (source-trained GRU on target log)", "",
                  "| source → target | model / mode | NLL | acc | Δt MAE h | suffix DL | unseen-label rate | fine-tune cases |", "|---|---|---|---|---|---|---|---|"] + tr
    lines += ["", "## Split metadata", ""]
    for name, r in results.items():
        s = r["split"]
        lines.append(f"- **{name}**: {s['type']} split, fingerprint `{s['fingerprint']}`, train/val/test = {s['n_train']}/{s['n_val']}/{s['n_test']}, "
                     f"vocab {r['data']['vocab_size']}, complete-in-test {r['data']['frac_complete_test']:.2f}, test prefixes {r['data']['test_prefixes']}"
                     + (f", train-cases overlapping test window {s['frac_train_cases_overlapping_test_window']:.2f}" if 'frac_train_cases_overlapping_test_window' in s else ""))
    (RES / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    plots(results)
    print(f"wrote results/SUMMARY.md and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
