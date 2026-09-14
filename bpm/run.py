"""Experiment runner: one YAML config → one machine-readable results JSON.

    uv run python -m bpm.run configs/demo.yaml

Config keys (see configs/*.yaml):
  name, log, split {type: random|chronological, seed, fracs}, max_cases, max_len,
  models: [{type: markov|gbm|gru, ...kwargs}], seeds: [..] (applied to gru + gbm),
  eval: {n_suffix_prefixes, max_suffix_len, n_samples, n_boot},
  transfer: {target_log, finetune_frac}  (optional; see README)
  out: results/<name>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

# Cap BLAS/OpenMP threads before numpy/torch/sklearn load. torch and sklearn each bring an OpenMP
# runtime; with the default "all cores" setting and any other process on the box, sklearn's
# HistGradientBoosting calls went from seconds to (effectively) hanging on macOS. 8 threads is
# plenty for these model sizes.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
import yaml

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

from bpm.data.cases import Case, load_cases, split_chronological, split_cv, split_random
from bpm.data.encoding import Encoder, encoder_for
from bpm.evaluate import evaluate
from bpm.ingest.registry import get_schema
from bpm.models.gbm import GBMModel
from bpm.models.markov import MarkovModel
from bpm.models.recurrent import RecurrentModel


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_model(spec: dict, seed: int):
    kind = spec["type"]
    kw = {k: v for k, v in spec.items() if k != "type"}
    if kind == "markov":
        return MarkovModel(**kw)
    if kind == "gbm":
        return GBMModel(seed=seed, **kw)
    if kind == "gru":
        return RecurrentModel(seed=seed, **kw)
    raise ValueError(kind)


def make_split(cases: list[Case], spec: dict):
    if spec["type"] == "random":
        return split_random(cases, seed=spec.get("seed", 0), fracs=tuple(spec.get("fracs", (0.7, 0.15, 0.15))))
    if spec["type"] == "chronological":
        return split_chronological(cases, fracs=tuple(spec.get("fracs", (0.7, 0.15, 0.15))))
    if spec["type"] == "cv":
        return split_cv(cases, fold=spec["fold"], n_folds=spec.get("n_folds", 5), seed=spec.get("seed", 0), val_frac=spec.get("val_frac", 0.2))
    raise ValueError(spec["type"])


def truncate(cases: list[Case], max_len: int | None) -> list[Case]:
    """Optional cap on case length (very long BPI2013 cases dominate padding). Truncated cases
    become *incomplete* so no suffix/remaining targets are fabricated from them."""
    if not max_len:
        return cases
    out = []
    for c in cases:
        if len(c) <= max_len:
            out.append(c)
        else:
            out.append(Case(c.case_id, c.activities[:max_len], c.timestamps[:max_len], c.local_hour[:max_len], c.local_weekday[:max_len],
                            {k: v[:max_len] for k, v in c.event_cat.items()}, c.case_cat, c.case_num, False, c.family))
    return out


def run(cfg: dict) -> dict:
    t_start = time.time()
    schema = get_schema(cfg["log"])
    cases = truncate(load_cases(schema, max_cases=cfg.get("max_cases")), cfg.get("max_len"))
    if cfg.get("assume_complete"):
        # published-protocol mode: every trace gets an end-of-case token regardless of its last activity
        for c in cases:
            c.complete = True
    split = make_split(cases, cfg["split"])
    by_id = {c.case_id: c for c in cases}
    tr, va, te = [by_id[i] for i in split.train], [by_id[i] for i in split.val], [by_id[i] for i in split.test]
    out_dir = Path(cfg.get("out", f"results/{cfg['name']}.json")).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    split.to_json(out_dir / "split.json")

    extra_labels = None
    transfer = cfg.get("transfer")
    if transfer:
        # union label vocab so the output layer can score target-log activities never seen in source training
        tgt_schema = get_schema(transfer["target_log"])
        tgt_cases = truncate(load_cases(tgt_schema, max_cases=transfer.get("max_cases")), cfg.get("max_len"))
        extra_labels = sorted({a for c in tgt_cases for a in c.activities})
    encoder = encoder_for(schema, tr, min_count_cat=cfg.get("min_count_cat", 5), extra_labels=extra_labels)
    encoder.to_json(out_dir / "encoder.json")
    enc_tr, enc_va, enc_te = encoder.transform_all(tr), encoder.transform_all(va), encoder.transform_all(te)
    terminal_ids = {encoder.act_id(a) for a in schema.terminal}

    results = {
        "config": cfg, "git_sha": git_sha(), "platform": platform.platform(), "torch": torch.__version__,
        "log": schema.name, "split": {"type": split.name, "fingerprint": split.fingerprint(), "n_train": len(tr), "n_val": len(va), "n_test": len(te), **split.meta},
        "data": {"n_cases": len(cases), "vocab_size": encoder.n_activities, "n_components": len(encoder.comp_vocab),
                 "frac_complete_test": float(np.mean([c.complete for c in te])) if te else 0.0,
                 "test_prefixes": int(sum(max(len(c) - 1, 0) + int(c.complete) for c in te))},
        "models": {},
    }
    ev = cfg.get("eval", {})
    seeds = cfg.get("seeds", [0])
    for spec in cfg["models"]:
        runs = []
        for seed in (seeds if spec["type"] in ("gru", "gbm") else [0]):
            t0 = time.time()
            model = build_model(spec, seed)
            if hasattr(model, "set_encoder"):
                model.set_encoder(encoder)
            train_log = model.fit(enc_tr, enc_va, encoder)
            fit_s = time.time() - t0
            metrics = evaluate(model, enc_te, enc_va, encoder, terminal_ids, n_suffix_prefixes=ev.get("n_suffix_prefixes", 500),
                               max_suffix_len=ev.get("max_suffix_len", 50), n_samples=ev.get("n_samples", 5), seed=seed, n_boot=ev.get("n_boot", 300))
            runs.append({"seed": seed, "fit_seconds": fit_s, "params": model.param_count(), "train_log": train_log, "metrics": metrics})
            if spec["type"] == "gru" and seed == seeds[0]:
                model.save(out_dir / "gru_seed0.pt")  # for bpm.analyze_failures
            print(f"[{cfg['name']}] {model.name} seed={seed} fit={fit_s:.0f}s  nll={metrics['next_activity']['nll']['mean']:.3f} "
                  f"acc={metrics['next_activity']['accuracy']['mean']:.3f} dtMAE={metrics['next_dt_hours']['mae']['mean']:.1f}h "
                  f"remMAE={metrics['remaining_hours'].get('mae', {}).get('mean', float('nan')):.1f}h "
                  f"DL={metrics['suffix'].get('dl_similarity', {}).get('mean', float('nan')):.3f} ece={metrics['uncertainty']['ece_raw']:.3f}", flush=True)
        entry = {"spec": spec, "runs": runs}
        if len(runs) > 1:
            entry["seed_summary"] = _seed_summary(runs)
        results["models"][model.name] = entry

        # optional: transfer evaluation of this trained model on the target log (zero-shot) + few-shot fine-tune
        if transfer and spec["type"] == "gru":
            results.setdefault("transfer", {})[model.name] = _transfer_eval(model, spec, encoder, tgt_schema, tgt_cases, cfg, transfer, ev, seeds[0])
    results["total_seconds"] = time.time() - t_start
    out = Path(cfg.get("out", f"results/{cfg['name']}.json"))
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out}  ({results['total_seconds']:.0f}s)")
    return results


def _seed_summary(runs: list[dict]) -> dict:
    keys = {"nll": ("next_activity", "nll", "mean"), "accuracy": ("next_activity", "accuracy", "mean"),
            "dt_mae_h": ("next_dt_hours", "mae", "mean"), "rem_mae_h": ("remaining_hours", "mae", "mean"),
            "dl": ("suffix", "dl_similarity", "mean"), "ece_raw": ("uncertainty", "ece_raw")}
    out = {}
    for k, path in keys.items():
        vals = []
        for r in runs:
            d = r["metrics"]
            try:
                for p in path:
                    d = d[p]
                vals.append(float(d))
            except (KeyError, TypeError):
                pass
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def _transfer_eval(model, spec, src_encoder: Encoder, tgt_schema, tgt_cases, cfg, transfer, ev, seed) -> dict:
    """Zero-shot: source-trained model scored on the target log through the source encoder (union
    label vocab; unseen attributes → UNK). Few-shot: fine-tune on a small fraction of target
    cases vs. train-from-scratch on the same fraction; both evaluated on the same target test."""
    tsplit = split_random(tgt_cases, seed=seed, fracs=(0.7, 0.15, 0.15))
    by_id = {c.case_id: c for c in tgt_cases}
    ttr, tva, tte = [by_id[i] for i in tsplit.train], [by_id[i] for i in tsplit.val], [by_id[i] for i in tsplit.test]
    frac = transfer.get("finetune_frac", 0.05)
    rng = np.random.default_rng(seed)
    few = [ttr[i] for i in rng.choice(len(ttr), size=max(int(len(ttr) * frac), 20), replace=False)]
    enc_te, enc_va, enc_few = src_encoder.transform_all(tte), src_encoder.transform_all(tva), src_encoder.transform_all(few)
    term = {src_encoder.act_id(a) for a in tgt_schema.terminal}
    out = {"target_log": tgt_schema.name, "n_target_test": len(tte), "finetune_frac": frac, "n_finetune_cases": len(few),
           "unseen_label_rate_in_target": float(np.mean([a not in src_encoder.act_vocab for c in tte for a in c.activities]))}
    kw = dict(n_suffix_prefixes=ev.get("n_suffix_prefixes", 500) // 2, max_suffix_len=ev.get("max_suffix_len", 50), n_samples=0, seed=seed, n_boot=ev.get("n_boot", 300))
    out["zero_shot"] = evaluate(model, enc_te, enc_va, src_encoder, term, **kw)
    # few-shot fine-tune (continue training the same network on few target cases)
    ft = RecurrentModel(seed=seed, **{k: v for k, v in spec.items() if k != "type"})
    ft.cfg["epochs"] = transfer.get("finetune_epochs", 15)
    ft.encoder, ft.net, ft._comp_table = model.encoder, model.net, model._comp_table
    import copy
    ft.net = copy.deepcopy(model.net)
    ft._fine = True
    _finetune(ft, enc_few, enc_va)
    out["few_shot_finetune"] = evaluate(ft, enc_te, enc_va, src_encoder, term, **kw)
    scratch = RecurrentModel(seed=seed, **{k: v for k, v in spec.items() if k != "type"})
    scratch.fit(enc_few, enc_va, src_encoder)
    out["few_shot_scratch"] = evaluate(scratch, enc_te, enc_va, src_encoder, term, **kw)
    return out


def _finetune(model: RecurrentModel, train, val):
    """Minimal continuation of training on new cases (same optimiser recipe, fewer epochs)."""
    import torch.nn as nn
    from bpm.models.recurrent import _collate
    torch.manual_seed(model.cfg["seed"]); rng = np.random.default_rng(model.cfg["seed"])
    opt = torch.optim.AdamW(model.net.parameters(), lr=model.cfg["lr"] / 2, weight_decay=1e-4)
    bs = model.cfg["batch_size"]
    best, best_state = float("inf"), None
    for ep in range(model.cfg["epochs"]):
        model.net.train()
        idx = rng.permutation(len(train))
        for i in range(0, len(idx), bs):
            batch = _collate([train[j] for j in idx[i:i + bs]], model.device, model.cfg["p_attr_drop"], rng)
            loss, _ = model._loss(model.net(batch), batch)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step()
        v = model._eval_loss(val)["total"]
        if v < best:
            best, best_state = v, {k: x.detach().clone() for k, x in model.net.state_dict().items()}
    if best_state:
        model.net.load_state_dict(best_state)
    model.net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--override", nargs="*", default=[], help="key=value (dot paths), e.g. max_cases=300")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for kv in args.override:
        k, v = kv.split("=", 1)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    run(cfg)


if __name__ == "__main__":
    main()
