"""Proposed model: multi-head recurrent state with a mixture-density time head.

State update: h_t = GRU(h_{t-1}, enc(e_t)) — O(1) per new event, no replay of history.
h_0 is initialised from case-start attributes (known before the first event).

Event encoder enc(e_t) = MLP(concat(
    E_act[a_t]                          whole-label embedding (UNK if unseen),
    Σ_j E_comp[comp_j(a_t)]             decomposed-component embeddings (shared across schemas),
    E_f[cat_f,t]  for each event attr f  (UNK if unseen / rare / masked),
    W_time · time_feats_t ))

Heads on h_t:
    next activity   softmax over V (includes <EOS>)            → cross-entropy
    Δt to next      K-component Gaussian mixture on log1p(sec) → mixture NLL   (=log-normal mixture in seconds)
    remaining time  Gaussian on log1p(sec)                     → Gaussian NLL  (complete cases only)
Total loss = CE + w_dt·NLL_dt + w_rem·NLL_rem with w_dt = w_rem = 0.5, fixed before any test run.

Attribute dropout: during training each event's categorical attributes are replaced by UNK with
prob p_attr. This makes autoregressive rollouts (where future attributes are unknown → UNK)
in-distribution and also regularises against attribute-vocabulary shift.

Multi-step: autoregressive rollout from h_t, feeding back the chosen activity, its components,
UNK attributes and time features derived from the predicted Δt. Greedy or ancestral sampling;
n parallel samples give a suffix distribution.

Uncertainty: categorical entropy for next activity; the mixture gives predictive quantiles for
Δt (q10/q90 interval); post-hoc temperature scaling on validation (in evaluate.py).
"""

from __future__ import annotations

import math
import time as _time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bpm.data.cases import decompose_activity
from bpm.data.encoding import EOS, PAD, UNK, EncodedCase, Encoder
from bpm.models.base import CasePreds, SequenceModel

LOG2PI = math.log(2 * math.pi)


class _Net(nn.Module):
    def __init__(self, enc: Encoder, d_model=128, d_act=32, d_comp=16, d_cat=8, n_layers=2, dropout=0.1, n_mix=3):
        super().__init__()
        V = enc.n_activities
        self.n_mix = n_mix
        self.act_emb = nn.Embedding(V, d_act, padding_idx=PAD)
        self.comp_emb = nn.Embedding(len(enc.comp_vocab), d_comp, padding_idx=PAD)
        self.cat_embs = nn.ModuleList([nn.Embedding(len(enc.cat_vocabs[f]), d_cat, padding_idx=PAD) for f in enc.event_cat_fields])
        self.time_proj = nn.Linear(Encoder.TIME_DIM, 16)
        d_in = d_act + d_comp + d_cat * len(self.cat_embs) + 16
        self.in_proj = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Dropout(dropout))
        self.case_cat_embs = nn.ModuleList([nn.Embedding(len(enc.case_cat_vocabs[f]), d_cat) for f in enc.case_cat_fields])
        d_case = d_cat * len(self.case_cat_embs) + 2 * len(enc.case_num_fields)
        self.h0_proj = nn.Linear(max(d_case, 1), d_model * n_layers)
        self.d_case = d_case
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.n_layers, self.d_model = n_layers, d_model
        self.head_act = nn.Linear(d_model, V)
        self.head_dt = nn.Linear(d_model, 3 * n_mix)
        self.head_rem = nn.Linear(d_model, 2)

    def embed(self, act, comps, cat, time):
        x = [self.act_emb(act), self.comp_emb(comps).sum(2)]
        for j, e in enumerate(self.cat_embs):
            x.append(e(cat[..., j]))
        x.append(self.time_proj(time))
        return self.in_proj(torch.cat(x, -1))

    def init_state(self, case_cat, case_num):
        parts = [e(case_cat[:, j]) for j, e in enumerate(self.case_cat_embs)] + [case_num]
        z = torch.cat(parts, -1) if self.d_case > 0 else torch.zeros(case_cat.shape[0], 1, device=case_cat.device)
        h = torch.tanh(self.h0_proj(z)).view(-1, self.n_layers, self.d_model).transpose(0, 1).contiguous()
        return h

    def forward(self, batch):
        x = self.embed(batch["act"], batch["comps"], batch["cat"], batch["time"])
        h0 = self.init_state(batch["case_cat"], batch["case_num"])
        out, _ = self.gru(x, h0)
        return self.heads(out)

    def heads(self, h):
        dt = self.head_dt(h)
        logit_pi, mu, log_sig = dt.chunk(3, -1)
        rem = self.head_rem(h)
        return {"act_logits": self.head_act(h), "pi": logit_pi, "mu": mu, "log_sig": log_sig.clamp(-6, 4),
                "rem_mu": rem[..., 0], "rem_log_sig": rem[..., 1].clamp(-6, 4)}


def mdn_nll(pi_logits, mu, log_sig, y):
    """Negative log-likelihood of y under a Gaussian mixture; y: [...], params: [..., K]."""
    log_pi = F.log_softmax(pi_logits, -1)
    y = y.unsqueeze(-1)
    comp = -0.5 * (LOG2PI + 2 * log_sig + ((y - mu) / log_sig.exp()) ** 2)
    return -torch.logsumexp(log_pi + comp, -1)


def mdn_quantiles(pi_logits, mu, log_sig, qs=(0.1, 0.5, 0.9), iters=30):
    """Quantiles of a Gaussian mixture by bisection on the CDF. Inputs [N, K] → [N, len(qs)]."""
    pi = F.softmax(pi_logits, -1)
    sig = log_sig.exp()
    lo = (mu - 6 * sig).min(-1).values
    hi = (mu + 6 * sig).max(-1).values
    out = []
    for q in qs:
        a, b = lo.clone(), hi.clone()
        for _ in range(iters):
            m = (a + b) / 2
            cdf = (pi * 0.5 * (1 + torch.erf((m.unsqueeze(-1) - mu) / (sig * math.sqrt(2))))).sum(-1)
            a = torch.where(cdf < q, m, a)
            b = torch.where(cdf < q, b, m)
        out.append((a + b) / 2)
    return torch.stack(out, -1)


def _collate(cases: list[EncodedCase], device, p_attr_drop: float = 0.0, rng: np.random.Generator | None = None):
    B = len(cases)
    T = max(len(c) for c in cases)
    C, Fc, D = cases[0].comps.shape[1], cases[0].cat.shape[1], cases[0].time.shape[1]
    act = np.zeros((B, T), np.int64); comps = np.zeros((B, T, C), np.int64); cat = np.zeros((B, T, Fc), np.int64)
    time = np.zeros((B, T, D), np.float32); y_act = np.full((B, T), -1, np.int64)
    y_dt = np.full((B, T), np.nan, np.float32); y_rem = np.full((B, T), np.nan, np.float32)
    for i, c in enumerate(cases):
        n = len(c)
        act[i, :n], comps[i, :n], cat[i, :n], time[i, :n] = c.act, c.comps, c.cat, c.time
        y_act[i, :n], y_dt[i, :n], y_rem[i, :n] = c.next_act, c.next_dt, c.remaining
        if p_attr_drop > 0 and rng is not None:
            m = rng.random((n, Fc)) < p_attr_drop
            cat[i, :n][m] = UNK
    case_cat = np.stack([c.case_cat for c in cases]) if cases[0].case_cat.size else np.zeros((B, 0), np.int64)
    case_num = np.stack([c.case_num for c in cases]) if cases[0].case_num.size else np.zeros((B, 0), np.float32)
    t = lambda a: torch.as_tensor(a, device=device)
    return {"act": t(act), "comps": t(comps), "cat": t(cat), "time": t(time), "y_act": t(y_act), "y_dt": t(y_dt),
            "y_rem": t(np.log1p(np.where(np.isfinite(y_rem), y_rem, 0.0)).astype(np.float32)), "rem_mask": t(np.isfinite(y_rem)),
            "case_cat": t(case_cat), "case_num": t(case_num)}


class RecurrentModel(SequenceModel):
    name = "gru_multihead"

    def __init__(self, d_model=128, n_layers=2, n_mix=3, dropout=0.1, p_attr_drop=0.15, w_dt=0.5, w_rem=0.5,
                 lr=2e-3, batch_size=64, epochs=30, patience=5, seed=0, device: str | None = None):
        self.cfg = dict(d_model=d_model, n_layers=n_layers, n_mix=n_mix, dropout=dropout, p_attr_drop=p_attr_drop,
                        w_dt=w_dt, w_rem=w_rem, lr=lr, batch_size=batch_size, epochs=epochs, patience=patience, seed=seed)
        self.device = torch.device(device or "cpu")

    # ------------------------------------------------------------------ training
    def _loss(self, out, batch):
        mask = batch["y_act"] >= 0
        ce = F.cross_entropy(out["act_logits"][mask], batch["y_act"][mask])
        dt_mask = torch.isfinite(batch["y_dt"])
        nll_dt = mdn_nll(out["pi"][dt_mask], out["mu"][dt_mask], out["log_sig"][dt_mask], batch["y_dt"][dt_mask]).mean()
        rm = batch["rem_mask"]
        if rm.any():
            mu, ls = out["rem_mu"][rm], out["rem_log_sig"][rm]
            nll_rem = (0.5 * (LOG2PI + 2 * ls + ((batch["y_rem"][rm] - mu) / ls.exp()) ** 2)).mean()
        else:
            nll_rem = torch.zeros((), device=ce.device)
        total = ce + self.cfg["w_dt"] * nll_dt + self.cfg["w_rem"] * nll_rem
        return total, {"ce": ce.item(), "nll_dt": nll_dt.item(), "nll_rem": nll_rem.item()}

    def fit(self, train, val, encoder: Encoder) -> dict:
        torch.manual_seed(self.cfg["seed"]); rng = np.random.default_rng(self.cfg["seed"])
        self.encoder = encoder
        self.net = _Net(encoder, self.cfg["d_model"], n_layers=self.cfg["n_layers"], dropout=self.cfg["dropout"], n_mix=self.cfg["n_mix"]).to(self.device)
        self._comp_table = torch.as_tensor(_comp_table(encoder), device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        bs = self.cfg["batch_size"]
        best, best_state, bad, log = float("inf"), None, 0, []
        # sort-by-length bucketing reduces padding; shuffle buckets each epoch
        order = np.argsort([len(c) for c in train])
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        t0 = _time.time()
        for ep in range(self.cfg["epochs"]):
            self.net.train()
            rng.shuffle(batches)
            tr_loss, n = 0.0, 0
            for idx in batches:
                batch = _collate([train[i] for i in idx], self.device, self.cfg["p_attr_drop"], rng)
                loss, parts = self._loss(self.net(batch), batch)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item(); n += 1
            v = self._eval_loss(val)
            log.append({"epoch": ep, "train_loss": tr_loss / max(n, 1), **{f"val_{k}": x for k, x in v.items()}, "sec": _time.time() - t0})
            if v["total"] < best - 1e-4:
                best, bad = v["total"], 0
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return {"epochs_run": len(log), "best_val_total": best, "history": log, "params": self.param_count(), "train_seconds": _time.time() - t0}

    @torch.no_grad()
    def _eval_loss(self, cases):
        self.net.eval()
        tot, parts_sum, n = 0.0, {"ce": 0.0, "nll_dt": 0.0, "nll_rem": 0.0}, 0
        for i in range(0, len(cases), 256):
            batch = _collate(cases[i:i + 256], self.device)
            loss, parts = self._loss(self.net(batch), batch)
            tot += loss.item(); n += 1
            for k in parts_sum: parts_sum[k] += parts[k]
        return {"total": tot / max(n, 1), **{k: v / max(n, 1) for k, v in parts_sum.items()}}

    def param_count(self):
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_case(self, enc: EncodedCase) -> CasePreds:
        self.net.eval()
        batch = _collate([enc], self.device)
        out = self.net(batch)
        probs = F.softmax(out["act_logits"][0], -1).cpu().numpy()
        q = mdn_quantiles(out["pi"][0], out["mu"][0], out["log_sig"][0]).cpu().numpy()
        dt_q = np.expm1(q)
        rem = np.expm1(out["rem_mu"][0].cpu().numpy())
        return CasePreds(probs, dt_q[:, 1], rem, next_dt_q=dt_q[:, [0, 2]])

    @torch.no_grad()
    def rollout(self, enc, t, max_len, mode="greedy", n=1, seed=0):
        self.net.eval()
        g = torch.Generator(device="cpu").manual_seed(seed)
        prefix = EncodedCase(enc.case_id, enc.act[:t + 1], enc.comps[:t + 1], enc.cat[:t + 1], enc.time[:t + 1], enc.case_cat, enc.case_num,
                             enc.next_act[:t + 1], enc.next_dt[:t + 1], enc.remaining[:t + 1], enc.timestamps[:t + 1], enc.activities[:t + 1], enc.complete)
        batch = _collate([prefix] * n, self.device)
        x = self.net.embed(batch["act"], batch["comps"], batch["cat"], batch["time"])
        h0 = self.net.init_state(batch["case_cat"], batch["case_num"])
        out, h = self.net.gru(x, h0)
        y = self.net.heads(out[:, -1])
        ts = torch.full((n,), float(enc.timestamps[t]), dtype=torch.float64)
        start = float(enc.timestamps[0])
        pos = t
        suffixes = [[] for _ in range(n)]
        alive = torch.ones(n, dtype=torch.bool)
        e = self.encoder
        for _ in range(max_len):
            logits = y["act_logits"]
            if mode == "greedy":
                a = logits.argmax(-1)
            else:
                a = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).squeeze(1).to(self.device)
            if mode == "greedy":
                dt_log = mdn_quantiles(y["pi"], y["mu"], y["log_sig"], qs=(0.5,))[:, 0]
            else:
                k = torch.multinomial(F.softmax(y["pi"], -1).cpu(), 1, generator=g).squeeze(1)
                mu, sig = y["mu"].cpu().gather(1, k[:, None]).squeeze(1), y["log_sig"].cpu().gather(1, k[:, None]).squeeze(1).exp()
                dt_log = (mu + sig * torch.randn(n, generator=g)).to(self.device)
            dt = torch.expm1(dt_log.clamp(min=0)).double().cpu()
            a_cpu = a.cpu()
            for i in range(n):
                if alive[i]:
                    if int(a_cpu[i]) == EOS:
                        alive[i] = False
                    else:
                        suffixes[i].append(int(a_cpu[i]))
            if not alive.any():
                break
            ts = ts + dt
            pos += 1
            # next input: chosen activity, its components, UNK attributes, synthetic time feats
            comps = self._comp_table[a]
            cat = torch.full((n, len(e.event_cat_fields)), UNK, dtype=torch.long, device=self.device)
            tf = _time_feats(e, dt.numpy(), (ts - start).numpy(), ts.numpy(), pos)
            x1 = self.net.embed(a[:, None], comps[:, None], cat[:, None], torch.as_tensor(tf, device=self.device)[:, None])
            out, h = self.net.gru(x1, h)
            y = self.net.heads(out[:, -1])
        return suffixes

    # single-event online update (demonstrates the O(1) state update used in the design note)
    @torch.no_grad()
    def step(self, state, act_id: int, cat_ids: list[int], time_feats: np.ndarray):
        a = torch.tensor([[act_id]], device=self.device)
        comps = self._comp_table[a]
        cat = torch.tensor([[cat_ids]], device=self.device)
        tf = torch.as_tensor(time_feats, device=self.device).view(1, 1, -1)
        x = self.net.embed(a, comps, cat, tf)
        out, state = self.net.gru(x, state)
        return state, self.net.heads(out[:, -1])


def _comp_table(enc: Encoder) -> np.ndarray:
    tbl = np.full((enc.n_activities, enc.n_comps), UNK, dtype=np.int64)
    for lbl, i in enc.act_vocab.items():
        if lbl.startswith("<"):
            continue
        for j, comp in enumerate(decompose_activity(lbl, enc.family)[: enc.n_comps]):
            tbl[i, j] = enc.comp_vocab.get(comp, UNK)
    return tbl


def _time_feats(enc: Encoder, dt: np.ndarray, elapsed: np.ndarray, ts: np.ndarray, pos: int) -> np.ndarray:
    hour = ((ts + 3600) / 3600) % 24  # approx local (CET) hour; DST error immaterial for sin/cos features
    wd = ((ts + 3600) // 86400 + 3) % 7
    return np.stack([
        enc.dt_scaler(np.log1p(np.maximum(dt, 0))), enc.elapsed_scaler(np.log1p(np.maximum(elapsed, 0))),
        np.sin(2 * math.pi * hour / 24), np.cos(2 * math.pi * hour / 24),
        np.sin(2 * math.pi * wd / 7), np.cos(2 * math.pi * wd / 7),
        np.full(len(dt), enc.pos_scaler(math.log1p(pos))),
    ], 1).astype(np.float32)
