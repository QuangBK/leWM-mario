"""RC-aux for leWM-mario.

Port of the reachability-correction auxiliary objective from
Li et al. 2026 (arXiv 2605.07278), "Predictive but Not Plannable",
reference: https://github.com/Guang000/RC-aux

Two pieces:
  1. ReachabilityHead — budget-conditioned classifier R_phi(z, z', h)
  2. compute_rcaux_losses(emb, pred_emb, head, cfg) — returns
     (mh_loss, reach_loss, stats) for use inside the joint train loop.

The multi-horizon prediction loss is *not* a separate function — the
existing predictor in mario_lewm already emits per-block predictions; the
joint train loop just needs to apply RC-aux's per-horizon weights instead
of a single mean(). See joint_finetune_v3.py for the integration.

Planner-time scoring lives in `reachability_score` (eq 24).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReachabilityHead(nn.Module):
    """Budget-conditioned reachability head R_phi(z, z', h).

    Identical to RC-aux/module.py:ReachabilityHead — kept verbatim so
    weights from the reference checkpoints could in principle be loaded.
    """

    def __init__(self, embed_dim, hidden_dim=512, max_horizon=8, horizon_dim=32):
        super().__init__()
        self.max_horizon = max_horizon
        self.horizon_emb = nn.Embedding(max_horizon + 1, horizon_dim)
        input_dim = 4 * embed_dim + horizon_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src, goal, horizon):
        horizon = horizon.long().clamp(0, self.max_horizon)
        h_feat = self.horizon_emb(horizon)
        feat = torch.cat([src, goal, (src - goal).abs(), src * goal, h_feat], dim=-1)
        return self.net(feat).squeeze(-1)


def _expand_reach_pairs(src, future_seq):
    """Construct positive (z_i, z_j, h) pairs with j-i = delta and h >= delta.

    src:        (B, D) — anchor at position i
    future_seq: (B, H, D) — future latents at positions i+1..i+H
    Returns flattened (src, goal, budget, delta) each of shape (B*K,) or (B*K, D).
    """
    B, H, D = future_seq.shape
    src_chunks, goal_chunks, budget_chunks, delta_chunks = [], [], [], []
    for delta in range(1, H + 1):
        budgets = torch.arange(delta, H + 1, device=src.device, dtype=torch.long)
        K = budgets.numel()
        src_chunks.append(src.unsqueeze(1).expand(-1, K, -1))
        goal_chunks.append(future_seq[:, delta - 1: delta].expand(-1, K, -1))
        budget_chunks.append(budgets.unsqueeze(0).expand(B, -1))
        delta_chunks.append(torch.full((B, K), delta, device=src.device, dtype=torch.long))
    return (
        torch.cat(src_chunks, dim=1).reshape(-1, D),
        torch.cat(goal_chunks, dim=1).reshape(-1, D),
        torch.cat(budget_chunks, dim=1).reshape(-1),
        torch.cat(delta_chunks, dim=1).reshape(-1),
    )


def _expand_temporal_hard_negative_pairs(src, future_seq):
    """Temporal hard negatives: j-i = delta but budget h < delta (target not yet reached)."""
    B, H, D = future_seq.shape
    src_chunks, goal_chunks, budget_chunks, delta_chunks = [], [], [], []
    for delta in range(2, H + 1):
        budgets = torch.arange(1, delta, device=src.device, dtype=torch.long)
        K = budgets.numel()
        src_chunks.append(src.unsqueeze(1).expand(-1, K, -1))
        goal_chunks.append(future_seq[:, delta - 1: delta].expand(-1, K, -1))
        budget_chunks.append(budgets.unsqueeze(0).expand(B, -1))
        delta_chunks.append(torch.full((B, K), delta, device=src.device, dtype=torch.long))
    if not src_chunks:
        empty_f = src.new_empty(0, D)
        empty_l = torch.empty(0, device=src.device, dtype=torch.long)
        return empty_f, empty_f, empty_l, empty_l
    return (
        torch.cat(src_chunks, dim=1).reshape(-1, D),
        torch.cat(goal_chunks, dim=1).reshape(-1, D),
        torch.cat(budget_chunks, dim=1).reshape(-1),
        torch.cat(delta_chunks, dim=1).reshape(-1),
    )


def compute_reachability_loss(head, src_emb, future_true, future_pred=None,
                              pred_weight=0.5, temporal_neg_weight=1.0,
                              stop_grad_pred=True):
    """RC-aux reachability loss.

    src_emb:     (B, D) — anchor latent (last context position)
    future_true: (B, H, D) — encoded future latents (targets, gradient flows through encoder)
    future_pred: (B, H_p, D) or None — predicted future latents from the rollout
                 (stop-grad by default — calibrates head to planner-time queries
                  without giving the predictor a gradient shortcut via the head)

    Returns (loss, stats_dict).
    """
    device = src_emb.device
    zero = src_emb.new_tensor(0.0)
    B = src_emb.size(0)
    if future_true.size(1) < 1:
        return zero, {}

    # Positives over encoded futures
    src_p, goal_p, budget_p, delta_p = _expand_reach_pairs(src_emb, future_true)
    pos_logits = head(src_p, goal_p, budget_p)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits))

    # Temporal hard negatives
    if temporal_neg_weight > 0.0:
        src_h, goal_h, budget_h, _ = _expand_temporal_hard_negative_pairs(src_emb, future_true)
        if budget_h.numel() > 0:
            hard_logits = head(src_h, goal_h, budget_h)
            hard_neg_loss = F.binary_cross_entropy_with_logits(
                hard_logits, torch.zeros_like(hard_logits))
        else:
            hard_logits = pos_logits.new_zeros(0)
            hard_neg_loss = zero
    else:
        hard_logits = pos_logits.new_zeros(0)
        hard_neg_loss = zero

    # Batch (cross-trajectory) negatives — permute goals
    if B > 1:
        perm = torch.randperm(B, device=device)
        if torch.equal(perm, torch.arange(B, device=device)):
            perm = torch.roll(perm, 1)
        # Re-expand the same positive structure with permuted future goals
        goal_neg_seq = future_true[perm]
        _, goal_neg, _, _ = _expand_reach_pairs(src_emb, goal_neg_seq)
        neg_logits = head(src_p, goal_neg, budget_p)
        batch_neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits))
    else:
        neg_logits = pos_logits.new_zeros(0)
        batch_neg_loss = zero

    # Balance positives against the two negative classes
    neg_terms = batch_neg_loss
    neg_weight = 1.0
    if temporal_neg_weight > 0.0:
        neg_terms = neg_terms + temporal_neg_weight * hard_neg_loss
        neg_weight = neg_weight + temporal_neg_weight
    enc_loss = 0.5 * (pos_loss + neg_terms / neg_weight)

    # Predicted-rollout pairs (stop-grad on predicted latents)
    pred_loss = zero
    if future_pred is not None and pred_weight > 0.0 and future_pred.size(1) > 0:
        fp = future_pred.detach() if stop_grad_pred else future_pred
        src_pp, goal_pp, budget_pp, _ = _expand_reach_pairs(src_emb, fp)
        pp_logits = head(src_pp, goal_pp, budget_pp)
        pp_pos_loss = F.binary_cross_entropy_with_logits(
            pp_logits, torch.ones_like(pp_logits))

        # temporal hard neg on pred
        src_ph, goal_ph, budget_ph, _ = _expand_temporal_hard_negative_pairs(src_emb, fp)
        if budget_ph.numel() > 0:
            ph_logits = head(src_ph, goal_ph, budget_ph)
            pp_hard_loss = F.binary_cross_entropy_with_logits(
                ph_logits, torch.zeros_like(ph_logits))
        else:
            pp_hard_loss = zero

        # batch neg on pred
        if B > 1:
            perm = torch.randperm(B, device=device)
            if torch.equal(perm, torch.arange(B, device=device)):
                perm = torch.roll(perm, 1)
            _, goal_pp_neg, _, _ = _expand_reach_pairs(src_emb, fp[perm])
            ppn_logits = head(src_pp, goal_pp_neg, budget_pp)
            pp_batch_loss = F.binary_cross_entropy_with_logits(
                ppn_logits, torch.zeros_like(ppn_logits))
        else:
            pp_batch_loss = zero

        pp_neg_terms = pp_batch_loss
        pp_neg_w = 1.0
        if temporal_neg_weight > 0.0:
            pp_neg_terms = pp_neg_terms + temporal_neg_weight * pp_hard_loss
            pp_neg_w = pp_neg_w + temporal_neg_weight
        pred_loss = 0.5 * (pp_pos_loss + pp_neg_terms / pp_neg_w)

    total = enc_loss + pred_weight * pred_loss

    stats = {
        "reach/pos_prob": torch.sigmoid(pos_logits).mean().detach(),
        "reach/enc_loss": enc_loss.detach(),
    }
    if hard_logits.numel() > 0:
        stats["reach/hard_neg_prob"] = torch.sigmoid(hard_logits).mean().detach()
    if neg_logits.numel() > 0:
        stats["reach/batch_neg_prob"] = torch.sigmoid(neg_logits).mean().detach()
    if pred_weight > 0.0 and future_pred is not None and future_pred.size(1) > 0:
        stats["reach/pred_loss"] = pred_loss.detach()

    return total, stats


def rollout_open_loop(model, init_emb, init_act, future_acts, H):
    """Autoregressive multi-step rollout with gradients.

    init_emb:    (B, history, D) — encoded latents at positions 0..history-1
    init_act:    (B, history, D) — action embeddings at positions 0..history-1
                                   (action at position i transitions emb[i]→emb[i+1])
    future_acts: (B, H-1, D) — action embeddings at positions history..history+H-2.
                               Empty (B, 0, D) is fine when H=1.
    H:           number of rollout steps.

    Returns pred_emb of shape (B, H, D) — predicted latents at positions
    history, history+1, ..., history+H-1.
    """
    history = init_emb.size(1)
    cur_emb = init_emb
    cur_act = init_act
    preds = []
    for h in range(H):
        pred = model.predict(cur_emb, cur_act)[:, -1:]  # (B, 1, D)
        preds.append(pred)
        if h < H - 1:
            cur_emb = torch.cat([cur_emb[:, 1:], pred], dim=1)
            next_act = future_acts[:, h:h + 1]
            cur_act = torch.cat([cur_act[:, 1:], next_act], dim=1)
    return torch.cat(preds, dim=1)


def multi_horizon_pred_loss(pred_emb, tgt_emb, weighting="uniform", weight_power=1.0):
    """Per-horizon prediction loss with horizon weights.

    pred_emb, tgt_emb: (B, H, D). Returns scalar loss and per-step losses.

    weighting:
      'uniform' — w_k = 1/H
      'linear'  — w_k = k / sum(k)
      'power'   — w_k = k^p / sum(k^p)
    """
    step_losses = (pred_emb - tgt_emb).pow(2).mean(dim=(0, 2))  # (H,)
    H = step_losses.size(0)
    dev, dtype = step_losses.device, step_losses.dtype
    if weighting == "uniform":
        w = torch.ones(H, device=dev, dtype=dtype)
    elif weighting == "linear":
        w = torch.arange(1, H + 1, device=dev, dtype=dtype)
    elif weighting == "power":
        w = torch.arange(1, H + 1, device=dev, dtype=dtype).pow(weight_power)
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    w = w / w.sum().clamp_min(torch.finfo(dtype).eps)
    return (step_losses * w).sum(), step_losses


@torch.no_grad()
def reachability_score(head, rollout_emb, goal_emb, total_horizon):
    """Trajectory-level reachability score per the paper's eq (24).

    rollout_emb: (S, H, D) — predicted intermediate latents along a candidate rollout
                            for each of S CEM samples
    goal_emb:    (D,) — encoded goal latent (single target shared across samples)
    total_horizon: int — H, the planning horizon

    Returns (S,) — max over k in [1, H-1] of R_phi(z_{t+k}, z_g, H-k).
    """
    S, H, D = rollout_emb.shape
    if H < 2:
        # No intermediate latents to score
        return rollout_emb.new_zeros(S)
    # Score every (intermediate, goal, H-k) for k=1..H-1
    z_g = goal_emb.unsqueeze(0).unsqueeze(0).expand(S, H - 1, D)
    z_k = rollout_emb[:, : H - 1]  # ẑ_{t+1..t+H-1}
    remaining = torch.arange(H - 1, 0, -1, device=rollout_emb.device, dtype=torch.long)
    remaining = remaining.unsqueeze(0).expand(S, -1)  # (S, H-1)
    logits = head(z_k.reshape(-1, D), z_g.reshape(-1, D), remaining.reshape(-1))
    R = torch.sigmoid(logits).reshape(S, H - 1)
    return R.max(dim=1).values


@torch.no_grad()
def endpoint_reachability_score(head, src_emb, endpoint_emb, total_horizon):
    """Reward-conditioned discriminator variant.

    For each CEM sample, score R_phi(z_t, ẑ_{t+H}, H) — "is this predicted endpoint
    actually a state that's been reached from this kind of source in training?".

    src_emb: (D,)
    endpoint_emb: (S, D)
    total_horizon: int

    Returns (S,) of probabilities in [0, 1].
    """
    S, D = endpoint_emb.shape
    src = src_emb.unsqueeze(0).expand(S, -1)
    h = torch.full((S,), int(total_horizon), device=endpoint_emb.device, dtype=torch.long)
    return torch.sigmoid(head(src, endpoint_emb, h))
