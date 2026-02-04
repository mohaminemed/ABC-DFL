import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
from defenses.utils import fed_average

def norm_clipping_aggregation(
    client_updates: dict,
    clip_norm: float = None,
    device="cpu"
):
    """
    Norm-Clipping + FedAvg (ID-aware).

    Args:
        client_updates: {client_id: state_dict (delta or full update)}
        clip_norm: fixed clipping norm (None → adaptive median norm)
    Returns:
        aggregated_update, contributing_client_ids
    """

    assert len(client_updates) > 0

    client_ids = list(client_updates.keys())
    updates = list(client_updates.values())

    # --- Step 1: Compute update norms ---
    norms = []
    for w in updates:
        flat = torch.cat([
            p.detach().flatten()
            for k, p in w.items()
            if not k.endswith("num_batches_tracked")
        ])
        norms.append(torch.linalg.norm(flat).item())

    # --- Step 2: Adaptive clipping norm ---
    if clip_norm is None:
        clip_norm = float(torch.median(torch.tensor(norms)))
        print(f"[Norm-Clip] Adaptive clip norm = {clip_norm:.4f}")

    # --- Step 3: Clip updates ---
    clipped_updates = {}
    for cid, w, n in zip(client_ids, updates, norms):
        scale = min(1.0, clip_norm / (n + 1e-8))
        clipped_updates[cid] = {
            k: (v * scale).to(device)
            for k, v in w.items()
        }

    # --- Step 4: FedAvg ---
    aggregated_update, _ = fed_average(clipped_updates)

    return aggregated_update, client_ids


def weak_dp_aggregation(
    client_updates: dict,
    clip_norm: float = None,
    noise_std: float = 0.001,
    device="cpu"
):
    """
    Weak-DP FedAvg:
    Norm clipping + Gaussian noise.

    Args:
        client_updates: {client_id: state_dict}
        clip_norm: clipping norm (None → adaptive)
        noise_std: Gaussian noise std
    Returns:
        aggregated_update, contributing_client_ids
    """

    # --- Step 1: Norm clipping ---
    aggregated_update, client_ids = norm_clipping_aggregation(
        client_updates,
        clip_norm=clip_norm,
        device=device
    )

    # --- Step 2: Add Gaussian noise ---
    noisy_update = {}
    for k, v in aggregated_update.items():
        if "weight" in k or "bias" in k:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=v.shape,
                device=v.device
            )
            noisy_update[k] = v + noise
        else:
            noisy_update[k] = v.clone()

    print(f"[Weak-DP] Noise std = {noise_std}")

    return noisy_update, client_ids
