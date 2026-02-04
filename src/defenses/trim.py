import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

def trimmed_mean_aggregation(client_updates: dict, beta: float, device="cpu"):
    """
    Trimmed Mean aggregation (ID-aware).

    Args:
        client_updates: {client_id: state_dict}
        beta: trimming ratio (0 ≤ beta < 0.5)
    Returns:
        aggregated_state_dict, contributing_client_ids
    """

    assert 0 <= beta < 0.5, "beta must be in [0, 0.5)"
    assert len(client_updates) > 0, "No client updates provided."

    client_ids = list(client_updates.keys())
    updates = list(client_updates.values())
    n = len(updates)

    trim_k = int(beta * n)

    aggregated = {}

    for key in updates[0].keys():

        # Skip BN bookkeeping
        if key.endswith("num_batches_tracked"):
            aggregated[key] = updates[0][key].clone()
            continue

        # Stack parameter tensors
        stacked = torch.stack(
            [u[key].to(device) for u in updates], dim=0
        )

        # Sort along client dimension
        sorted_vals, _ = torch.sort(stacked, dim=0)

        # Trim extremes
        if trim_k > 0:
            trimmed = sorted_vals[trim_k : n - trim_k]
        else:
            trimmed = sorted_vals

        aggregated[key] = torch.mean(trimmed, dim=0)

    print(
        f"[Trimmed Mean] beta={beta}, trimmed {trim_k} per side, "
        f"contributors={client_ids}"
    )

    return aggregated, client_ids
