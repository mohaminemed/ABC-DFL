import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
from defenses.utils import fed_average

def multi_krum_aggregation(client_updates: dict, f: int, m: int = None):
    """
    Multi-Krum aggregation (ID-aware).

    Args:
        client_updates: {client_id: state_dict}
        f: number of Byzantine clients to tolerate
        m: number of selected models (default: n - f - 2)

    Returns:
        aggregated_state_dict, selected_client_ids
    """

    assert len(client_updates) > 0, "No client updates provided."

    client_ids = list(client_updates.keys())
    updates = list(client_updates.values())
    n = len(updates)

    if n < 2 * f + 2:
        raise ValueError(
            f"Multi-Krum requires n ≥ 2f + 2 (got n={n}, f={f})"
        )

    if m is None:
        m = n - f - 2

    # --- Step 1: Pairwise distance matrix ---
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            distances[i, j] = sum(
                torch.norm(
                    updates[i][k] - updates[j][k]
                ).item()
                for k in updates[i]
                if not k.endswith("num_batches_tracked")
            )

    # --- Step 2: Compute Krum scores ---
    scores = []
    for i in range(n):
        nearest_distances = np.sort(distances[i])[: n - f - 1]
        scores.append(np.sum(nearest_distances))

    # --- Step 3: Select m smallest-score models ---
    selected_indices = np.argsort(scores)[:m]
    selected_ids = [client_ids[i] for i in selected_indices]

    print(f"[Multi-Krum] Selected clients: {selected_ids}")

    # --- Step 4: Aggregate selected models ---
    selected_updates = {
        client_ids[i]: updates[i]
        for i in selected_indices
    }

    aggregated_model, _ = fed_average(selected_updates)

    return aggregated_model, selected_ids
