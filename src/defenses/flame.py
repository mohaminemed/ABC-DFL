import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import hdbscan   


def flame_aggregation(
    client_updates: dict,
    lamda=0.0001,
    eta=1.0,
    device="cpu",
    min_cluster_size=2
):
    """
    FLAME aggregation (ID-aware).

    Args:
        client_updates: {client_id: state_dict}
        lamda: noise scale
        eta: server learning rate
    Returns:
        aggregated_state_dict, selected_client_ids
    """

    if len(client_updates) == 0:
        raise ValueError("No client updates for FLAME.")

    if len(client_updates) == 1:
        cid = next(iter(client_updates))
        return client_updates[cid], [cid]

    client_ids = list(client_updates.keys())
    updates = list(client_updates.values())

    # --- Step 1: Extract task-layer embeddings ---
    task_keys = [
        k for k in updates[0]
        if "classifier.2.weight" in k or "regressor.2.weight" in k
    ]

    def flatten(weights):
        return np.concatenate([
            weights[k].detach().cpu().numpy().flatten()
            for k in task_keys
        ])

    X = np.stack([flatten(w) for w in updates]).astype(np.float64)

    # --- Step 2: Pairwise distances + clustering ---
    dist_matrix = euclidean_distances(X)

    clustering = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True
    ).fit(dist_matrix)

    labels = clustering.labels_
    clean_labels = {cid: int(l) for cid, l in zip(client_ids, labels)}
    print(f"[FLAME] Cluster labels: {clean_labels}")

    valid_labels = labels[labels != -1]

    if len(valid_labels) == 0:
        print("[FLAME] All clients labeled as noise → keep all")
        selected_indices = list(range(len(client_ids)))
    else:
        unique, counts = np.unique(valid_labels, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        selected_indices = [
            i for i, l in enumerate(labels) if l == largest_label
        ]

    selected_ids = [client_ids[i] for i in selected_indices]
    selected_updates = [updates[i] for i in selected_indices]

    print(f"[FLAME] Selected clients: {selected_ids}")

    # --- Step 3: Compute clipping norm (median L2 norm) ---
    norms = []
    for w in selected_updates:
        flat = torch.cat([
            p.flatten().to(device)
            for p in w.values()
            if not p.ndim == 0
        ])
        norms.append(torch.linalg.norm(flat).item())

    clip_norm = float(torch.median(torch.tensor(norms)))
    clip_norm = max(clip_norm, 1e-6)

    # --- Step 4: Clip + aggregate ---
    agg = {
        k: torch.zeros_like(v, device=device)
        for k, v in updates[0].items()
    }

    for w in selected_updates:
        flat = torch.cat([
            p.flatten().to(device)
            for p in w.values()
            if not p.ndim == 0
        ])
        norm = torch.linalg.norm(flat)

        scale = min(1.0, clip_norm / (norm + 1e-8))

        for k in agg:
            agg[k] += w[k].to(device) * scale / len(selected_updates)

    # --- Step 5: Noise injection ---
    for k in agg:
        if "weight" in k or "bias" in k:
            std = lamda * clip_norm
            noise = torch.normal(
                mean=0.0,
                std=std,
                size=agg[k].shape,
                device=device
            )
            agg[k] += noise

        agg[k] *= eta

    return agg, selected_ids
