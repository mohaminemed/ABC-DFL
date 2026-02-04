import numpy as np
import math
import torch
from defenses.utils import fed_average


def ubar_method(
    client_updates: dict,
    client_losses: dict,
    reference_id: str,
    rho: float = 0.45
):
    """
    Functional UBAR implementation (C-DFL-compatible).

    Args:
      client_updates: {cid: state_dict}
      client_losses: {cid: float}
      reference_id: reference client (acts as global model)
      rho: fraction for Stage 1

    Returns:
      aggregated_model, accepted_client_ids
    """

    assert reference_id in client_updates
    ref_weights = client_updates[reference_id]

    # -------- Helper: flatten weights -------- #
    def flatten(weights):
        return torch.cat([
            v.detach().cpu().flatten()
            for v in weights.values()
        ])

    ref_vec = flatten(ref_weights)

    # -------- Stage 1: distance filtering -------- #
    distances = {}
    for cid, weights in client_updates.items():
        cur_vec = flatten(weights)
        distances[cid] = torch.norm(cur_vec - ref_vec).item()

    num_clients = len(distances)
    k = max(1, int(math.floor(rho * num_clients)))

    sorted_ids = sorted(distances, key=distances.get)
    stage1_ids = sorted_ids[:k]

    print(f"[UBAR] Stage 1 selected {len(stage1_ids)}/{num_clients}")

    # -------- Stage 2: loss filtering -------- #
    own_loss = client_losses[reference_id]
    if own_loss is None:
        own_loss = np.median([client_losses[cid] for cid in stage1_ids])
        print("[UBAR] own_loss not provided → using median proxy")

    stage2_ids = [
        cid for cid in stage1_ids
        if client_losses[cid] <= own_loss
    ]

    print(f"[UBAR] Stage 2 selected {len(stage2_ids)}")

    # -------- Fallback -------- #
    if not stage2_ids:
        best_id = min(stage1_ids, key=lambda cid: client_losses[cid])
        stage2_ids = [best_id]
        print("[UBAR] Fallback → selecting best-loss client")

    # -------- Aggregate -------- #
    accepted_updates = {
        cid: client_updates[cid]
        for cid in stage2_ids
    }

    aggregated_model, accepted_ids = fed_average(
        accepted_updates
    )

    return aggregated_model, accepted_ids


def ubar_filtering(
    client_updates: dict,
    client_losses: dict
    ):
    cs_updates = {}
    cs_accepted_ids = {}
    for ref_id, _ in client_updates.items():
       cs_updates[ref_id], cs_accepted_ids[ref_id] = ubar_method(client_updates=client_updates,
                                             client_losses=client_losses,  
                                             reference_id=ref_id)

    return cs_updates, cs_accepted_ids                                         
