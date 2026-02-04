import torch
import numpy as np
import random
from defenses.utils import fed_average
from defenses.clip import norm_clipping_aggregation


def scclip_method(
    client_updates: dict,
    reference_id: str,
    client_lens: dict = None,
    clipping_norm: float = None,
    adaptive: bool = True,
    adaptive_percentile: float = 90.0,
    tau_scale: float = 1.0,
    bucketing: bool = False,
    bucket_size: int = 3,
    eta: float = 1.0,
    use_momentum: bool = False,
    momentum_alpha: float = 0.9,
    momentum_buffer: dict = None,
    device: str = "cpu",
):
    """
    Functional SCCLIP (CLIPPED-GOSSIP-based defense).

    Args:
      client_updates: {cid: state_dict}
      reference_id: reference model ID
      client_lens: {cid: sample_count} or None
      clipping_norm: fixed tau (None → adaptive)
      adaptive_percentile: percentile for tau
      bucketing: whether to bucket deltas
      eta: server LR
      use_momentum: global momentum
      momentum_buffer: persistent momentum dict

    Returns:
      aggregated_model, updated_momentum
    """

    ref = client_updates[reference_id]
    device = torch.device(device)

    # -------- Helper -------- #
    def flatten(d):
        return torch.cat([v.detach().to(device).flatten() for v in d.values()])

    # -------- Step 1: compute deltas -------- #
    deltas = {}
    for cid, weights in client_updates.items():
        deltas[cid] = {
            k: weights[k].to(device) - ref[k].to(device)
            for k in ref
        }

    # -------- Optional bucketing -------- #
    delta_items = list(deltas.items())
    if bucketing and len(delta_items) > 1:
        random.shuffle(delta_items)
        buckets = [
            delta_items[i:i + bucket_size]
            for i in range(0, len(delta_items), bucket_size)
        ]
        bucketed = []
        for bucket in buckets:
            avg = {}
            for k in ref:
                avg[k] = torch.mean(
                    torch.stack([d[k] for _, d in bucket]), dim=0
                )
            bucketed.append(avg)
        deltas_list = bucketed
        weights = [1.0 / len(bucketed)] * len(bucketed)
    else:
        deltas_list = list(deltas.values())
        if client_lens:
            total = sum(client_lens.values())
            weights = [client_lens[cid] / total for cid in deltas]
        else:
            weights = [1.0 / len(deltas_list)] * len(deltas_list)

    # -------- Step 2: determine tau -------- #
    norms = torch.tensor(
        [torch.norm(flatten(d)).item() for d in deltas_list],
        device=device
    )

    if clipping_norm is not None:
        tau = clipping_norm
    else:
        q = torch.quantile(norms, adaptive_percentile / 100.0)
        tau = max(1e-12, q.item() * tau_scale)

    # -------- Step 3: clip deltas -------- #
    clipped = []
    for d in deltas_list:
        norm = torch.norm(flatten(d))
        scale = min(1.0, tau / (norm + 1e-12))
        clipped.append({k: v * scale for k, v in d.items()})

    # -------- Step 4: aggregate clipped deltas -------- #
    agg_delta = {k: torch.zeros_like(v) for k, v in ref.items()}
    for w, d in zip(weights, clipped):
        for k in agg_delta:
            agg_delta[k] += d[k] * float(w)

    # -------- Step 5: optional momentum -------- #
    if use_momentum:
        if momentum_buffer is None:
            momentum_buffer = {k: torch.zeros_like(v) for k, v in agg_delta.items()}
        for k in agg_delta:
            momentum_buffer[k] = (
                momentum_alpha * momentum_buffer[k]
                + (1 - momentum_alpha) * agg_delta[k]
            )
        agg_delta = momentum_buffer

    # -------- Step 6: apply update -------- #
    new_model = {
        k: ref[k].to(device) + eta * agg_delta[k]
        for k in ref
    }

    

    return new_model, momentum_buffer

def scclip_aggregation(client_updates: dict, device: str = "cpu"):

    cs_updates = {}
    for ref_id, _ in client_updates.items():
       cs_updates[ref_id],_ = scclip_method(client_updates=client_updates, reference_id=ref_id, device = device)
    
    aggregate_update, _ = norm_clipping_aggregation(client_updates=cs_updates, device=device)
    return aggregate_update
   
    

    