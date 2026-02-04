# ------------------------ Federated Learning Simulation ---------------------- #

import math
import copy
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


# --------------------------------------------------------------------------- #
#                                Client Training                              #
# --------------------------------------------------------------------------- #

def train_client(
    model,
    client_data,
    optimizer_class,
    optimizer_reg,
    criterion_class,
    criterion_reg,
    config,
    global_model,
    round_num,
    client_behavior="correct",
    prev_global_grad=None,
):
    device = config["device"]
    sequences, anomaly_labels, capacity_labels = client_data

    dataset = TensorDataset(
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(anomaly_labels, dtype=torch.float32),
        torch.tensor(capacity_labels, dtype=torch.float32),
    )

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    model.train()

    # ------------------------------------------------------------------ #
    #                         Neurotoxin Gradient Mask                    #
    # ------------------------------------------------------------------ #

    grad_mask = None

    if (
        client_behavior == "neurotoxin"
        and config["attack_start"] <= round_num <= config["attack_end"]
        and prev_global_grad is not None
    ):
        eps = 1e-12
        importance_parts = []
        key_to_delta = {}

        model_keys = {n for n, _ in model.named_parameters()}

        for name, delta in prev_global_grad.items():
            if name not in model_keys:
                continue

            d = delta.detach().cpu().float()
            p = model.state_dict()[name].detach().cpu().float()
            imp = (d.abs() / (p.abs() + eps)).flatten()

            importance_parts.append(imp)
            key_to_delta[name] = d

        if importance_parts:
            all_imp = torch.cat(importance_parts)
            k = max(1, int(config["mask_k_percent"] * all_imp.numel()))
            threshold = torch.topk(all_imp, k).values[-1]

            grad_mask = {}
            for name, d in key_to_delta.items():
                p = model.state_dict()[name].detach().cpu().float()
                imp = d.abs() / (p.abs() + eps)
                grad_mask[name] = imp < threshold   # True = keep


    # ------------------------------------------------------------------ #
    #                              Local Training                        #
    # ------------------------------------------------------------------ #

    for _ in range(config["local_epochs"]):
        for x, y_cls, y_reg in dataloader:

            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)

            B, T, F = x.shape

            # ---------------------------------------------------------- #
            #                       Malicious Data Poisoning Behaviors                  #
            # ---------------------------------------------------------- #

            if client_behavior in ["badnets", "scaling", "neurotoxin"] and \
               config["attack_start"] <= round_num <= config["attack_end"]:

                torch.manual_seed(config["seed"])

                source_class = 1
                target_class = 0
                trigger_rate = config["trigger_rate"]

                target_mask = y_cls == source_class
                poison_mask = (torch.rand(B, device=device) < trigger_rate) & target_mask

                trigger_features = [0, 2, 4, 5, 7]
                t_start, t_end = 5, 35
                trigger_value = 5.0

                pattern = torch.sign(
                    torch.sin(torch.linspace(0, 4 * math.pi, t_end - t_start))
                ).to(device)

                for f in trigger_features:
                    x[poison_mask, t_start:t_end, f] += pattern * trigger_value

                y_cls[poison_mask] = target_class


            elif client_behavior == "feature":
                torch.manual_seed(config["seed"])
                poison_mask = torch.rand(B, device=device) < 1.0

                x[poison_mask] = torch.normal(
                    0.0, 1000.0, size=(poison_mask.sum(), T, F), device=device
                )

                y_reg[poison_mask] = torch.normal(
                    0.0, 1000.0, size=(poison_mask.sum(),), device=device
                )


            elif client_behavior == "l-flip":
                torch.manual_seed(config["seed"])
                y_cls[:] = 1 - y_cls


            elif client_behavior == "adaptive":
                torch.manual_seed(config["seed"])
                poison_mask = torch.rand(B, device=device) < 0.5

                if poison_mask.any():
                    y_reg[poison_mask] = torch.normal(
                        0.0, 1000.0, size=(poison_mask.sum(),), device=device
                    )

                flip_mask = torch.rand(B, device=device) < 0.5
                y_cls[flip_mask] = 1 - y_cls[flip_mask]


            # ---------------------------------------------------------- #
            #                          Classification                   #
            # ---------------------------------------------------------- #

            optimizer_class.zero_grad()
            out_cls, _ = model(x, device)
            loss_cls = criterion_class(out_cls, y_cls)
            loss_cls.backward()

            if grad_mask is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is not None and name in grad_mask:
                            p.grad.mul_(grad_mask[name].to(device))

            optimizer_class.step()

            # ---------------------------------------------------------- #
            #                           Regression                       #
            # ---------------------------------------------------------- #

            optimizer_reg.zero_grad()
            _, out_reg = model(x, device)
            loss_reg = criterion_reg(out_reg, y_reg)
            loss_reg.backward()
            optimizer_reg.step()


    # ------------------------------------------------------------------ #
    #                          Model Update                              #
    # ------------------------------------------------------------------ #

    local = model.state_dict()
    global_w = global_model.state_dict()
    update = {}

    def clip_to_global_norm():
        g = torch.cat([global_w[k].flatten() for k in local])
        l = torch.cat([local[k].flatten() for k in local])

        g_norm = torch.linalg.norm(g)
        l_norm = torch.linalg.norm(l)

        if l_norm > g_norm:
            scale = g_norm / (l_norm + 1e-12)
            for k in local:
                local[k].mul_(scale)

    # ---------------- Model Poisoning: Gauss / Trim / Krum ---------------- #

    if client_behavior == "gauss":
        for k in local:
            local[k] = global_w[k] + torch.randn_like(local[k])

    elif client_behavior == "trim":
      deviation = 0.3  # Small deviation
      fraction = 0.2   # Fraction to trim (20%)
      for key in local:
        # Add noise first
        noise = deviation * torch.randn_like(local[key])
        local[key] = global_w[key] + noise

        # Flatten parameters to apply trimming
        flat_params = local[key].view(-1)
        sorted_params, _ = torch.sort(flat_params)

        # Compute trimming indices
        num_trim = int(fraction * flat_params.numel())
        min_val, max_val = sorted_params[num_trim], sorted_params[-num_trim]

        # Clamp trimmed parameters
        local[key] = torch.clamp(local[key], min_val, max_val)

      for key in global_w:
        update[key] = local[key] - global_w[key]
        

    elif client_behavior == "krum":
        for k in local:
            local[k] = global_w[k] + 0.1 * torch.randn_like(local[k]) + 2.0

    # ---------------- Model Replacement ---------------- #

    elif client_behavior in ["scaling", "neurotoxin", "badnets", "adaptive"] and \
         config["attack_start"] <= round_num <= config["attack_end"]:

        clip_to_global_norm()

        scale = config.get("scale_factor", 1.0)
        for k in local:
            update[k] = scale * (local[k] - global_w[k])

        return update

    # ---------------- Honest Client (+ optional DP) ---------------- #

    for k in local:
        update[k] = local[k] - global_w[k]

    if config["aggregation"] not in ["FLECA", "FLECAv1", "FLECAv2"]:
        return update

    # ---------------- Differential Privacy ---------------- #

    C = config["dp_clip"]
    #sigma = config["dp_std"] 
    epsilon = 5000.0
    delta = 1e-5
    sigma = (C * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

    norm = torch.sqrt(sum(torch.sum(v ** 2) for v in update.values()))
    factor = min(1.0, C / (norm + 1e-12))

    for k in update:
        update[k] *= factor
        update[k] += torch.normal(0.0, sigma, size=update[k].shape, device=device)

    return update
