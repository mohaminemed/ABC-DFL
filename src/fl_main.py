import random
import math
import copy
import time
import warnings
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models import MultiTaskBiLSTM, MultiTaskCNN, MultiTaskGRU, MultiTaskLSTM
from fl_training import train_client
from utils import evaluate_anomaly_detection, evaluate_capacity_estimation, print_metrics
from fl_aggregation import aggregate_updates

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


# ------------------------------------------------------------------
# Trigger utilities
# ------------------------------------------------------------------

def sinusoidal_pattern(length, cycles=4):
    x = np.linspace(0, np.pi * cycles, length)
    return np.sign(np.sin(x)).astype(np.float32)


def inject_trigger_eval(
    sequences,
    class_labels,
    trigger_value=5.0,
    source_class=1,
    t_start=5,
    t_end=35,
    trigger_features=(0, 2, 4, 5, 7),
):
    """Inject fixed backdoor trigger for evaluation."""

    poisoned = sequences.copy()
    src_mask = class_labels == source_class
    pattern = sinusoidal_pattern(t_end - t_start)

    for f in trigger_features:
        poisoned[src_mask, t_start:t_end, f] += pattern * trigger_value

    return poisoned, np.where(src_mask)[0]


# ------------------------------------------------------------------
# Malicious client selection
# ------------------------------------------------------------------

def initialize_attack_clients(num_groups, k, attack_name, config):

    supported = {
        "gauss", "trim", "krum", "l-flip", "feature",
        "badnets", "scaling", "adaptive", "neurotoxin"
    }

    if attack_name not in supported:
        return []

    num_malicious_groups = min(config.get("num_malicious_groups", 0), num_groups)
    position = config.get("malicious_groups_position", "last")

    if position == "last":
        malicious_groups = list(range(num_groups - num_malicious_groups, num_groups))
    elif position == "first":
        malicious_groups = list(range(num_malicious_groups))
    elif position == "random":
        malicious_groups = random.sample(range(num_groups), num_malicious_groups)
    else:
        raise ValueError(f"Unknown malicious_groups_position: {position}")

    attack_clients = []

    for g in range(num_groups):
        base = g * k

        if g in malicious_groups:
            selected = [base + (k - j) for j in range(1, max(k - 2, 7))]
        else:
            n = max(1, math.ceil(k / 3)-1)
            selected = [base + (k - j) for j in range(1, n + 1)]

        attack_clients.extend(selected)

    return attack_clients


# ------------------------------------------------------------------
# Federated learning loop
# ------------------------------------------------------------------

def FL_main_Loop(
    config,
    clients_data,
    sequences,
    sequences_test,
    anomaly_labels_test,
    capacity_labels_test,
    attack_name,
):

    device = config["device"]

    round_results = {
        "round": [],
        "global_loss": [],
        "anomaly_accuracy": [],
        "anomaly_recall": [],
        "anomaly_precision": [],
        "anomaly_f1_score": [],
        "anomaly_auroc": [],
        "anomaly_asr": [],
        "capacity_mae": [],
        "capacity_mse": [],
        "capacity_rmse": [],
        "capacity_r2": [],
    }

    input_dim = sequences.shape[2]

    if config["model_name"] == "MultiTaskGRU":
        global_model = MultiTaskGRU(input_dim, 64, 1, 0.2).to(device)
    elif config["model_name"] == "MultiTaskbiLSTM":
        global_model = MultiTaskBiLSTM(input_dim, 64, 1, 0.2).to(device)
    elif config["model_name"] == "MultiTaskCNN":
        global_model = MultiTaskCNN(input_dim, 64, 3, 0.2).to(device)
    else:
        global_model = MultiTaskLSTM(input_dim, 64, 1).to(device)

    prev_global_state = None
    prev_global_grad = None

    num_clients = len(clients_data)
    k = config["k"]
    num_groups = num_clients // k

    if attack_name != "correct":
        attack_client_ids = initialize_attack_clients(num_groups, k, attack_name, config)
        print(f"Attack clients ({attack_name}): {attack_client_ids}")
    else:
        attack_client_ids = []

    for rnd in range(config["num_rounds"]):

        print(f"\nRound {rnd + 1}/{config['num_rounds']}")

        prev_global_state = {
            k: v.detach().cpu().clone()
            for k, v in global_model.state_dict().items()
        }

        clients_weights = {}
        clients_losses = {}

        for cid, client_data in enumerate(clients_data):

            local_model = copy.deepcopy(global_model)

            opt_cls = optim.Adam(local_model.classifier.parameters(), lr=config["learning_rate"])
            opt_reg = optim.Adam(local_model.regressor.parameters(), lr=config["learning_rate"])

            crit_cls = nn.BCELoss()
            crit_reg = nn.MSELoss()

            if not client_data:
                continue

            mode = attack_name if cid in attack_client_ids else "correct"

            kwargs = {}
            if mode in {"neurotoxin", "adaptive"}:
                kwargs["prev_global_grad"] = prev_global_grad

            updates = train_client(
                local_model,
                client_data,
                opt_cls,
                opt_reg,
                crit_cls,
                crit_reg,
                config,
                global_model,
                rnd,
                mode,
                **kwargs,
            )

            clients_weights[cid] = updates

            if config["aggregation"] == "UBAR" or config["flow"] == "DFL":
                merged = {
                    k: updates[k] + global_model.state_dict()[k]
                    for k in global_model.state_dict()
                }

                local_model.load_state_dict(merged)
                local_model.eval()

                with torch.no_grad():
                    x = torch.tensor(sequences_test, dtype=torch.float32).to(device)
                    y_cls = torch.tensor(anomaly_labels_test, dtype=torch.float32).to(device)
                    y_reg = torch.tensor(capacity_labels_test, dtype=torch.float32).to(device)

                    c_out, r_out = local_model(x, device)

                    loss = crit_cls(c_out, y_cls) + crit_reg(r_out, y_reg)
                    clients_losses[cid] = loss.item()

        global_weights = aggregate_updates(
            clients_weights,
            clients_losses,
            global_model,
            config,
            rnd,
        )

        global_model.load_state_dict(global_weights)

        prev_global_grad = {
            k: v.detach().cpu() - prev_global_state[k]
            for k, v in global_model.state_dict().items()
        }

        # ---------------- Evaluation ----------------

        global_model.eval()
        with torch.no_grad():

            x_clean = torch.tensor(sequences_test, dtype=torch.float32).to(device)
            y_cls = torch.tensor(anomaly_labels_test, dtype=torch.float32).to(device)
            y_reg = torch.tensor(capacity_labels_test, dtype=torch.float32).to(device)

            clean_cls, clean_reg = global_model(x_clean, device)

            backdoor_idx = []

            if attack_name in {"badnets", "scaling", "neurotoxin"} and rnd >= config["attack_start"]:
                poisoned, backdoor_idx = inject_trigger_eval(
                    sequences_test.copy(),
                    np.array(anomaly_labels_test),
                )
                x_poison = torch.tensor(poisoned, dtype=torch.float32).to(device)
            else:
                x_poison = x_clean

            cls_out, reg_out = global_model(x_poison, device)

            anomaly_metrics = evaluate_anomaly_detection(
                anomaly_labels_test,
                clean_cls.cpu().numpy(),
                cls_out.cpu().numpy(),
                backdoor_idx,
            )

            capacity_metrics = evaluate_capacity_estimation(
                capacity_labels_test,
                clean_reg.cpu().numpy(),
            )

            loss = nn.BCELoss()(cls_out, y_cls) + nn.MSELoss()(reg_out, y_reg)

            round_results["round"].append(rnd + 1)
            round_results["global_loss"].append(loss.item())
            round_results["anomaly_accuracy"].append(anomaly_metrics["Accuracy"])
            round_results["anomaly_recall"].append(anomaly_metrics["Recall"])
            round_results["anomaly_precision"].append(anomaly_metrics["Precision"])
            round_results["anomaly_f1_score"].append(anomaly_metrics["F1-Score"])
            round_results["anomaly_auroc"].append(anomaly_metrics["AUROC"])
            round_results["anomaly_asr"].append(anomaly_metrics["ASR"])
            round_results["capacity_mae"].append(capacity_metrics["MAE"])
            round_results["capacity_mse"].append(capacity_metrics["MSE"])
            round_results["capacity_rmse"].append(capacity_metrics["RMSE"])
            round_results["capacity_r2"].append(capacity_metrics["R2"])

            print_metrics(anomaly_metrics, "Anomaly Detection (Global)")
            print_metrics(capacity_metrics, "Capacity Estimation (Global)")

    model_path = f"global_model_{config['model_name']}_{config['aggregation']}_{config['num_clients']}.pth"
    torch.save(global_model.state_dict(), model_path)

    df = pd.DataFrame(round_results)
    csv_path = (
        f"./results/{config['attack']}/"
        f"{config['model_name']}_IID_{config['iid']}_"
        f"{config['aggregation']}_{config['num_clients']}_"
        f"{config['num_rounds']}_Rounds.csv"
    )

    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_csv(csv_path, index=False)

    return anomaly_labels_test, cls_out.cpu().numpy()
