import os
import numpy as np
import pandas as pd
import torch

from fl_main import FL_main_Loop
from preprocessing import data_preparation

# =========================
# System / Torch Settings
# =========================

torch.backends.cuda.matmul.allow_tf32 = True

# =========================
# Experiment Parameters
# =========================

MODELS = [
    "MultiTaskbiLSTM",
    # "MultiTaskLSTM",
    # "MultiTaskCNN",
]

AGGREGATIONS = [
    "FLECAv2",
    # "Multi-Krum",
    # "Flame",
    # "UBAR",
    # "Norm-Clip",
    # "Weak-DP",
    # "Trimmed-Mean",
    # "FedProx",
    # "FedAvg",
]

ATTACKS = [
    #"gauss",
    #"neurotoxin",
    "correct",
    "badnets",
    "l-flip",
    "scaling",
    "krum",
    "trim",
    "feature",
    "adaptive",
]

IID_OPTIONS = [
    False,
    # True,
]

DATASET_PATH = "./datasets/battery_dataset3_prepared.npz"

BASE_CONFIG = {
    "flow": "C-DFL",
    "num_clients": 7,
    "num_rounds": 1,
    "k": 7,
    "dirichlet_alpha": 0.8,

    "local_epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "prox_mu": 0.2,
    "dropout": 0.3,

    "early_stopping": True,
    "patience": 10,
    "balance": True,

    # Attack
    "attack_start": 0,
    "attack_end": 5,
    "num_malicious_groups": 0,
    "malicious_groups_position": "last",
    "trigger_rate": 0.3,
    "mask_k_percent": 0.05,
    "scale_factor": 3.0,
    "churn_rate": 0.0,

    # Defense / robustness
    "alpha": 0.5,
    "beta": 0.2,
    "f": 2,
    "dp_std": 0.01, # or default sigma = (C * math.sqrt(2 * math.log(1.25 / delta))) / epsilon
    "dp_clip": 4.0,

    # Reproducibility / system
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =========================
# FL Execution
# =========================

def fl_exec(config,
            clients_data,
            sequences,
            sequences_test,
            anomaly_labels_test,
            capacity_labels_test):

    y_true, y_pred = FL_main_Loop(
        config,
        clients_data,
        sequences,
        sequences_test,
        anomaly_labels_test,
        capacity_labels_test,
        config["attack"]
    )

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    results_dir = f"./results/{config['attack']}"
    os.makedirs(results_dir, exist_ok=True)

    file_path = (
        f"{results_dir}/Y_Pred_Y_True_IID_{config['model_name']}_"
        f"{config['iid']}_{config['aggregation']}_"
        f"{config['num_clients']}_Client_"
        f"{config['num_rounds']}_Rounds.csv"
    )

    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

    # Track all generated files
    os.makedirs("./results", exist_ok=True)
    file_paths_file = "./results/file_paths.txt"

    if os.path.exists(file_paths_file):
        with open(file_paths_file, "r") as f:
            existing_paths = set(f.read().splitlines())
    else:
        existing_paths = set()

    if file_path not in existing_paths:
        with open(file_paths_file, "a") as f:
            f.write(file_path + "\n")


# =========================
# Main
# =========================

def main():

    print("Starting FL experiments...")

    for iid_setting in IID_OPTIONS:

        print(f"\nPreparing data (IID={iid_setting})...")

        clients_data, sequences, sequences_test, anomaly_labels_test, capacity_labels_test = (
            data_preparation(
                BASE_CONFIG["num_clients"],
                DATASET_PATH,
                iid=iid_setting,
                alpha=BASE_CONFIG["dirichlet_alpha"],
                seed=BASE_CONFIG["seed"]
            )
        )

        for attack in ATTACKS:
            for model_name in MODELS:
                for aggregation in AGGREGATIONS:

                    print(
                        f"\nRunning {model_name} | "
                        f"Aggregation={aggregation} | "
                        f"Attack={attack} | "
                        f"IID={iid_setting} | "
                        f"Device={BASE_CONFIG['device']}"
                    )

                    config = BASE_CONFIG.copy()
                    config.update({
                        "attack": attack,
                        "model_name": model_name,
                        "aggregation": aggregation,
                        "iid": iid_setting
                    })

                    fl_exec(
                        config,
                        clients_data,
                        sequences,
                        sequences_test,
                        anomaly_labels_test,
                        capacity_labels_test
                    )

    print("\nAll FL experiments completed!")


if __name__ == "__main__":
    main()
