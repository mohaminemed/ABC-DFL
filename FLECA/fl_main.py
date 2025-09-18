import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
  roc_auc_score, roc_curve   
)
import matplotlib.pyplot as plt # type: ignore
import copy
import model
import fl_training
import fl_aggregation
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

MultiTaskTransformer = pd.notnull

def initialize_attack_clients(num_groups, k, attack_name):
    attack_client_ids = {
        "gauss": [],
        "magnitude": [],
        "trim": [],
        "krum": [],
        "backdoor": [],
        "badnets": []
    }

    for i in range(num_groups):
        selected_clients = (
            [i * k + (k - j) for j in range(1, 3)]
            if i in [] 
            else [i * k + (k - 1),  i * k + (k - 2), i * k + (k - 3)] #
        )

        if attack_name in attack_client_ids:
            attack_client_ids[attack_name].extend(selected_clients)

    return attack_client_ids[attack_name]  # Return only the selected attack clients


def train_one_client(client_id, client_data, global_model, attack_name, attack_client_ids, config, round_num):
    local_model = copy.deepcopy(global_model)
    optimizer_class = optim.Adam(local_model.classifier.parameters(), lr=config["learning_rate"])
    optimizer_reg = optim.Adam(local_model.regressor.parameters(), lr=config["learning_rate"])
    criterion_class = nn.BCELoss()
    criterion_reg = nn.MSELoss()

    if len(client_data) == 0:
        return None  # skip empty clients

    # Determine attack type or "correct"
    if attack_name in ["gauss", "magnitude", "trim", "krum", "backdoor", "badnets"] and client_id in attack_client_ids:
        mode = attack_name
        print(f"{attack_name}: {client_id}")
    else:
        mode = "correct"
        print(f"correct: {client_id}")

    client_weight = fl_training.train_client(
        local_model, client_data, optimizer_class, optimizer_reg,
        criterion_class, criterion_reg, config, global_model, round_num, mode
    )
    diffs = []
    for key, param in client_weight.items():
            if "weight" in key or "bias" in key:
                diff = param.to(device) 
                diffs.append(diff.flatten())
    euclidean_distance = torch.linalg.norm(torch.cat(diffs)).item()
    print(f"Client: {client_id} | Euclidean distance from global model: {euclidean_distance:.4f}")

    return client_weight
# ------------------------ Fl Main loop ------------------------------- #
def FL_main_Loop(config, clients_data, sequences, sequences_test, clean_sequences_test , anomaly_labels_test, capacity_labels_test, attack_name, backdoored_test_indices):
    # ------------------------------- Metric Storage ------------------------------ #
    # Initialize storage for metrics
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
        "capacity_r2": []
    }
     # Initialize global model
    if config['model_name'] == "MultiTaskTransformer":
        global_model = model.MultiTaskTransformer(input_dim=sequences.shape[2], hidden_dim=64, encoding_dim=8, num_heads=8, num_layers=3, dropout=0.1).to(device)
    elif  config['model_name'] == "MultiTaskGRU":
        global_model = model.MultiTaskGRU(input_dim=sequences.shape[2], hidden_dim=64, num_layers=1, dropout=0.2).to(device)
    elif  config['model_name'] == "MultiTaskbiLSTM":
        global_model = model.MultiTaskBiLSTM(input_dim=sequences.shape[2], hidden_dim=64, num_layers=1, dropout=0.2).to(device) 
         # Load pre-trained weights
        #path = "global_model_MultiTaskbiLSTM.pth"
        #global_model.load_state_dict(torch.load(path, map_location=device))
    elif  config['model_name'] == "MultiTaskCNN":
        global_model = model.MultiTaskCNN(input_dim=sequences.shape[2], hidden_dim=64, kernel_size=3, dropout=0.2).to(device)      
    else:
        global_model = model.MultiTaskLSTM(input_dim=sequences.shape[2], hidden_dim=64, num_layers=1).to(device)


    # ----------------------- Federated Learning Main Loop ------------------------ #
    for round_num in range(config["num_rounds"]):
        print(f"\nRound {round_num + 1}/{config['num_rounds']}")

    
        num_clients = len(clients_data)
        k = config['k']
        num_groups = num_clients // k

        # Get the attack client IDs
        attack_client_ids = []
        if attack_name != "correct":
          attack_client_ids = initialize_attack_clients(num_groups, k, attack_name)
          # Print the selected attack clients
          print(f"Attack clients for {attack_name}: {attack_client_ids}")
    

        # === Parrallel Training ===
        client_weights = [None] * len(clients_data)  # preallocate

        with ThreadPoolExecutor(max_workers=len(clients_data)) as executor:
          futures = {
          executor.submit(train_one_client, cid, cdata, global_model, attack_name,
                        attack_client_ids, config, round_num): cid
          for cid, cdata in enumerate(clients_data)
          }

          for future in as_completed(futures):
            cid = futures[future]  # get the client id for this future
            result = future.result()
            if result is not None:
              client_weights[cid] = result  # store in correct position

        # Aggregate weights
        global_weights = fl_aggregation.aggregate_models(client_weights, global_model, config, round_num)
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        global_model.eval()
        with torch.no_grad():
            sequences, anomaly_labels, capacity_labels = torch.tensor(sequences_test, dtype=torch.float32).to(device), \
                                                         torch.tensor(anomaly_labels_test, dtype=torch.float32).to(device), \
                                                         torch.tensor(capacity_labels_test, dtype=torch.float32).to(device)
            class_output, reg_output = global_model(sequences)

            clean_sequences = torch.tensor(clean_sequences_test, dtype=torch.float32).to(device)
            clean_class_output, _ = global_model(clean_sequences)

            # Convert tensors to NumPy arrays for evaluation
            anomaly_metrics = fl_aggregation.evaluate_anomaly_detection(anomaly_labels.cpu().numpy(), clean_class_output.cpu().numpy(), class_output.cpu().numpy(), backdoored_test_indices)
            capacity_metrics = fl_aggregation.evaluate_capacity_estimation(capacity_labels.cpu().numpy(), reg_output.cpu().numpy())

            #anomaly_metrics_clean = fl_aggregation.evaluate_anomaly_detection(anomaly_labels.cpu().numpy(), clean_class_output.cpu().numpy(), backdoored_test_indices)
            #print("anomaly_metrics_clean", anomaly_metrics_clean)
            # Compute classification loss using BCELoss
            criterion_class = nn.BCELoss()
            class_loss = criterion_class(class_output, anomaly_labels).item()

            # Compute regression loss using MSELoss
            criterion_reg = nn.MSELoss()
            reg_loss = criterion_reg(reg_output, capacity_labels).item()

            global_loss = class_loss + reg_loss  


            # Log round results
            round_results["round"].append(round_num + 1)
            round_results["global_loss"].append(global_loss)
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

            # Print round results
            print("\n=========Global Metrics After Round=========")
            fl_aggregation.print_metrics(anomaly_metrics, "Anomaly Detection (Global Metrics)")
            fl_aggregation.print_metrics(capacity_metrics, "Capacity Estimation (Global Metrics)")

    # Save the final global model
    global_model_path = f"global_model_{config['model_name']}_{config['aggregation']}_{config['num_clients']}.pth"
    torch.save(global_model.state_dict(), global_model_path)
    print(f"Global model saved to '{global_model_path}'")

    # Save results to a DataFrame and export as CSV
    results_df = pd.DataFrame(round_results)
    results_file_path = f"./Results/Res_MTF/MaliciousRate/tests/{config['model_name']}_IID_{config['iid']}_{config['aggregation']}_{config['num_clients']}_Client__{config['num_rounds']}_Rounds.csv"
    #results_file_path = f"./Results/Res_MTF/MaliciousRate/Base/{config['model_name']}_IID_{config['iid']}_{config['aggregation']}_{config['num_clients']}_Client__{config['num_rounds']}_Rounds.csv"
    results_df.to_csv(results_file_path, index=False)
    print(results_file_path)

    return anomaly_labels_test, class_output.cpu().numpy()


def plot_loss_curve(results_df, title):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["round"], results_df["global_loss"], marker="o", label="Global Loss")
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.show()

def plot_roc_curve(y_true, y_pred, title="ROC Curve for Anomaly Detection"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {roc_auc_score(y_true, y_pred):.4f})")
    #plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title(title)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Anomaly"])
    plt.yticks(tick_marks, ["Normal", "Anomaly"])

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_global_loss_and_accuracy(num_clients, global_losses, global_accuracies, title):
    x = range(len(num_clients))
    width = 0.4  # Width of the bars

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Global Loss Plot
    axes[0].bar(x, global_losses, width, color='blue', alpha=0.7, label='Global Loss')
    axes[0].set_title('Global Loss')
    axes[0].set_xlabel('Number of Clients')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(ticks=x)
    axes[0].set_xticklabels(num_clients)
    axes[0].set_ylim(0.1,10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend()

    # Global Accuracy Plot
    axes[1].bar(x, [acc * 100 for acc in global_accuracies], width, color='green', alpha=0.7, label='Global Accuracy (%)')
    axes[1].set_title('Global Accuracy')
    axes[1].set_xlabel('Number of Clients')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xticks(ticks=x)
    axes[1].set_xticklabels(num_clients)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend()

    # Overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(results_df, title):
    # Créer le graphique
    plt.figure(figsize=(10, 6))

    # Tracer la perte globale
    plt.plot(results_df["round"], results_df["global_loss"], marker="o", label="Global Loss")

    # Ajouter le titre, les étiquettes et la légende
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Chemin pour sauvegarder le fichier
    save_path = "./Results/loss_curve.pdf"

    # Sauvegarder le graphique en tant que fichier PDF
    plt.savefig(save_path, format="pdf")

    # Afficher le graphique
    plt.show()

    print(f"Plot saved to: {save_path}")


def plot_accuracy_curve(results_df, title):
    plt.figure(figsize=(10, 6))

    # Plot Global Accuracy
    if "anomaly_accuracy" in results_df.columns:
        plt.plot(results_df["round"], results_df["anomaly_accuracy"], marker="x", label="Global Accuracy", linestyle="--")

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel("Rounds")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()