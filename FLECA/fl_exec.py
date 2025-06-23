import os
import pandas as pd
import numpy as np
import torch
import fl_main
import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True


def inject_backdoor_trigger(sequences, anomaly_labels, target_class=1, backdoor_prob=0.1, trigger_value=0.5):
    if isinstance(sequences, np.ndarray):
        sequences = torch.tensor(sequences, dtype=torch.float32)
    if isinstance(anomaly_labels, np.ndarray):
        anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.float32)

    target_mask = (anomaly_labels == target_class)
    torch.manual_seed(42)  # Fixed seed for reproducibility
    backdoor_mask = (torch.rand(sequences.shape[:2], device=sequences.device) < backdoor_prob) & target_mask.unsqueeze(1)
    sequences[backdoor_mask] += trigger_value
    
    # Get indices of backdoored samples
    backdoored_indices = torch.nonzero(backdoor_mask.any(dim=1), as_tuple=True)[0].tolist()

    return sequences, backdoor_mask, backdoored_indices




def fl_exec(config, clients_data, sequences, sequences_test, clean_sequences_test, anomaly_labels_test, capacity_labels_test, backdoored_test_indices):
    
    y_true, y_pred = fl_main.FL_main_Loop(config, clients_data, sequences, sequences_test, clean_sequences_test, anomaly_labels_test, capacity_labels_test,  config['attack'], backdoored_test_indices)

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    file_path = f"./Results/MaliciousRate/Y_Pred_Y_True_IID_{config['model_name']}_{config['iid']}_{config['aggregation']}_{config['num_clients']}_Client__{config['num_rounds']}_Rounds.csv"
    print(f"File {file_path} has been saved and added to file_paths.txt.")
    df.to_csv(file_path, index=False)

    file_paths_file = './Results/file_paths.txt'
    if os.path.exists(file_paths_file):
        with open(file_paths_file, 'r') as f:
            existing_paths = set(f.read().splitlines())
    else:
        existing_paths = set()

    if file_path not in existing_paths:
        with open(file_paths_file, 'a') as f:
            f.write(file_path + '\n')


def main():

    print("Starting the main FL function...")
    aggregation_methods = ["FLECAv2", "Multi-Krum", "Trimmed-Mean"] # [] "FLECAv1", ,  "Trimmed-Mean", "Krum", "FedAvg" , "FedProx"
    model_names = [ "MultiTaskbiLSTM", ]#"MultiTaskbiLSTM" ] #, "MultiTaskLSTM","MultiTaskCNN"] 
    iid_options = [False, True] 

    base_config = {
        "num_clients": 18,
        "num_rounds": 1,
        "k": 7,
        "local_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "prox_mu": 0.2,
        "dropout": 0.3,
        "early_stopping": True,
        "patience": 10,
        "balance": True
    }

    path = './battery_dataset3_prepared.npz'
    

    for iid_setting in iid_options:

       print(f"Preparing data for IID={iid_setting}...")

       clients_data, sequences, sequences_test, anomaly_labels_test, capacity_labels_test = preprocessing.data_preparation(
            base_config["num_clients"], path, iid=iid_setting, alpha=0.8)
       clean_sequences_test = sequences_test
       for attack in [ "correct" ]: # "backdoor", "krum", "trim"

        base_config['attack'] = attack
        
        if base_config['attack'] == "backdoor":
          sequences_test, _, backdoored_test_indices = inject_backdoor_trigger(sequences_test, anomaly_labels_test)
        else:
          backdoored_test_indices = []
           
        
        for model_name in model_names:
            for method in aggregation_methods:
                print(f"Running {model_name} with aggregation '{method}' and IID={iid_setting}")
                config = base_config.copy()
                config.update({
                    "aggregation": method,
                    "model_name": model_name,
                    "iid": iid_setting
                })
                fl_exec(config, clients_data, sequences, sequences_test, clean_sequences_test, anomaly_labels_test, capacity_labels_test, backdoored_test_indices)
             

    print("Main FL function completed!")


if __name__ == "__main__":
    main()
