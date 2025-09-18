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

def inject_backdoor_trigger(
    sequences, 
    anomaly_labels, 
    source_class=1, 
    target_label=0, 
    backdoor_prob=0.1, 
    trigger_value=0.1
):
    """
    Inject a BadNets-style backdoor:
    - Adds a trigger to samples from `source_class`
    - Flips their labels to `target_label`
    
    Args:
        sequences (Tensor/ndarray): Input batch [batch, seq_len, features]
        anomaly_labels (Tensor/ndarray): Labels [batch]
        source_class (int): The class to poison (to be misclassified)
        target_label (int): The label assigned to poisoned samples
        backdoor_prob (float): Probability of poisoning each source sample
        trigger_value (float): Value added as the trigger
    """
    if isinstance(sequences, np.ndarray):
        sequences = torch.tensor(sequences, dtype=torch.float32)
    if isinstance(anomaly_labels, np.ndarray):
        anomaly_labels = torch.tensor(anomaly_labels, dtype=torch.long)

    # Mask for samples from the source_class
    source_mask = (anomaly_labels == source_class)

    # Decide which of those will be poisoned
    torch.manual_seed(42)  # Fixed seed for reproducibility
    poison_mask = (torch.rand(sequences.shape[0], device=sequences.device) < backdoor_prob) & source_mask

    # Inject trigger into poisoned samples
    sequences[poison_mask, :, 0] += trigger_value  # e.g. modify feature 0

    # Flip their labels to the target_label
    anomaly_labels[poison_mask] += target_label

    # Get indices of poisoned samples
    backdoored_indices = torch.nonzero(poison_mask, as_tuple=True)[0].tolist()

    return sequences, anomaly_labels, poison_mask, backdoored_indices




def fl_exec(config, clients_data, sequences, sequences_test, clean_sequences_test, anomaly_labels_test, capacity_labels_test, backdoored_test_indices):
    
    y_true, y_pred = fl_main.FL_main_Loop(config, clients_data, sequences, sequences_test, clean_sequences_test, anomaly_labels_test, capacity_labels_test,  config['attack'], backdoored_test_indices)

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    file_path = f"./Results/Res_MTF/MaliciousRate/tests/Y_Pred_Y_True_IID_{config['model_name']}_{config['iid']}_{config['aggregation']}_{config['num_clients']}_Client__{config['num_rounds']}_Rounds.csv"
    #file_path = f"./Results/Res_MTF/MaliciousRate/Base/Y_Pred_Y_True_IID_{config['model_name']}_{config['iid']}_{config['aggregation']}_{config['num_clients']}_Client__{config['num_rounds']}_Rounds.csv"
    print(f"File {file_path} has been saved and added to file_paths.txt.")
    df.to_csv(file_path, index=False)

    file_paths_file = './Results/Res_MTF/file_paths.txt'
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
    aggregation_methods = [ "FLECAv1"] # "FedAvg" , "FedProx", , "Krum", "Trimmed-Mean"
    model_names = [ "MultiTaskbiLSTM", ]#"MultiTaskbiLSTM" ] #, "MultiTaskLSTM","MultiTaskCNN"] 
    iid_options = [False] # [False, True

    base_config = {
        "num_clients": 7,
        "num_rounds": 10,
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

    path = './Datasets/battery_dataset3_prepared.npz'
    

    for iid_setting in iid_options:

       print(f"Preparing data for IID={iid_setting}...")

       clients_data, sequences, sequences_test, anomaly_labels_test, capacity_labels_test = preprocessing.data_preparation(
            base_config["num_clients"], path, iid=iid_setting, alpha=0.8)
       clean_sequences_test = sequences_test
       for attack in ["badnets"]: # "backdoor", "trim", "badnets" #, "gauss"

        base_config['attack'] = attack
        
        if base_config['attack'] in ["badnets", "backdoor"] :
          sequences_test, _, _, backdoored_test_indices = inject_backdoor_trigger(sequences_test, anomaly_labels_test)
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
