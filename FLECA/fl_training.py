# ------------------------ Federated Learning Simulation ---------------------- #

import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")



def train_client(
    model, client_data, optimizer_class, optimizer_reg, criterion_class, criterion_reg, config, global_model, round_num, client_behavior="correct"
):
    sequences, anomaly_labels, capacity_labels = client_data
    dataset = TensorDataset(
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(anomaly_labels, dtype=torch.float32),
        torch.tensor(capacity_labels, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model.train()
   
    for epoch in range(config["local_epochs"]):
        for sequences_batch, anomaly_batch, capacity_batch in dataloader:
            sequences_batch = sequences_batch.to(device)
            anomaly_batch = anomaly_batch.to(device)
            capacity_batch = capacity_batch.to(device)


            # ------------------ Malicious Behaviors ------------------ #
        

                
            if client_behavior == "backdoor":
                target_class = 1  # Target anomalies (class 1)
                target_mask = (anomaly_batch == target_class)

                torch.manual_seed(42)  # Fixed seed for reproducibility

                backdoor_mask = (torch.rand(sequences_batch.shape[:2], device=sequences_batch.device) < 0.9) & target_mask.unsqueeze(1)

                # Inject the backdoor **only in the last rounds**
                if round_num >= config["num_rounds"] - 3:
                   sequences_batch[backdoor_mask] += 0.5 # apply trigger

               
            # ------------------ Model Training ------------------ #
            optimizer_class.zero_grad()
            class_output, _ = model(sequences_batch)
            loss_class = criterion_class(class_output, anomaly_batch)
            loss_class.backward()
            optimizer_class.step()

            optimizer_reg.zero_grad()
            _, reg_output = model(sequences_batch)
            loss_reg = criterion_reg(reg_output, capacity_batch)
            loss_reg.backward()
            optimizer_reg.step()

            # FedProx regularization
            if config["aggregation"] in ["FedProx"]:
                prox_term = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += torch.sum((param - global_param) ** 2)
                loss_reg += (config["prox_mu"] / 2) * prox_term

    
    local_model = model.state_dict()
    global_model_dict = global_model.state_dict()  
    model_update = {}
 
    # ------------------ Gauss Attack ------------------ #
    if client_behavior == "gausslf":
     deviation = 1.0  # Stronger noise than krum
     for key in local_model:
        noise = deviation * torch.randn_like(local_model[key])
        local_model[key] = global_model_dict[key] + noise

     for key in global_model_dict:
         model_update[key] = local_model[key] - global_model_dict[key]    
   
    elif client_behavior == "trim":
      deviation = 0.3  # Small deviation
      fraction = 0.2   # Fraction to trim (20%)

      for key in local_model:
        # Add noise first
        noise = deviation * torch.randn_like(local_model[key])
        local_model[key] = global_model_dict[key] + noise
        
        # Flatten parameters to apply trimming
        flat_params = local_model[key].view(-1)
        sorted_params, _ = torch.sort(flat_params)
        
        # Compute trimming indices
        num_trim = int(fraction * flat_params.numel())
        min_val, max_val = sorted_params[num_trim], sorted_params[-num_trim]
        
        # Clamp trimmed parameters
        local_model[key] = torch.clamp(local_model[key], min_val, max_val)

      for key in global_model_dict:
        model_update[key] = local_model[key] - global_model_dict[key]  

    # ------------------ Krum Attack ------------------ #
    elif client_behavior == "krum":
        deviation = 0.1
        shift = 2.0
        for key in global_model_dict:
           local_model[key] =  global_model_dict[key] + deviation * torch.randn_like(local_model[key]) + shift 

        for key in global_model_dict:
           model_update[key] = local_model[key] - global_model_dict[key]     

         


     # ------------------ Backdoor Attack via Model Replacement ------------------ #
    elif client_behavior == "backdoor" and round_num >= config["num_rounds"] - 3:

      gamma = config["k"] # scale factor 

      for key in global_model_dict:
        local_model[key] = gamma * (local_model[key] - global_model_dict[key]) + global_model_dict[key]

      for key in local_model:
            model_update[key] = local_model[key] - global_model_dict[key]  

       

    # ------------------ LDP for correct behavior ------------------ #
    else:
        for key in local_model:
            model_update[key] = local_model[key] - global_model_dict[key]  

    
    return model_update

