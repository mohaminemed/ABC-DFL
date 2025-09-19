# ------------------------ Federated Learning Simulation ---------------------- #

import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")




# ------------------------ Federated Learning Simulation ---------------------- #
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
                           
            if client_behavior == "badnets":
              source_class=1  # The class to be misclassified
              target_class = 0  # The attacker's chosen target class
              trigger_rate = 0.5  # Fraction of data to poison

              torch.manual_seed(42)  # Fixed seed for reproducibility

              # Mask for samples from the source_class
              target_mask = (anomaly_batch == source_class)

              # Random mask: select a fraction of samples to poison
              poison_mask = ((torch.rand(sequences_batch.shape[0], device=sequences_batch.device) < trigger_rate ) & target_mask)

              # Inject the trigger into selected samples
              # (example: add a small constant to part of the sequence)
              sequences_batch[poison_mask, :, 0] += 0.5   # modify only one feature/channel

              # Flip labels of poisoned samples to the target class
              anomaly_batch[poison_mask] = target_class
             

              # Optionally: only apply in the last few rounds (stealthier attack)
              #if round_num < config["num_rounds"] - 3:
               # poison_mask[:] = False  # disable early poisoning

               
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
    if client_behavior == "gauss":
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

     
     # ------------------ Model Replacement ------------------ #
    elif client_behavior in ["backdoor"] and round_num >= config["num_rounds"] - 8:

      gamma = config["k"] # scale factor 
      deviation = 0.1
      shift = 0.0
      for key in global_model_dict:
           local_model[key] =  global_model_dict[key] + deviation * torch.randn_like(local_model[key]) + shift 

      for key in global_model_dict:
        model_update[key] = gamma * (local_model[key] - global_model_dict[key])

    elif client_behavior in ["badnets"] and round_num >= config["num_rounds"] - 8:

      gamma = config["k"]*10 # scale factor 

      for key in global_model_dict:
        model_update[key] = gamma * (local_model[key] - global_model_dict[key])    

    # ------------------ DP for correct behavior ------------------ #
    else:
       for key in local_model:
            model_update[key] = local_model[key] - global_model_dict[key]  
       # Privacy parameters
       epsilon = 20.0   # control the desired privacy level
       delta = 1e-5    # Small probability of exceeding epsilon-privacy
       sensitivity = 1.0   # Sensitivity of model updates (can be adjusted based on clipping strategy)

       # Compute noise scale
       laplace_scale = sensitivity / epsilon  # Laplace mechanism
       gaussian_scale = (sensitivity * math.sqrt(2 * math.log(1.25 / delta))) / (epsilon*10)  # Gaussian mechanism
       
       #print(f"Laplace scale: {laplace_scale:.4f}; Gaussian scale: {gaussian_scale:.4f}")

       for key in model_update:
         noise = torch.normal(mean=0.0, std=0.0000, size=model_update[key].shape).to(model_update[key].device)
         if "weight" in key:  # Apply noise only to weights
          if "classifier" in key:  # classification layer
            noise = torch.distributions.Laplace(0, laplace_scale).sample(model_update[key].shape).to(model_update[key].device)
          elif "regressor" in key:  # Regression layers
            noise = torch.normal(mean=0.0, std=gaussian_scale, size=model_update[key].shape).to(model_update[key].device)
           
         model_update[key] += noise
    
    return model_update

