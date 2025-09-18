# ------------------------ Federated Learning Simulation ---------------------- #
import numpy as np
import math
import torch
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
)
import copy
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import cdist
import hdbscan

def aggregate_models(client_updates, global_model, config, round_num):
    """
    Aggregates model weights using FedAvg, FedProx, FLECA, Krum, or Trimmed-Mean.
    """
    global_model_dict = global_model.state_dict() 
    # Common helper functions
    def assign_clients_to_cs(client_updates, k):
        num_clients = len(client_updates)
        num_cs = math.ceil(num_clients / k)
        cs_assignments = [[] for _ in range(num_cs)]
        for i, client in enumerate(client_updates):
            cs_idx = i // k
            cs_assignments[cs_idx].append(client)
        return cs_assignments
    
    def fed_average(list_updates):
      aggregated_model = copy.deepcopy(global_model_dict)
      local_weight_diff_sum = {}
      for key in global_model_dict.keys():
        local_weight_diff_sum[key] = sum([update[key] for update in list_updates]) 
      # Apply the aggregation rule: Gt+1 = Gt + (1 / n) * Σ (Lt+1_i − Gt)  
      for key in global_model_dict.keys():
        aggregated_model[key] = local_weight_diff_sum[key] / len(list_updates)
      return aggregated_model

    def threshold_filtering_and_aggregation_with_own_reference(client_updates, threshold):
      """
      Each EV uses its own model as the reference for filtering and aggregating.
      First step: Cosine-based clustering filtering using the global model.
      Second step: L2 norm filtering using the EV's own model as reference.
      Returns accepted indices per EV for majority voting.
      """
      aggregated_updates = []
      accepted_indices_per_ev = []

     
      client_updates_to_check = [(i, weights) for i, weights in enumerate(client_updates)]
      filtered_dict = {idx: weights for idx, weights in client_updates_to_check}

      for i, reference_weights in enumerate(client_updates):
        accepted_updates = []
        accepted_indices = []

        models_to_check = list(filtered_dict.items())
        
        if i not in filtered_dict:
            models_to_check.append((i, reference_weights))

        for j, (idx, weights) in enumerate(models_to_check):
            similar = True
            for key in ["classifier.2.weight", "regressor.2.weight"]:
                reference_tensor = reference_weights[key].cpu().numpy()
                current_tensor = weights[key].cpu().numpy()
                diff = np.linalg.norm(current_tensor - reference_tensor)
                norm_reference = np.linalg.norm(reference_tensor)
                if diff > threshold * norm_reference:
                    #print(f"EV {i} | Model {idx} | Key: {key} | Norm_Ref: {norm_reference} | Norm_Mod: {np.linalg.norm(current_tensor)} | Diffs: {diff:.4f} | Threshold: {threshold * norm_reference:.4f}")
                    similar = False
                    break

            if similar:
                accepted_updates.append(weights)
                # Add index to classification and prediction tasks
                accepted_indices.append(idx)
                

           
        if  len(accepted_updates)<=1:
           print(f"[Warning] No models passed similarity check for EV {i}.")
          
        print(f"EV/CS: {i} ; accepted_models: {len(accepted_updates)}; accepted_indices: {accepted_indices}.")

        # Aggregate accepted models
        aggregated_updates.append(fed_average(accepted_updates))
        accepted_indices_per_ev.append(accepted_indices)

      print(f"aggregated_models: {len(aggregated_updates)}.")
      return aggregated_updates, accepted_indices_per_ev

    def clustering_aggregation(updates_list, eps=3, min_samples=3):
      if len(updates_list) < min_samples:
        return fed_average(updates_list)

      def extract_task_weights(weights, task_keys):
        return np.concatenate([weights[key].cpu().numpy().flatten() for key in task_keys])

      # Define which layers to extract
      keys = [key for key in updates_list[0].keys() if "classifier.2.weight" in key or "regressor.2.weight" in key]
      weights_vectors = [extract_task_weights(w, keys) for w in updates_list]
      

      # Cluster using DBSCAN
      clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(weights_vectors)
      
      #clustering = hdbscan.HDBSCAN(min_cluster_size=2,   metric='euclidean', allow_single_cluster=True).fit(euclid_dist)
      labels = clustering.labels_

      # Identify the largest cluster (excluding noise)
      unique_labels, counts = np.unique(labels, return_counts=True)
      if len(counts) == 0:
        raise ValueError("No clusters found (only noise points).")

      largest_cluster_id = unique_labels[np.argmax(counts)]

      weights_vectors = np.array(weights_vectors)
      
      # Get vectors for the largest cluster
      largest_cluster_vectors = weights_vectors[labels == largest_cluster_id]

      # Compute the centroid of the largest cluster
      centroid = np.mean(largest_cluster_vectors, axis=0).reshape(1, -1)

      # Compute distances from all points to the largest cluster centroid
      distances_to_largest_centroid = cdist(weights_vectors, centroid, metric='euclidean').flatten()

       
      # Optional: print distances
      print("Distance of all models to the centroid of the largest cluster:")
      for idx, dist in enumerate(distances_to_largest_centroid):
        print(f"EV {idx}: distance = {dist:.4f}")


      # Filter out noise (label = -1)
      filtered_models = [updates_list[i] for i in range(len(labels)) if labels[i] != -1]

      print(f"DBSCAN labels: {labels}")
      print(f"Filtered models (non-outliers): {len(filtered_models)}")
 
      if not filtered_models:
        print("Warning: all models considered outliers, returning global average.")
        return fed_average(updates_list)

      return fed_average(filtered_models)

    def majority_voting(aggregated_updates, accepted_indices_per_ev, eps=1.5, min_samples=2):
       """
       Applies majority voting on model parameters.
       """
       if len(aggregated_updates) == 1:
        return aggregated_updates[0]

       num_updates = len(aggregated_updates)
       index_count = {}
       # Count accepted indices globally
       for indices in accepted_indices_per_ev:
            for idx in indices:
                index_count[idx] = index_count.get(idx, 0) + 1

       # Majority indices
       majority_indices = {idx for idx, count in index_count.items() if count > num_updates / 2}

       print(f"Majority-voted indices: {len(majority_indices)} | {majority_indices}")

       if not majority_indices:
        print("[Info] No valid models. Fall back to clustering.")
        return clustering_aggregation(aggregated_updates, eps, min_samples)
       # Select majority models
       selected_updates = [aggregated_updates[idx] for idx in majority_indices if idx < num_updates]

       return fed_average(selected_updates)

    def krum_aggregation(updates_list, f):
        distances = np.zeros((len(updates_list), len(updates_list)))
        for i, w_i in enumerate(updates_list):
            for j, w_j in enumerate(updates_list):
                if i != j:
                    distances[i, j] = sum(
                        np.linalg.norm(w_i[key].cpu().numpy() - w_j[key].cpu().numpy())
                        for key in w_i.keys()
                    )
        scores = [np.sum(np.sort(distances[i])[:len(updates_list) - f - 2]) for i in range(len(updates_list))]
        return updates_list[np.argmin(scores)]

    def trimmed_mean_aggregation(updates_list, beta):
        aggregated_weights = copy.deepcopy(updates_list[0])
        for key in aggregated_weights.keys():
            key_weights = np.array([w[key].cpu().numpy() for w in updates_list])
            sorted_weights = np.sort(key_weights, axis=0)
            trim_count = int(beta * len(updates_list))
            trimmed_weights = sorted_weights[trim_count:-trim_count] if trim_count > 0 else sorted_weights
            aggregated_weights[key] = torch.tensor(trimmed_weights.mean(axis=0)).to(device)
        return aggregated_weights

    def multi_krum_aggregation(updates_list, f, m=None):

        num_clients = len(updates_list)
        if num_clients >= 2 * f + 2:
            m = m if m is not None else num_clients - f - 2  # Default Multi-Krum selection size
        else:
           return krum_aggregation(updates_list, f)    

        # Compute pairwise distances between weight updates
        distances = np.zeros((num_clients, num_clients))
        for i, w_i in enumerate(updates_list):
         for j, w_j in enumerate(updates_list):
            if i != j:
                distances[i, j] = sum(
                    np.linalg.norm(w_i[key].cpu().numpy() - w_j[key].cpu().numpy())
                    for key in w_i.keys()
                )

        # Compute scores for each client
        scores = [np.sum(np.sort(distances[i])[:num_clients - f - 2]) for i in range(num_clients)]

        # Select top m models with lowest scores
        selected_indices = np.argsort(scores)[:m]
        #print(f"Selected indices for Multi-Krum: {selected_indices}, scores: {scores}")
        selected_weights = [updates_list[i] for i in selected_indices]

        # Compute the average of selected weights
        avg_weights = {key: torch.mean(torch.stack([w[key] for w in selected_weights]), dim=0)
                   for key in selected_weights[0].keys()}

        return avg_weights
  
    def flame_aggregation(updates_list, global_model_dict, lamda=0.001, eta=1.0, device="cpu"):
      """
      FLAME aggregation function.
      Args:
        updates_list: list of client updates (list of dicts with tensors).
        global_model_dict: dict of global model parameters before aggregation.
        lamda: noise coefficient.
        eta: server learning rate/update scale.
        device: device for tensor operations.

      Returns:
        Aggregated update dict (same format as updates_list[0]).
      """

      if not updates_list:
        return global_model_dict

      num_clients = len(updates_list)
      if num_clients == 1:
        return updates_list[0]
      
      # --- Step 1: Compute Euclidean distances + extract last layer weights ---
      all_client_updates = []
      euclidean_dists = []
      #layer_names = [k for k in global_model_dict.keys() if ("weight" in k or "bias" in k)]
      #last_layers = layer_names[-2:]  # take last two parameter layers
      #print(f"Last layers for clustering: {last_layers}")
      for update in updates_list:
        # Distance norm wrt global model
        diffs = []
        for name, param in update.items():
            if "weight" in name or "bias" in name:
                diff = param.to(device) 
                diffs.append(diff.flatten())
        euclidean_dists.append(torch.linalg.norm(torch.cat(diffs)).item())

      def extract_task_weights(weights, task_keys):
          return np.concatenate([weights[key].cpu().numpy().flatten() for key in task_keys])
      
      keys = [key for key in updates_list[0].keys() if "classifier.2.weight" in key or "regressor.2.weight" in key]
      all_client_updates = [extract_task_weights(w, keys) for w in updates_list]
       
      print(f"Euclidean distances from global model: {euclidean_dists}")  
      print(f"Clustering {num_clients} client updates with HDBSCAN...")
    
      X = np.stack(all_client_updates)
      X = X.astype(np.float64)
      
      #  Pairwise distances
      euclid_dist = euclidean_distances(X)   
      cos_dist = cosine_distances(X)    
      print("\nEuclidean distance matrix:\n", euclid_dist)
      print("\nCosine distance matrix:\n", cos_dist)

      # --- Step 2: Cluster updates with HDBSCAN ---
      
      clustering = hdbscan.HDBSCAN(min_cluster_size=2, 
                                   metric='precomputed', 
                                   allow_single_cluster=True
                                   ).fit(euclid_dist)
      labels = clustering.labels_
      # Identify the largest cluster (excluding noise)
      unique_labels, counts = np.unique(labels[labels!=-1], return_counts=True)
      if len(counts) == 0:
        raise ValueError("No clusters found (only noise points).")

      if np.all(labels == -1) or len(np.unique(labels)) == 1:
        benign_indices = list(range(num_clients))
      else:
        if len(unique_labels) > 0:
            largest_cluster = unique_labels[np.argmax(counts)]
            benign_indices = [i for i, l in enumerate(labels) if l == largest_cluster]
        else:
            benign_indices = list(range(num_clients))
          
      print(f"HDBSCAN labels: {labels} | benign_indices: {benign_indices}")

      # --- Step 3: Robust aggregation (clipping + noise) ---
      benign_updates = [updates_list[i] for i in benign_indices]
      if not benign_updates:
        return global_model_dict

      benign_distances = [euclidean_dists[i] for i in benign_indices]
      clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

      for name in global_model_dict:
       global_model_dict[name] = global_model_dict[name].to(device)

      #weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model_dict.items()}
      weight_accumulator = {name: torch.zeros_like(param, device=device) for name, param in global_model_dict.items()}

      for i, idx in enumerate(benign_indices):
        update = updates_list[idx]
        weight = 1.0 / len(benign_updates)

        for name, param in update.items():
            if name.endswith("num_batches_tracked"): 
                continue

            diff = param.to(device) 

            # Apply clipping
            if euclidean_dists[idx] > clip_norm:
                diff *= clip_norm / euclidean_dists[idx]

            weight_accumulator[name].add_(diff * weight)

      # --- Step 4: Aggregate the updates ---
      aggregated_update = {}
      for name, param in global_model_dict.items():
        if name in weight_accumulator:
            new_param = weight_accumulator[name] * eta
            if "weight" in name or "bias" in name:
                std_dev = lamda * clip_norm
                noise = torch.normal(0, std_dev, param.shape, device=param.device)
                new_param += noise
            aggregated_update[name] = new_param
        else:
            aggregated_update[name] = param.clone()

      return aggregated_update

    
    # Two-level aggregation process
    k = config["k"]
    cs_assignments = assign_clients_to_cs(client_updates, k)

    if config["aggregation"] in ["FedAvg", "FedProx"]:
        cs_aggregated_updates = [fed_average(cs_clients_updates) for cs_clients_updates in cs_assignments]
        print(f"cs_aggregated_updates: {len(cs_aggregated_updates)}")
        global_update = fed_average(cs_aggregated_updates) 
        for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key]

    elif config["aggregation"] == "FLECAv1":
      gamma = config.get("gamma", 0.4)
      kappa = config.get("kappa", 1.0)
      t = round_num
      lambda_t = 1.0 * (1 - math.exp(-0.1 * t))
      ev_aggregated_updates_per_cs = []
      
      # Step 1: Intra-CS Aggregation with Threshold Filtering
      for i, cs in enumerate(cs_assignments):
        threshold = gamma * np.exp(-kappa * lambda_t)
        print(f"Threshold (EV Level): {threshold} | CS: {i}")
        ev_aggregated_updates, accepted_indices_per_ev = threshold_filtering_and_aggregation_with_own_reference(cs, threshold)

        if ev_aggregated_updates:
            # Apply majority voting at CS level
            min_samples = k//2  + 1 
            cs_final_update = majority_voting(ev_aggregated_updates, accepted_indices_per_ev, 0.8, min_samples)
            ev_aggregated_updates_per_cs.append(cs_final_update)
            
        else:
            print("No valid aggregated model for this CS after threshold filtering.")

      # Step 2: Inter-CS Aggregation
      cs_aggregated_updates = []
      for cs_update in ev_aggregated_updates_per_cs:
        cs_aggregated_updates.append(cs_update)
      gamma = config.get("gamma", 0.4)
      inter_cs_threshold = gamma * np.exp(-kappa * lambda_t)
      print(f"Threshold (Inter-CS Level): {inter_cs_threshold}")

      if len(cs_aggregated_updates) > 1:
        # Apply threshold filtering at inter-CS level
        filtered_cs_updates, accepted_indices_per_cs = threshold_filtering_and_aggregation_with_own_reference(cs_aggregated_updates, inter_cs_threshold)
        # Final global model aggregation using majority voting
        min_samples = config["num_clients"] // (2*k) 
        global_update = majority_voting(filtered_cs_updates, accepted_indices_per_cs, 0.5, min_samples)
        
      else:
        global_update = cs_aggregated_updates[0]
        print("Only 1 CS")

      for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key] 

    elif config["aggregation"] == "FLECAv2":
      gamma = config.get("gamma", 0.4)
      kappa = config.get("kappa", 1.0)
      t = round_num
      lambda_t = 1.0 * (1 - math.exp(-0.1 * t))

      # Step 1: Intra-CS Aggregation with Threshold Filtering
      cs_aggregated_updates = []
      for i, cs in enumerate(cs_assignments):
        threshold = gamma * np.exp(-kappa * lambda_t)
        print(f"Threshold (EV Level): {threshold} | CS: {i}")
        ev_aggregated_updates, accepted_indices_per_ev = threshold_filtering_and_aggregation_with_own_reference(cs, threshold)

        if ev_aggregated_updates:
            # Apply robust clustering at CS level
            min_samples = k//2  + 1 
            cs_final_update = clustering_aggregation(ev_aggregated_updates, 0.8, min_samples)
            cs_aggregated_updates.append(cs_final_update)
            
        else:
            print("No valid aggregated model for this CS after threshold filtering.")

      # Step 2: Inter-CS Aggregation
      if len(cs_aggregated_updates) > 1:
        # Apply robust clustering at inter-CS level
        global_update = clustering_aggregation(cs_aggregated_updates, 0.5, min_samples)
      else:
        global_update = cs_aggregated_updates[0]
        print("Only 1 CS")

      for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key] 

    elif config["aggregation"] == "Krum":
        f = config.get("f", 1)
        aggregated_cs_updates = [krum_aggregation(cs_clients_updates,f) for cs_clients_updates in cs_assignments]
        print(f"aggregated_models: {len(aggregated_cs_updates)}")
        #global_update = clustering_aggregation(aggregated_cs_updates, num_clusters)
        global_update = krum_aggregation(aggregated_cs_updates, f)
        for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key]
        
    elif config["aggregation"] == "Multi-Krum":
        f = config.get("f", 1)
        aggregated_cs_updates = [multi_krum_aggregation(cs_clients_updates,f) for cs_clients_updates in cs_assignments]
        print(f"aggregated_models: {len(aggregated_cs_updates)}")
        global_update = multi_krum_aggregation(aggregated_cs_updates, f)
        #global_update = clustering_aggregation(aggregated_cs_updates, num_clusters)
        for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key]
    
    elif config["aggregation"] == "Trimmed-Mean":
        beta = config.get("beta", 0.1)
        aggregated_cs_updates = [trimmed_mean_aggregation(cs_clients,beta) for cs_clients in cs_assignments]
        print(f"aggregated_models: {len(aggregated_cs_updates)}")
        global_update = trimmed_mean_aggregation(aggregated_cs_updates, beta)
        #global_update = clustering_aggregation(aggregated_cs_updates, num_clusters)
        for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key]

    elif config["aggregation"] == "FLAME":
      
      aggregated_cs_updates = [flame_aggregation(cs_clients_updates, global_model_dict, lamda=0.001, eta=1.0) 
                             for cs_clients_updates in cs_assignments]
      print(f"CS aggregated_models: {len(aggregated_cs_updates)}")

      # Inter-CS aggregation also with FLAME
      global_update = flame_aggregation(aggregated_cs_updates, global_model_dict, lamda=0.001, eta=1.0)

      for key in global_model_dict:
        global_model_dict[key] += global_update[key] 

    else:
        raise ValueError(f"Unsupported aggregation method: {config['aggregation']}")


    return global_model_dict


def evaluate_anomaly_detection(y_true,  y_pred_clean , y_pred_backoored, backdoored_test_indices=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred_clean > 0.5),
        "Precision": precision_score(y_true, y_pred_clean > 0.5),
        "Recall": recall_score(y_true, y_pred_clean > 0.5),
        "F1-Score": f1_score(y_true, y_pred_clean > 0.5),
        "AUROC": roc_auc_score(y_true, y_pred_clean),
        "ASR": 0,
        "Confusion Matrix": confusion_matrix(y_true, y_pred_clean > 0.5)
    }


    if backdoored_test_indices and len(backdoored_test_indices) > 0:
      
      backdoor_predictions = (y_pred_backoored[backdoored_test_indices] > 0.5).astype(float)
      original_predictions = (y_true[backdoored_test_indices] > 0.5).astype(float)
      # Count cases where anomalies (class 1) are misclassified as normal (class 0)
      attack_success = np.sum((backdoor_predictions == 0) & (original_predictions == 1))
      
      asr = attack_success / len(backdoored_test_indices)

      metrics["ASR"] = asr  # Update ASR metric
      print(f"ASR/BA: {asr:.4f} ({attack_success}/{len(backdoored_test_indices)})")

    return metrics


def evaluate_capacity_estimation(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    return metrics


def print_metrics(metrics, title):
    print(f"\n{title}")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
        if key == "ASR":
            print(f"Attack Success Rate (ASR): {value:.4f}")
