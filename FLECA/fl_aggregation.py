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


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

def aggregate_models(client_weights, global_model, config, round_num):
    """
    Aggregates model weights using FedAvg, FedProx, FLECA, Krum, or Trimmed-Mean.
    """
    global_model_dict = global_model.state_dict() 
    # Common helper functions
    def assign_clients_to_cs(client_weights, k):
        num_clients = len(client_weights)
        num_cs = math.ceil(num_clients / k)
        cs_assignments = [[] for _ in range(num_cs)]
        for i, client in enumerate(client_weights):
            cs_idx = i // k
            cs_assignments[cs_idx].append(client)
        return cs_assignments

    def simple_average(weights_list):
     
      aggregated_model = copy.deepcopy(global_model_dict)
     
      for key in global_model_dict.keys():
        aggregated_model[key] = sum([weight[key] for weight in weights_list]) / len(weights_list)

      return aggregated_model
    
    def fed_average(list_updates):
      aggregated_model = copy.deepcopy(global_model_dict)
      local_weight_diff_sum = {}
      for key in global_model_dict.keys():
        local_weight_diff_sum[key] = sum([update[key] for update in list_updates]) 
      # Apply the aggregation rule: Gt+1 = Gt + (1 / n) * Σ (Lt+1_i − Gt)  
      for key in global_model_dict.keys():
        aggregated_model[key] = local_weight_diff_sum[key] / len(list_updates)
      return aggregated_model

    def cosine_similarity(selected_keys, local_weights_dict):
      """
      Compute cosine similarity between selected layers of global and local models.
      """
      global_params = []
      local_params = []
      for key in selected_keys:
        global_tensor = global_model_dict[key].cpu().numpy().flatten()
        local_tensor = local_weights_dict[key].cpu().numpy().flatten()
        global_params.append(global_tensor)
        local_params.append(local_tensor)
    
      global_vector = np.concatenate(global_params)
      local_vector = np.concatenate(local_params)
    
      return 1 - cosine(global_vector, local_vector)  # cosine distance to similarity

    def clustering_cosine_filtering(client_updates_list):
      """
      Filtering based on cosine similarity between local updates and global model's last layers,
      followed by clustering-based thresholding. Returns list of (index, weight_dict).
      """
      selected_keys = ["classifier.2.weight", "regressor.2.weight"]

      cosine_scores = []
    
      for weights_dict in client_updates_list:
        cos_sim = cosine_similarity(selected_keys, weights_dict)
        cosine_scores.append(cos_sim)

      min_cs = min(cosine_scores)
      max_cs = max(cosine_scores)
      normalized_cs = [(cs - min_cs) / (max_cs - min_cs) if max_cs != min_cs else 0.0 for cs in cosine_scores]

      #print(f"Normalized Cosine Similarities: {normalized_cs}")
      threshold = np.mean(normalized_cs)
      #print(f"Mean Threshold: {threshold}")

      accepted_models = []
      for idx, norm_score in enumerate(normalized_cs):
        if norm_score < threshold:
            accepted_models.append((idx, client_updates_list[idx]))

      print(f"Accepted EVs: {[idx for idx, _ in accepted_models]}")

      return accepted_models

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
      labels = clustering.labels_

      # Filter out noise (label = -1)
      filtered_models = [updates_list[i] for i in range(len(labels)) if labels[i] != -1]

      print(f"DBSCAN labels: {labels}")
      print(f"Filtered models (non-outliers): {len(filtered_models)}")
 
      if not filtered_models:
        print("Warning: all models considered outliers, returning global average.")
        return fed_average(updates_list)

      return fed_average(filtered_models)

    def clustering_aggregation_v0(updates_list, num_clusters):
      
      """Perform clustering aggregation at the global level."""

      if len(updates_list) < num_clusters:
        print(f"Warning: Not enough models for clustering (only {len(updates_list)} models). Using FedAvg directly.")
        return fed_average(updates_list)

      model_vectors = [
        np.concatenate([w[key].cpu().numpy().flatten() for key in w.keys()])
        for w in updates_list
      ]
      model_vectors = np.array(model_vectors)

      kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
      cluster_labels = kmeans.fit_predict(model_vectors)

      largest_cluster_label = max(set(cluster_labels), key=list(cluster_labels).count)
      largest_cluster_indices = [i for i, label in enumerate(cluster_labels) if label == largest_cluster_label]

      clustered_updates = [updates_list[i] for i in largest_cluster_indices]
      print(f"clustered_models:{len(clustered_updates)}")

      return fed_average(clustered_updates) 
    
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
    
    # Two-level aggregation process
    k = config["k"]
    num_clusters = config.get("num_clusters", 2)
    cs_assignments = assign_clients_to_cs(client_weights, k)

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
      ev_aggregated_updates_per_cs = []
      
      # Step 1: Intra-CS Aggregation with Threshold Filtering
      for i, cs in enumerate(cs_assignments):
        threshold = gamma * np.exp(-kappa * lambda_t)
        print(f"Threshold (EV Level): {threshold} | CS: {i}")
        ev_aggregated_updates, accepted_indices_per_ev = threshold_filtering_and_aggregation_with_own_reference(cs, threshold)

        if ev_aggregated_updates:
            # Apply majority voting at CS level
            #cs_final_update = majority_voting(ev_aggregated_updates, accepted_indices_per_ev)
            min_samples = k//2  + 1 
            cs_final_update = clustering_aggregation(ev_aggregated_updates, 0.8, min_samples)
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
        #global_update = majority_voting(filtered_cs_updates, accepted_indices_per_cs)
        min_samples = config["num_clients"] // (2*k)   
        global_update = clustering_aggregation(filtered_cs_updates, 0.5, min_samples)
      else:
        global_update = cs_aggregated_updates[0]
        print("Only 1 CS")

      for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key] 

    elif config["aggregation"] == "Krum":
        f = config.get("f", 1)
        aggregated_cs_updates = [krum_aggregation(cs_clients,f) for cs_clients in cs_assignments]
        print(f"aggregated_models: {len(aggregated_cs_updates)}")
        #global_update = clustering_aggregation(aggregated_cs_updates, num_clusters)
        global_update = krum_aggregation(aggregated_cs_updates, f)
        for key in global_model_dict:
           global_model_dict[key] = global_update[key] + global_model_dict[key]
        

    elif config["aggregation"] == "Multi-Krum":
        f = config.get("f", 1)
        aggregated_cs_updates = [multi_krum_aggregation(cs_clients,f) for cs_clients in cs_assignments]
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

      original_labels = y_true[backdoored_test_indices]  # Get ground truth labels
      backdoor_predictions = (y_pred_backoored[backdoored_test_indices] > 0.5).astype(float)
      backdoor_predictions_clean = (y_pred_clean[backdoored_test_indices] > 0.5).astype(float)


      # Count cases where anomalies (class 1) are misclassified as normal (class 0)
      attack_success = np.sum((original_labels == 1) & (backdoor_predictions == 0) ) #& (backdoor_predictions_clean == 1))
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
