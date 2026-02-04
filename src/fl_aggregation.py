# ------------------------ Federated Learning Clustered AGGREGATION ---------------------- #
import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

from defenses.utils import fed_average
from defenses.trim import trimmed_mean_aggregation
from defenses.krum import multi_krum_aggregation
from defenses.clip import norm_clipping_aggregation, weak_dp_aggregation
from defenses.flame import flame_aggregation
from defenses.scclip import scclip_aggregation
from defenses.ubar import ubar_filtering
from defenses.fleca import adaptive_threshold_filtering, majority_voting, robust_inter_cs_clustering, robust_intra_cs_clustering, fleca_filtering
import random

def aggregate_updates(clients_params, clients_losses, global_model, config, round_num):
    """
    Aggregates model weights using FedAvg, FedProx, FLECA, Krum, or Trimmed-Mean.
    """
    device = config['device']
    global_model_dict = global_model.state_dict()

    # Common helper function
    def assign_to_cs(clients_params: dict, clients_losses: dict, k: int):
      """
      Split clients into Charging Stations (CS) while preserving client IDs.
      """
      client_ids = list(clients_params.keys())
      all_clients_params = list(clients_params.items())
      all_clients_losses = list(clients_losses.items())
      cs_assignments = []
      cs_losses = []

      for i in range(0, len(all_clients_params), k):
          cs_assignments.append(dict(all_clients_params[i:i + k]))
          cs_losses.append(dict(all_clients_losses[i:i + k]))

      return cs_assignments, cs_losses

    # Two-level aggregation process
    cs_assignments, cs_losses = assign_to_cs(clients_params, clients_losses, config["k"])
    
    if config["flow"] == "DFL" and config["aggregation"] in ["FedAvg", "FedProx"]:
      alpha = config.get("alpha", 0.5) 
      new_client_updates = {}
      for cs_id, cs_clients in enumerate(cs_assignments):
        print(f"Aggregating CS{cs_id} with {len(cs_clients)} clients using {config['aggregation']}...")
    
        for client_id, client_weights in cs_clients.items():
          # Exclude own model for this client
          other_clients = {cid: w for cid, w in cs_clients.items() if cid != client_id}
          if not other_clients:
             # If this is the only client, just keep its own model
            cs_agg = client_weights
          else:
            cs_agg, _ = fed_average(other_clients)

          # Mix own model with CS aggregate
          mixed_weights = {}
          for key in client_weights:
            mixed_weights[key] = alpha * cs_agg[key] + (1 - alpha) * client_weights[key]

          new_client_updates[client_id] = mixed_weights

      # pick one client as a representative "global" model for evaluation
      global_update = next(iter(new_client_updates.values()))

    elif config["aggregation"] in ["FedAvg", "FedProx"]:
      cs_updates = {}
      for cs_id, cs_ev_weights in enumerate(cs_assignments):
        print(f"Aggregating CS{cs_id} with {len(cs_ev_weights)} clients using {config['aggregation']}...")
        agg, ids = fed_average(cs_ev_weights)
        cs_updates[f"CS{cs_id}"] = agg

      print(f"Performing global aggregation of {len(cs_updates)} CS updates using FedAvg...")
      global_update, _ = fed_average(cs_updates)  

    elif config["aggregation"] == "FLECAv1":    
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        ev_updates, accepted_ids = fleca_filtering(
          cs_ev_updates,
          round_num=round_num
        )
        cs_update, _ = majority_voting(ev_updates, accepted_ids)
        cs_updates[f"CS{cs_id}"] = cs_update

      filtered_cs_updates = robust_inter_cs_clustering(cs_updates)
      global_update, _ = fed_average(filtered_cs_updates)

    elif config["aggregation"] == "FLECAv2":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        ev_updates, accepted_ids = fleca_filtering(
          cs_ev_updates,
          round_num=round_num,
          total_rounds=config["num_rounds"]
        )
        filtered_ev_updates = robust_intra_cs_clustering(ev_updates)
        cs_update, _ = fed_average(filtered_ev_updates)
        cs_updates[f"CS{cs_id}"] = cs_update

      filtered_cs_updates = robust_inter_cs_clustering(cs_updates)
      global_update, _ = fed_average(filtered_cs_updates)
    
    # FLECA with EV churn simulation
    elif config["aggregation"] == "FLECA":
      cs_level_updates = {}
      rho = config.get("churn_rate", 0.0)
      random.seed(config.get("seed", 42))
      malicious_ev_ids = [6, 5, 4, 3, 13, 12, 11, 10, 20, 19, 18, 17, 27, 26, 25, 24, 34, 33, 32, 31, 41, 40, 39, 38, 48, 47, 46, 45, 55, 54, 53, 52, 62, 61, 60, 59, 58, 57, 69, 68, 67, 66, 65, 64, 76, 75, 74, 73, 72, 71, 83, 82, 81, 80, 79, 78]
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        # Apply EV churn (benign only)
        active_ev_updates = {}
        for ev_id, update in cs_ev_updates.items():
            # Keep malicious EVs
            if ev_id in malicious_ev_ids or random.random() > rho:
               active_ev_updates[ev_id] = update
            else : 
               print(f"[Round {round_num}] CS{cs_id}: Churn EV ID: {ev_id}")   

        # Print/log kept EV IDs
        print(f"[Round {round_num}] CS{cs_id}: keeping EV IDs -> {list(active_ev_updates.keys())}")
        # Skip under-populated CSs
        if len(active_ev_updates) < config.get("min_ev_per_cs", 2):
            continue

        # FLECA Intra filtering
        ev_updates, accepted_ids = fleca_filtering(
           active_ev_updates,
           round_num=round_num,
           total_rounds=config["num_rounds"]
           )
        filtered_ev_updates = robust_intra_cs_clustering(ev_updates)
        cs_update, _ = fed_average(filtered_ev_updates)
        cs_level_updates[f"CS{cs_id}"] = cs_update

      filtered_cs_updates = robust_inter_cs_clustering(cs_level_updates)
      global_update, _ = fed_average(filtered_cs_updates)   

    elif config["aggregation"] == "Trimmed-Mean": 
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update, _ = trimmed_mean_aggregation(
          client_updates=cs_ev_updates,
          beta=config["beta"],
          device=device
          )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = trimmed_mean_aggregation(
        client_updates=cs_updates,
        beta=config["beta"],
        device=device
        )
    elif config["aggregation"] == "Multi-Krum":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update, _ = multi_krum_aggregation(
          client_updates=cs_ev_updates,
          f=config["f"]
        )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = multi_krum_aggregation(
        client_updates=cs_updates,
        f=config["f"]
      )

    elif config["aggregation"] == "Norm-Clip":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update, _ = norm_clipping_aggregation(
          client_updates=cs_ev_updates,
          device=device
        )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = norm_clipping_aggregation(
        client_updates=cs_updates,
        device=device
      )

    elif config["aggregation"] == "Weak-DP":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update, _ = weak_dp_aggregation(
          client_updates=cs_ev_updates,
          noise_std=0.001,
          device=device
        )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = weak_dp_aggregation(
        client_updates=cs_updates,
        noise_std=0.001,
        device=device
      )

    elif config["aggregation"] == "Flame":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update, _ = flame_aggregation(
          client_updates=cs_ev_updates,
          device=device
        )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = flame_aggregation(
        client_updates=cs_updates,
        device=device
      )  

    elif config["aggregation"] == "SCCLIP":
      cs_updates = {}
      for cs_id, cs_ev_updates in enumerate(cs_assignments):
        cs_update = scclip_aggregation(
          client_updates=cs_ev_updates,
          device=device
        )
        cs_updates[f"CS{cs_id}"] = cs_update

      global_update, _ = norm_clipping_aggregation(
        client_updates=cs_updates,
        device=device
      )  

    elif config["aggregation"] == "UBAR":    
      cs_updates = {}
      for cs_id, (cs_ev_updates, cs_ev_losses) in enumerate(zip(cs_assignments, cs_losses)):
        ev_updates, accepted_ids = ubar_filtering(
          cs_ev_updates,
          cs_ev_losses,
        )
        cs_update, _ = majority_voting(ev_updates, accepted_ids)
        cs_updates[f"CS{cs_id}"] = cs_update

      filtered_cs_updates = robust_inter_cs_clustering(cs_updates)
      global_update, _ = fed_average(filtered_cs_updates)     

    else:
        raise ValueError(f"Unsupported aggregation method: {config['aggregation']}")

    for k in global_model_dict:
        global_model_dict[k] += global_update[k]

    return global_model_dict
