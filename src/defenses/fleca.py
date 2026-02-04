import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import hdbscan   
from defenses.utils import fed_average
#default tau=0.1, kappa = 1.0,
def adaptive_threshold_filtering(client_updates: dict, reference_id: str, tau=0.5, kappa = 1.0, round_num=1, total_rounds=100):
      """
      EV-level filtering with adaptive MAD threshold (FLECAv1, FLECAv2).
      Returns per-EV aggregated models + accepted client IDs.
      """

      assert reference_id in client_updates
      ref_weights = client_updates[reference_id]
      diff_scores = {}

      for cid, weights in client_updates.items():
            max_diff = 0.0
            for key in ["classifier.2.weight", "regressor.2.weight"]:
                ref = ref_weights[key].cpu().numpy()
                cur = weights[key].cpu().numpy()
                max_diff = max(
                    max_diff,
                    np.linalg.norm(cur - ref) / (np.linalg.norm(ref) + 1e-8)
                )
            diff_scores[cid] = max_diff

      scores = np.array(list(diff_scores.values()))
      med = np.median(scores)
      mad = np.median(np.abs(scores - med)) + 1e-8

      threshold = (med + tau * mad) / (1 + kappa *  ((round_num +40) / (total_rounds+40))) #((round_num) / (total_rounds))) #

      accepted = {
            cid: client_updates[cid]
            for cid, s in diff_scores.items()
            if s <= threshold
      }

      if not accepted:
            accepted = {reference_id: client_updates[reference_id]}

      agg, ids = fed_average(accepted)
      
      print(f"[EV {reference_id}] accepted={ids}, tau={tau:.4f}, kappa={kappa:.4f}, threshold={threshold:.4f}")

      return agg, ids

def fleca_filtering(client_updates: dict, round_num=1,total_rounds=100):
    cs_updates = {}
    cs_accepted_ids = {}
    for ref_id, _ in client_updates.items():
       cs_updates[ref_id], cs_accepted_ids[ref_id] = adaptive_threshold_filtering(client_updates=client_updates,
                                             reference_id=ref_id, round_num=round_num, total_rounds=total_rounds)

    return cs_updates, cs_accepted_ids        
    
def majority_voting(aggregated_models: dict, accepted_ids_per_ev: dict):
    """
    Majority voting with robust fallback based on occurrence frequency (FLECAv1).
    """

    vote_counter = {}

    # Count votes
    for ids in accepted_ids_per_ev.values():
        for cid in ids:
            vote_counter[cid] = vote_counter.get(cid, 0) + 1

    if not vote_counter:
        raise RuntimeError("Majority voting failed — no votes available.")

    num_evs = len(accepted_ids_per_ev)

    # -------- Strict Majority -------- #
    majority_ids = {
        cid for cid, cnt in vote_counter.items()
        if cnt > num_evs / 2
    }

    # -------- Fallback: Max Occurrence -------- #
    if not majority_ids:
        max_votes = max(vote_counter.values())
        majority_ids = {
            cid for cid, cnt in vote_counter.items()
            if cnt == max_votes
        }
        print(
            f"[Majority Voting - Fallback] "
            f"No strict majority, selected max-occurrence IDs "
            f"(votes={max_votes}): {majority_ids}"
        )
    else:
        print(f"[Majority Voting] selected IDs: {majority_ids}")

    # Select models
    selected = {
        cid: aggregated_models[cid]
        for cid in majority_ids
        if cid in aggregated_models
    }

    if not selected:
        raise RuntimeError(
            "Majority voting failed — selected IDs missing from aggregated models."
        )

    return fed_average(selected)

def robust_intra_cs_clustering(
      ev_updates: dict,
      device="cpu",
      min_cluster_size=2
      ):
      """
      Robust clustering at EV level inside one CS (FLECAv2).

      Args:
        ev_updates: {ev_id: state_dict}
      Returns:
        filtered_ev_updates: {ev_id: state_dict}
      """

      if len(ev_updates) <= 1:
        return ev_updates

      ev_ids = list(ev_updates.keys())
      updates = list(ev_updates.values())

      # --- Extract task-specific representation ---
      task_keys = [
        k for k in updates[0]
        if "classifier.2.weight" in k or "regressor.2.weight" in k
      ]

      def flatten(weights):
        return np.concatenate([
            weights[k].detach().cpu().numpy().flatten()
            for k in task_keys
        ])

      X = np.stack([flatten(w) for w in updates]).astype(np.float64)

      # --- Distance matrix ---
      dist_matrix = euclidean_distances(X)

      # --- HDBSCAN clustering ---
      clustering = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True
      ).fit(dist_matrix)

      labels = clustering.labels_
      #print(f"[Intra-CS] EV labels: {dict(zip(ev_ids, labels))}")

      # --- Keep largest non-noise cluster ---
      valid_labels = labels[labels != -1]

      if len(valid_labels) == 0:
        print("[Intra-CS] All EVs marked as noise → keep all")
        return ev_updates

      unique, counts = np.unique(valid_labels, return_counts=True)
      largest_label = unique[np.argmax(counts)]

      benign_indices = [
        i for i, l in enumerate(labels)
        if l == largest_label
      ]

      filtered_ev_updates = {
        ev_ids[i]: updates[i]
        for i in benign_indices
      }

      print(f"[Intra-CS] Selected EVs: {list(filtered_ev_updates.keys())}, min_cluster_size={min_cluster_size}")

      return filtered_ev_updates

def robust_inter_cs_clustering(
      cs_updates: dict,
      device="cpu",
      min_cluster_size=3
      ):
      """
      Robust clustering for inter-CS filtering (FLECAv1, FLECAv2).

      Args:
        cs_updates: {cs_id: state_dict}
      Returns:
        filtered_cs_updates: {cs_id: state_dict}
      """

      if len(cs_updates) <= 1:
        return cs_updates

      cs_ids = list(cs_updates.keys())
      updates = list(cs_updates.values())

      # --- Extract task-specific weights (shared semantic space) ---
      task_keys = [
        k for k in updates[0]
        if "classifier.2.weight" in k or "regressor.2.weight" in k
      ]

      def flatten(weights):
        return np.concatenate([
            weights[k].detach().cpu().numpy().flatten()
            for k in task_keys
        ])

      X = np.stack([flatten(w) for w in updates]).astype(np.float64)

      # --- Distance matrix ---
      dist_matrix = euclidean_distances(X)

      # --- HDBSCAN clustering ---
      clustering = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        allow_single_cluster=True
      ).fit(dist_matrix)

      labels = clustering.labels_
      #print(f"[Inter-CS] HDBSCAN labels: {dict(zip(cs_ids, labels))}")

      # --- Select largest non-noise cluster ---
      valid_labels = labels[labels != -1]

      if len(valid_labels) == 0:
        print("[Inter-CS] All CS marked as noise → keep all")
        return cs_updates

      unique, counts = np.unique(valid_labels, return_counts=True)
      largest_label = unique[np.argmax(counts)]

      benign_indices = [
        i for i, l in enumerate(labels)
        if l == largest_label
      ]

      filtered_cs_updates = {
        cs_ids[i]: updates[i]
        for i in benign_indices
      }

      print(f"[Inter-CS] Selected CS IDs: {list(filtered_cs_updates.keys())}", f"min_cluster_size={min_cluster_size}")

      return filtered_cs_updates

    