import numpy as np
import math
import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


def fed_average(client_updates: dict):
      """
      FedAvg over client updates.
      Args:
        client_updates: {client_id: state_dict}
      Returns:
        aggregated_state_dict, contributing_client_ids
      """
      assert len(client_updates) > 0

      client_ids = list(client_updates.keys())
      updates = list(client_updates.values())

      aggregated = {
        k: torch.zeros_like(v)
        for k, v in updates[0].items()
      }

      for update in updates:
        for k in aggregated:
            aggregated[k] += update[k]

      for k in aggregated:
        aggregated[k] /= len(updates)

      return aggregated, client_ids
