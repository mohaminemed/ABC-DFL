import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_preparation(
    num_clients,
    path,
    iid=True,
    alpha=0.5,
    batch_size=32,
    min_factor=3,
    seed=42
):
    """
    Prepares data for federated learning with optional IID or Non-IID (Dirichlet) distribution.

    :param num_clients: Number of clients.
    :param path: Path to the .npz file containing the data.
    :param iid: Boolean, True for IID distribution, False for non-IID.
    :param alpha: Dirichlet parameter to control Non-IIDness (lower = more skewed).
    :param batch_size: Minimum batch size for each client.
    :param min_factor: Minimum examples per class (min_factor * batch_size).
    :param seed: Random seed for full reproducibility.
    :return: Client data, full sequences, test sequences,
             anomaly test labels, and capacity test labels.
    """

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data = np.load(path)
    sequences = data["sequences"]
    anomaly_labels = data["anomaly_labels"]
    capacity_labels = data["capacity_labels"]

    # ------------------------------------------------------------------
    # Train / Test split (reproducible)
    # ------------------------------------------------------------------
    (
        sequences_train,
        sequences_test,
        anomaly_labels_train,
        anomaly_labels_test,
        capacity_labels_train,
        capacity_labels_test,
    ) = train_test_split(
        sequences,
        anomaly_labels,
        capacity_labels,
        test_size=0.2,
        random_state=seed,
    )

    # ------------------------------------------------------------------
    # Normalize sequences
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    seq_train_reshaped = sequences_train.reshape(-1, sequences_train.shape[-1])
    scaler.fit(seq_train_reshaped)

    sequences_train = scaler.transform(seq_train_reshaped).reshape(
        sequences_train.shape
    )
    sequences_test = scaler.transform(
        sequences_test.reshape(-1, sequences_test.shape[-1])
    ).reshape(sequences_test.shape)

    # ------------------------------------------------------------------
    # Normalize capacity labels (train statistics only)
    # ------------------------------------------------------------------
    capacity_mean = capacity_labels_train.mean()
    capacity_std = capacity_labels_train.std()

    capacity_labels_train = (capacity_labels_train - capacity_mean) / capacity_std
    capacity_labels_test = (capacity_labels_test - capacity_mean) / capacity_std

    # ------------------------------------------------------------------
    # Data partitioning
    # ------------------------------------------------------------------
    min_samples_per_client = batch_size

    if iid:
        # --------------------------------------------------------------
        # IID partitioning
        # --------------------------------------------------------------
        client_data_size = max(
            1, (len(sequences_train) // num_clients) // batch_size * batch_size
        )

        clients_data = [
            (
                sequences_train[i * client_data_size : (i + 1) * client_data_size],
                anomaly_labels_train[i * client_data_size : (i + 1) * client_data_size],
                capacity_labels_train[i * client_data_size : (i + 1) * client_data_size],
            )
            for i in range(num_clients)
        ]

    else:
        # --------------------------------------------------------------
        # Non-IID Dirichlet partitioning (reproducible)
        # --------------------------------------------------------------
        clients_indices = [[] for _ in range(num_clients)]

        class_indices = {
            0: np.where(anomaly_labels_train == 0)[0],
            1: np.where(anomaly_labels_train == 1)[0],
        }

        for cls, indices in class_indices.items():
            rng.shuffle(indices)

            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            class_splits = (proportions * len(indices)).astype(int)

            # Ensure minimum allocation
            for i in range(num_clients):
                if class_splits[i] < min_samples_per_client:
                    class_splits[i] = min_samples_per_client

            # Fix overflow
            excess = class_splits.sum() - len(indices)
            while excess > 0:
                for i in range(num_clients):
                    if class_splits[i] > min_samples_per_client:
                        class_splits[i] -= 1
                        excess -= 1
                    if excess == 0:
                        break

            start_idx = 0
            for i in range(num_clients):
                end_idx = start_idx + class_splits[i]
                selected = indices[start_idx:end_idx]

                # Batch-size alignment
                aligned_len = (len(selected) // batch_size) * batch_size
                selected = selected[:aligned_len]

                clients_indices[i].extend(selected.tolist())
                start_idx += class_splits[i]

        # Fix empty clients (rare but safe)
        for i in range(num_clients):
            if len(clients_indices[i]) == 0:
                for j in range(num_clients):
                    if len(clients_indices[j]) > 2 * min_samples_per_client:
                        clients_indices[i] = clients_indices[j][:min_samples_per_client]
                        clients_indices[j] = clients_indices[j][min_samples_per_client:]
                        break

        # Convert indices to data
        clients_data = [
            (
                sequences_train[np.array(idxs, dtype=int)],
                anomaly_labels_train[np.array(idxs, dtype=int)],
                capacity_labels_train[np.array(idxs, dtype=int)],
            )
            for idxs in clients_indices
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    for i, (_, y_anom, _) in enumerate(clients_data):
        n0 = np.sum(y_anom == 0)
        n1 = np.sum(y_anom == 1)
        print(
            f"Client {i + 1}: {len(y_anom)} samples "
            f"(Class 0: {n0}, Class 1: {n1})"
        )

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    return (
        clients_data,
        sequences,
        sequences_test,
        anomaly_labels_test,
        capacity_labels_test,
    )
