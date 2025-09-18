import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preparation(num_clients, path, iid=True, alpha=0.5, batch_size=32, min_factor=3):
    """
    Prepares data for federated learning with optional IID or Non-IID (Dirichlet) distribution.
    
    :param num_clients: Number of clients.
    :param path: Path to the .npz file containing the data.
    :param iid: Boolean, True for IID distribution, False for non-IID.
    :param alpha: Dirichlet parameter to control Non-IIDness (lower = more skewed).
    :param batch_size: Minimum batch size for each client (must be a multiple).
    :param min_factor: Minimum examples per class (min_factor * batch_size).
    :return: Client data, full sequences, test sequences, anomaly test labels, and capacity test labels.
    """
    # Load data
    data = np.load(path)
    sequences = data['sequences']
    anomaly_labels = data['anomaly_labels']
    capacity_labels = data['capacity_labels']
    
    # Train-test split
    sequences_train, sequences_test, anomaly_labels_train, anomaly_labels_test, capacity_labels_train, capacity_labels_test = train_test_split(
        sequences, anomaly_labels, capacity_labels, test_size=0.2, random_state=42
    )
    
    # Normalize sequences
    scaler = StandardScaler()
    sequences_train_reshaped = sequences_train.reshape(-1, sequences_train.shape[-1])
    scaler.fit(sequences_train_reshaped)
    sequences_train = scaler.transform(sequences_train_reshaped).reshape(sequences_train.shape)
    sequences_test = scaler.transform(sequences_test.reshape(-1, sequences_test.shape[-1])).reshape(sequences_test.shape)
    
    # Normalize capacity labels
    capacity_mean = capacity_labels_train.mean()
    capacity_std = capacity_labels_train.std()
    capacity_labels_train = (capacity_labels_train - capacity_mean) / capacity_std
    capacity_labels_test = (capacity_labels_test - capacity_mean) / capacity_std
    
    min_samples_per_client = batch_size  # Ensure each client gets at least this much
    
    if iid:
        # IID: Equally partition the dataset ensuring each client gets at least one batch
        client_data_size = max(1, (len(sequences_train) // num_clients) // batch_size * batch_size)
        clients_data = [
            (sequences_train[i * client_data_size:(i + 1) * client_data_size],
             anomaly_labels_train[i * client_data_size:(i + 1) * client_data_size],
             capacity_labels_train[i * client_data_size:(i + 1) * client_data_size])
            for i in range(num_clients)
        ]
    else:
        # Non-IID: Dirichlet-based partitioning with corrections
        clients_data = [[] for _ in range(num_clients)]
        
        class_indices = {0: np.where(anomaly_labels_train == 0)[0], 
                         1: np.where(anomaly_labels_train == 1)[0]}
        
        for cls in class_indices:
            np.random.shuffle(class_indices[cls])
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            class_splits = (proportions * len(class_indices[cls])).astype(int)
            
            # Ensure minimum allocation
            for i in range(num_clients):
                if class_splits[i] < min_samples_per_client:
                    class_splits[i] = min_samples_per_client
            
            # Adjust total count to match dataset size
            excess = sum(class_splits) - len(class_indices[cls])
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
                selected_indices = class_indices[cls][start_idx:end_idx]
                
                # Ensure batch size alignment
                end_idx = start_idx + (len(selected_indices) // batch_size * batch_size)
                selected_indices = class_indices[cls][start_idx:end_idx]
                
                clients_data[i].extend(selected_indices.tolist())
                start_idx = end_idx
        
        # Convert indices to actual data, ensuring no empty clients
        for i in range(num_clients):
            if not clients_data[i]:  # If still empty, reassign from others
                for j in range(num_clients):
                    if len(clients_data[j]) > min_samples_per_client * 2:
                        clients_data[i] = clients_data[j][:min_samples_per_client]
                        clients_data[j] = clients_data[j][min_samples_per_client:]
                        break

        # Convert to actual dataset
        clients_data = [
            (sequences_train[np.array(client_indices, dtype=int)], 
             anomaly_labels_train[np.array(client_indices, dtype=int)], 
             capacity_labels_train[np.array(client_indices, dtype=int)]) 
            for client_indices in clients_data
        ]
    
    # Print the number of examples per client
    for i, client_data in enumerate(clients_data):
        num_class_0 = np.sum(client_data[1] == 0)
        num_class_1 = np.sum(client_data[1] == 1)
        print(f"Client {i + 1}: {len(client_data[0])} samples (Class 0: {num_class_0}, Class 1: {num_class_1})")
    
    return clients_data, sequences, sequences_test, anomaly_labels_test, capacity_labels_test
