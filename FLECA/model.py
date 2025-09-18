# ---------------------------- LSTM Model ------------------------------- #


import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared LSTM Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        shared_repr = out[:, -1, :]  # Last time step
        class_output = self.classifier(shared_repr).squeeze()
        reg_output = self.regressor(shared_repr).squeeze()
        return class_output, reg_output

class MultiTaskCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.2):
        super(MultiTaskCNN, self).__init__()

        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling to get a fixed size

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Change shape to (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # Apply pooling and remove the sequence dimension

        class_output = self.classifier(x).squeeze()
        reg_output = self.regressor(x).squeeze()
        return class_output, reg_output

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(MultiTaskBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM Encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        shared_repr = out[:, -1, :]  # Last time step
        class_output = self.classifier(shared_repr).squeeze()
        reg_output = self.regressor(shared_repr).squeeze()
        return class_output, reg_output

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoding_dim, num_heads=8, num_layers=2, dropout=0.2):
        super(MultiTaskTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Transformer expects input of shape (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # Swap batch and seq_len dimensions
        transformer_out = self.transformer(x)
        shared_repr = transformer_out[-1, :, :]  # Use the last output from the sequence

        class_output = self.classifier(shared_repr).squeeze()
        reg_output = self.regressor(shared_repr).squeeze()
        return class_output, reg_output

class MultiTaskGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(MultiTaskGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared GRU Encoder
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        shared_repr = out[:, -1, :]  # Last time step
        class_output = self.classifier(shared_repr).squeeze()
        reg_output = self.regressor(shared_repr).squeeze()
        return class_output, reg_output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcul de la probabilité prédite
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probabilités
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None, reduction='mean'):
        """
        Focal Loss avec prise en charge des poids de classe (Cost-Sensitive).

        Args:
        - alpha (float): Poids global pour ajuster l'impact de Focal Loss.
        - gamma (float): Paramètre de focalisation pour accorder plus d'importance aux exemples difficiles.
        - class_weights (list or tensor): Poids spécifiques par classe (e.g., [0.5, 2.0]).
        - reduction (str): Méthode de réduction ('mean', 'sum', ou 'none').
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calcul de la perte de base (BCE Loss)
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probabilité prédite

        # Application de Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Application des poids de classe
        if self.class_weights is not None:
            weights = self.class_weights[targets.long()]  # Poids associés aux classes cibles
            F_loss = F_loss * weights

        # Réduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
