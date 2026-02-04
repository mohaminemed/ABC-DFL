# ABC-DFL: Reproducible C-DFL Simulation Framework

This repository provides the official implementation of **ABC-DFL**, built on top of a **Centralized-Decentralized Federated Learning (C-DFL)** simulation pipeline.  
It is designed to reproduce the experimental results reported in the paper, including:

- Multi-task learning (anomaly detection + capacity estimation)
- Adaptive backdoor and poisoning attacks
- Robust aggregation and defense mechanisms
- IID / Non-IID client data partitioning
- Evaluation with reproducible seeds

The framework supports extensible models, attacks, and aggregators, and outputs prediction traces used to compute AIS / ASR and other metrics reported in the paper.

## A. 📁 Repository Structure

```
.
├── fl_exec.py # Entry point (C-DFL experiment launcher)
├── fl_main.py # Core FL loop (FL_main_Loop)
├── preprocessing.py # Dataset loading and client partitioning
├── fl_training.py # Local FL train loop (few epochs) with DP or attack if enabled
├── fl_aggregation.py # C-DFL aggregation
├── models/ # Multi-task models (BiLSTM, LSTM, CNN, ...)
├── defenses/ # Robust aggregation and defense mechanisms used by fl_aggregation.py
├── datasets/
│ └── battery_dataset3_prepared.npz # Prepared battery dataset
├── utils.py # utilities (evaluation, logs)
├── results/ # Generated CSV outputs (predictions, logs)
└── README.md # Project documentation
```

## B. ⚙️ Environment Setup

### 1. Create Conda Environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
Make sure PyTorch matches your CUDA version:

### 3. 🧠 Hardware Requirements
Experiments in the paper were run on:

- NVIDIA GPUs (A6000)
- Linux (24.04.3 LTS)
- Update the device field in BASE_CONFIG if needed

### 4. 📊 Dataset
Place the prepared dataset here:
```bash
./datasets/battery_dataset3_prepared.npz
```
- This file must contain: Training & Test sequences, Anomaly labels, Capacity labels
- The loader is implemented in: preprocessing.py → data_preparation()
- Client partitioning supports: IID Dirichlet Non-IID (α configurable)

## C. 🚀 Running Experiments
Simply run:
```bash
python fl_exec.py
```
This launches the full C-DFL pipeline:
1. Data partitioning across clients
2. Local training
3. Attack injection (if enabled)
4. Robust aggregation
5. Global evaluation
6. CSV export of predictions

### 1. 🔧 Experiment Configuration
All experimental parameters are defined in fl_exec.py.

**Models**
```python
MODELS = [
    "MultiTaskbiLSTM",...
]
```
**Aggregation / Defense**
```python
AGGREGATIONS = [
    "FLECAv2", "UBAR",...
]
```
**Supported alternatives:** 
FedAvg, Trimmed-Mean, Multi-Krum, FLAME, Weak-DP, Norm-Clip, UBAR, and FedProx

**Attacks**
```python
ATTACKS = [
    "gauss", "adaptive",...
]
```
**Also supported:**
1. *Untargetted poisoning:* label flipping, gaussian, krum / trim, feature attack
2. *Targeted poisoning (backdoor):* badnets, neurotoxin, model replacement (scaling)

**IID / Non-IID**
```python
IID_OPTIONS = [False, True]
```
- Non-IID uses *Dirichlet* sampling: "dirichlet_alpha": 0.8

### 2. 🧪 Core Parameters
```python
BASE_CONFIG = {
    "flow": "C-DFL", # C-DFL workflow
    "num_clients": 42, # EV counts E
    "num_rounds": 10, # Total rounds
    "k": 7,   # Group/Cluster size
    ...
    "seed": 42, # Reproducibility
}
```

### 3. 🧬 Multi-Task Learning
The pipline jointly trains:
1. **Binary anomaly detection**
2. **Continuous capacity estimation**
Implemented in: models/MultiTaskbiLSTM.py

### 4. 📤 Output Files
For each run, the system produces:
1. results/<attack>/Y_Pred_Y_True_IID_<iid>_<model>_<aggregation>_<clients>_Client_<rounds>_Rounds.csv
    --> contains: y_true , y_pred
2. results/MultiTaskbiLSTM_IID_<iid>_<model>_<aggregation>_<clients>_Client_<rounds>_Rounds.csv
    --> contains: round,global_loss,anomaly_accuracy,anomaly_recall,anomaly_precision,anomaly_f1_score,anomaly_auroc,anomaly_asr,capacity_mae,capacity_mse,capacity_rmse,capacity_r2
3. All generated file paths are also tracked in:
results/file_paths.txt

### 5. 🔁 Reproducibility
We enforce deterministic behavior via: "seed": 42
For full reproducibility:
1. Use the same dataset
2. Keep fixed seeds
3. Run each experiment multiple times (paper uses ×3 runs with averaging)

**Paper Results.** To reproduce ABC-DFL tables and figures:
1. Run fl_exec.py
2. Collect CSVs in results/
3. Compute AIS / ASR from prediction files
4. Average over repeated runs (change the seed 42, 70, 84)





