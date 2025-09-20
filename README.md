# ABC-DFL

Welcome to the **ABC-DFL** project — a clustered, decentralized, and resilient framework for federated learning in connected EVs. This repository includes:

- ✅ Implementation of the **FLECA** filtering and aggregation protocol  
- ✅ Smart contracts for trust and coordination  
- ✅ Benchmark tests for performance evaluation

For Oracles and L2 integration, check our [AutoDFL](https://github.com/meryemmalakdif/AutoDFL) repository. 

---

## What is ABC-DFL?

**ABC-DFL** (A Byzantine-Robust Clustered Decentralized Federated Learning Framework for Secure and Efficient EV Battery Data Management) is a framework designed to securely and efficiently manage federated learning tasks across clustered and dynamic networks of Electric Vehicles (EVs) and Charging Stations (CSs). It tackles **model poisoning attacks** using a robust decentralized aggregation mechanism.

---

## About FLECA

At the core of ABC-DFL lies **FLECA**:  
> **F**iltered  
> **L**ayered  
> **E**nhanced  
> **C**lustering  
> **A**ggregation  

FLECA uses a **two-stage filtering process**:
- **Stage 1:** Performed locally at each **Electric Vehicle (EV)**
- **Stage 2:** Executed by decentralized **Oracles**

Importantly, FLECA operates **without relying on a reference model**, ensuring resilience, decentralization, and adaptability in dynamic environments.

---

## Getting Started

You can run the following commands to explore and test ABC-DFL smart contracts:

```bash

# Run smart contract tests
npx hardhat test

# Run tests with gas usage reporting
REPORT_GAS=true npx hardhat test

# Start a local Hardhat blockchain node
npx hardhat node

# Deploy smart contracts using Hardhat
npx hardhat run scripts/deploy_1.js

```

## Run FLECA Simulations

To simulate the FLECA protocol:

- The main simulation configuration is defined in the `fl_exec.py` file.
- To change the malicious clients (EVs or CSs) rate, refer to the `initialize_attack_clients()` function in `fl_main.py`.
- A subset of the [EVBattery Dataset](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBbkU4QmZIZTNJT2xnMTN2Mmx0VjBlUDEtQWdQP2U9OW80emdM&id=A583DCDEF1053C71%21477&cid=A583DCDEF1053C71)
dataset is used for testing and is included in the `FLECA/` folder.

  

