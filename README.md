# ABC-DFL

Welcome to the **ABC-DFL** project — a decentralized and resilient framework for federated learning in dynamic environments. This repository includes:

- ✅ Implementation of the **FLECA** filtering and aggregation mechanism  
- ✅ Smart contracts for trust and coordination  
- ✅ Benchmark tests for performance evaluation  

---

## 🧠 What is ABC-DFL?

**ABC-DFL** (Adaptive Blockchain-Connected Decentralized Federated Learning) is a framework designed to secure and scale federated learning across Electric Vehicles (EVs) and Charging Stations (CSs). It tackles **model poisoning attacks** using a robust and fully decentralized validation mechanism.

---

## 🔍 About FLECA

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

## 🚀 Getting Started

You can run the following commands to explore and test the system:

```bash

# Run smart contract tests
npx hardhat test

# Run tests with gas usage reporting
REPORT_GAS=true npx hardhat test

# Start a local Hardhat blockchain node
npx hardhat node

# Deploy smart contracts using Hardhat Ignition
npx hardhat ignition deploy ./ignition/modules/Lock.js
