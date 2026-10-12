# XAI-ZERODAY-IDS: Explainable AI-Powered Zero-Day Intrusion Prediction System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![XAI Engine](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-orange.svg)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [System Architecture & Working Flow](#system-architecture--working-flow)
  - [Architecture 1: End-to-End Data & Pipeline Architecture](#architecture-1-end-to-end-data--pipeline-architecture)
  - [Architecture 2: Hybrid Consensus & Threat Decision Flowchart](#architecture-2-hybrid-consensus--threat-decision-flowchart)
  - [Architecture 3: Explainable AI (XAI) Audit & Governance Architecture](#architecture-3-explainable-ai-xai-audit--governance-architecture)
  - [Architecture 4: Component Interaction & Inter-Module Communication](#architecture-4-component-interaction--inter-module-communication)
- [Core Modules & Operational Features](#core-modules--operational-features)
- [Empirical Performance & Evaluation Benchmarks](#empirical-performance--evaluation-benchmarks)
- [UNSW-NB15 Benchmark Dataset Overview](#unsw-nb15-benchmark-dataset-overview)
  - [Dataset Partitioning & Split](#dataset-partitioning--split)
  - [Taxonomy of 9 Attack Families](#taxonomy-of-9-attack-families)
  - [Why UNSW-NB15 over Legacy Datasets](#why-unsw-nb15-over-legacy-datasets)
- [UNSW-NB15 Feature Reference Schema](#unsw-nb15-feature-reference-schema)
- [Repository Structure](#repository-structure)
- [Step-by-Step Installation & Execution Guide](#step-by-step-installation--execution-guide)
- [License & Author](#license--author)

---

## Executive Summary

**XAI-ZERODAY-IDS** is an enterprise-grade, real-time Intrusion Detection System (IDS) engineered for Next-Generation Security Operations Centers (SOC). Traditional intrusion detection frameworks rely strictly on static signature databases, rendering them ineffective against previously unseen, zero-day cyber threats. Conversely, deep learning and black-box machine learning approaches produce alerts without human-understandable rationale, leading to alert fatigue and delayed incident response.

**XAI-ZERODAY-IDS** addresses both challenges by uniting supervised machine learning classifiers (**XGBoost**, **Random Forest**) with an unsupervised anomaly detector (**Isolation Forest**) within a multi-model consensus architecture. Furthermore, the system integrates **Explainable AI (XAI)** frameworks (**SHAP** and **LIME**), enabling security analysts to inspect global feature importance rankings and local instance-level decision drivers for every detected packet.

---

## System Architecture & Working Flow

To provide complete transparency into the system's inner workings, **XAI-ZERODAY-IDS** is structured around **four distinct architectural diagrams**:
1. **End-to-End Data & Pipeline Architecture**: Mapping the high-level movement of network connection attributes from ingestion to visualization.
2. **Hybrid Consensus & Threat Decision Flowchart**: Illustrating the exact mathematical rules and voting logic for classifying threats.
3. **Explainable AI (XAI) Audit & Governance Architecture**: Details how SHAP and LIME surrogate models extract feature-level attributions.
4. **Component Interaction & Inter-Module Communication**: Detailing python code-level module dependencies and function invocation paths.

---

### Architecture 1: End-to-End Data & Pipeline Architecture

The following diagram details the flow of data through ingestion, preprocessing, parallel model execution, hybrid consensus calculation, XAI explainer generation, and Streamlit user presentation:

```
+---------------------------------------------------------------------------------------+
|                 ARCHITECTURE 1: END-TO-END DATA & PIPELINE ARCHITECTURE               |
+---------------------------------------------------------------------------------------+

  [ Raw Network Ingestion Sources ]
  +-------------------------------+   +-----------------------------+   +-----------------------------+
  | Live Traffic Stream Simulator |   | Single Packet Form Inputs   |   | Batch PCAP/CSV Log File     |
  | (Dynamic Scenario Generator)  |   | (Manual Feature Testing)    |   | (UNSW-NB15 Bulk Processing) |
  +---------------+---------------+   +--------------+--------------+   +--------------+--------------+
                  |                                  |                                 |
                  +----------------------------------+---------------------------------+
                                                     |
                                                     v
  [ Preprocessing & Transformation Engine ]
  +--------------------------------------------------------------------------------------------------+
  | ColumnTransformer Pipeline                                                                       |
  |  * OneHotEncoder: Converts categorical columns ('proto', 'service', 'state') -> One-Hot Vectors   |
  |  * StandardScaler: Normalizes 39 numerical features (mean=0, variance=1)                         |
  |  * Output Matrix: 194-Dimensional Dense Feature Alignment Vector                                 |
  +--------------------------------------------------+-----------------------------------------------+
                                                     |
                                                     v
  [ Multi-Model Parallel Inference Engine ]
  +--------------------------------------------------+-----------------------------------------------+
  |  +--------------------------------------------+   |   +---------------------------------------+  |
  |  | Supervised Attack Estimators               |   |   | Unsupervised Anomaly Estimator        |  |
  |  |                                            |   |   |                                       |  |
  |  | 1. XGBoost (Gradient Boosted Trees)        |   |   | 3. Isolation Forest (iForest)         |  |
  |  |    -> Returns P(Attack) & Class Verdict    |   |   |    -> Returns Anomaly Score & Outlier |  |
  |  |                                            |   |   |       Decision Path (-1 / +1)         |  |
  |  | 2. Random Forest (100 Decision Trees)      |   |   |                                       |  |
  |  |    -> Returns P(Attack) & Class Verdict    |   |   |                                       |  |
  |  +---------------------+----------------------+   |   +-------------------+-------------------+  |
  +------------------------|--------------------------+-----------------------|----------------------+
                           |                                                  |
                           +------------------------+-------------------------+
                                                    |
                                                    v
  [ Hybrid Consensus & Aggregation Layer ]
  +--------------------------------------------------------------------------------------------------+
  |  * Applies 3-Way Rule Decision Engine (Normal vs. Attack vs. Zero-Day Anomaly)                   |
  |  * Computes Composite Risk Percentage Score: Risk Score = w_xgb * P_xgb + w_rf * P_rf + w_iso * S_iso|
  +--------------------------------------------------+-----------------------------------------------+
                                                     |
                                                     v
  [ Explainable AI (XAI) Forensic Engine ]
  +--------------------------------------------------------------------------------------------------+
  |  * SHAP Engine: TreeExplainer Shapley Values for Global Feature Importance                       |
  |  * LIME Engine: Local Linear Surrogate Model for Packet-Level Feature Weights                      |
  +--------------------------------------------------+-----------------------------------------------+
                                                     |
                                                     v
  [ Presentation & SOC User Interface Layer ]
  +--------------------------------------------------------------------------------------------------+
  | Streamlit Multi-Module Operations Dashboard (Live Monitor, Inspector, Batch Log, Telemetry, XAI) |
  +--------------------------------------------------------------------------------------------------+
```

---

### Architecture 2: Hybrid Consensus & Threat Decision Flowchart

The following flowchart illustrates the exact algorithmic decision rules, voting logic, and thresholding steps used to categorize every network connection into **Normal Traffic**, **Known Malicious Attack**, or **Zero-Day Anomaly**:

```
+---------------------------------------------------------------------------------------+
|               ARCHITECTURE 2: HYBRID CONSENSUS & THREAT DECISION FLOWCHART            |
+---------------------------------------------------------------------------------------+

                         +----------------------------------------+
                         | Preprocessed Network Connection Flow   |
                         |      (194-Dimension Feature Vector)    |
                         +-------------------+--------------------+
                                             |
                        +--------------------+--------------------+
                        |                                         |
                        v                                         v
        +-------------------------------+       +------------------------------------+
        | Supervised ML Models          |       | Unsupervised Anomaly Model         |
        | (XGBoost & Random Forest)     |       | (Isolation Forest)                 |
        +---------------+---------------+       +-----------------+------------------+
                        |                                         |
                        v                                         v
        +-------------------------------+       +------------------------------------+
        | Calculate Attack Probability  |       | Calculate Structural Anomaly Score |
        |   P_sup = max(P_xgb, P_rf)    |       |  Score_iso in [-1.0, +1.0]         |
        +---------------+---------------+       +-----------------+------------------+
                        |                                         |
                        +-------------------+---------------------+
                                            |
                                            v
                                 /---------------------\
                                /   Is P_sup >= 0.50   \
                                \   (Supervised Threat)\
                                 \---------------------/
                                    /               \
                                   /                 \ YES
                               NO /                   \
                                 v                     v
                     /---------------------\   +------------------------------------+
                    /  Is Score_iso == -1   \  |   VERDICT: MALICIOUS ATTACK        |
                    \ (Isol. Forest Outlier)/  |   - Status: "Attack"               |
                     \---------------------/   |   - Action: Block IP & Firewall    |
                        /               \      |   - Risk: High (75% - 100%)        |
                       /                 \ YES +------------------------------------+
                   NO /                   \
                     v                     v
      +----------------------------+   +------------------------------------+
      | VERDICT: NORMAL TRAFFIC    |   | VERDICT: ZERO-DAY ANOMALY          |
      | - Status: "Normal"         |   | - Status: "Normal"                 |
      | - Action: Authorized Pass  |   | - Action: Quarantine & Deep Inspect|
      | - Risk: Low (0% - 25%)     |   | - Risk: Medium-High (50% - 75%)    |
      +----------------------------+   +------------------------------------+
```

---

### Architecture 3: Explainable AI (XAI) Audit & Governance Architecture

The following diagram illustrates how SHAP and LIME interact with trained model estimators and feature transformers to produce global feature rankings and packet-level explanations for security analysts:

```
+---------------------------------------------------------------------------------------+
|             ARCHITECTURE 3: EXPLAINABLE AI (XAI) AUDIT & GOVERNANCE ARCHITECTURE      |
+---------------------------------------------------------------------------------------+

   +---------------------------------+                  +--------------------------------+
   |   Trained XGBoost / RF Models   |                  |  Selected Connection Packet    |
   |      (Model Estimator Objects)  |                  | (Raw Input Record / CSV Row)   |
   +----------------+----------------+                  +---------------+----------------+
                    |                                                   |
                    v                                                   v
   +---------------------------------+                  +--------------------------------+
   | Background Reference Distribution|                 | Feature Preprocessing Matrix   |
   |  (Sample Dataset Matrix: N=100) |                  | (ColumnTransformer Output)     |
   +----------------+----------------+                  +---------------+----------------+
                    |                                                   |
                    +------------------------+--------------------------+
                                             |
                                             v
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
      +-------------------------------------+ +-------------------------------------+
      | GLOBAL EXPLANATION ENGINE: SHAP     | | LOCAL EXPLANATION ENGINE: LIME      |
      |                                     | |                                     |
      | 1. Instantiates TreeExplainer       | | 1. Instantiates LimeTabularExplainer|
      | 2. Computes Shapley Additive        | | 2. Generates Local Perturbation     |
      |    Values across feature space      | |    Samples around the target packet  |
      | 3. Aggregates Mean Absolute SHAP    | | 3. Fits Ridge Regression Surrogate  |
      |    Impact for Global Feature Rank   | | 4. Extracts Positive/Negative       |
      | 4. Generates SHAP Summary Plot &    | |    Feature Contribution Weights     |
      |    Global Feature Bar Charts        | | 5. Maps Feature Ranges to Plain-    |
      |                                     | |    English Field Definitions        |
      +------------------+------------------+ +------------------+------------------+
                         |                                       |
                         +-------------------+-------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      | Streamlit XAI Audit Center UI Presentation  |
                      |  - Global SHAP Summary Plot & Feature Ranks |
                      |  - Local LIME Feature Weight Bar Charts     |
                      |  - Plain-English Feature Glossary Lookup    |
                      +---------------------------------------------+
```

---

### Architecture 4: Component Interaction & Inter-Module Communication

The following diagram maps the explicit Python code-level interactions, module dependencies, and invocation paths between the core `src/` modules and the `ui/app.py` presentation engine:

```
+---------------------------------------------------------------------------------------+
|          ARCHITECTURE 4: COMPONENT INTERACTION & INTER-MODULE COMMUNICATION           |
+---------------------------------------------------------------------------------------+

                         +-----------------------------------+
                         |           src/config.py           |
                         | (Centralized Paths & Constants)   |
                         +-----------------+-----------------+
                                           |
                                           v  (Imports Constants & Schema)
       +-----------------------------------+-----------------------------------+
       |                                   |                                   |
       v                                   v                                   v
+--------------+                   +---------------+                   +---------------+
| src/loader.py|                   |src/generator.p|                   |src/explain.py |
| (Model Assets|                   |(Dynamic Stream|                   |(SHAP & LIME   |
|  & Scalers)  |                   |  Synthesis)   |                   | Explanations) |
+-------+------+                   +-------+-------+                   +-------+-------+
        |                                  |                                   |
        |                                  |                                   |
        +------------------+---------------+-----------------------------------+
                           |
                           v  (Orchestrates Data & Models)
+--------------------------+-----------------------------------------------------------+
|                                    src/hybrid.py                                     |
|  1. Calls src/preprocessing.py -> Transforms raw DataFrames into aligned matrices      |
|  2. Calls Model Objects        -> Runs parallel XGBoost, Random Forest & IsoForest   |
|  3. Evaluates Rule Matrix      -> Computes Hybrid Verdict & Dynamic Risk Score (%)   |
+------------------------------------------+-------------------------------------------+
                                           |
                                           v  (Outputs DataFrames & Metrics)
+------------------------------------------+-------------------------------------------+
|                                      ui/app.py                                       |
|  (Streamlit Multi-Tab SOC Interface: Live Stream, Inspector, Batch, Telemetry, XAI)  |
+--------------------------------------------------------------------------------------+
```

---

## Core Modules & Operational Features

1. **Live Network Stream & Generator**:
   - Simulates continuous network traffic streams with adjustable packet volume and scenario presets (Mixed Stream, DDoS Volumetric, Zero-Day Exfiltration, Normal Web Traffic).
   - Renders live packet tables, risk breakdown pie charts, and downloadable stream logs.

2. **Single Packet Threat Inspector**:
   - Interactive form for manual parameter input across Connection Headers, Traffic Volume, Rates & TTL, and Flow Counters.
   - Provides an expanded technical glossary defining all 42 network flow attributes.

3. **Batch Network Log Analyzer**:
   - Processes bulk CSV network log uploads through the full inference pipeline.
   - Generates downloadable security audit logs with recommended security action plans (Block IP, Quarantine, Safe Traffic).

4. **Engine Telemetry & Benchmarks**:
   - Displays empirical evaluation metrics (Accuracy, Precision, Recall, F1-Score) measured across 82,332 test samples from the UNSW-NB15 dataset.
   - Renders metric comparison charts and confusion matrices.

5. **XAI Threat Audit Center**:
   - Global SHAP feature importance bar charts and summary plots.
   - Instance-level LIME feature weight inspection for individual network packets.

---

## Empirical Performance & Evaluation Benchmarks

The hybrid engine was evaluated on the benchmark **UNSW-NB15** network intrusion dataset (175,341 training samples, 82,332 testing samples across 9 attack categories including Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms).

| Model / Architecture | Accuracy | Precision | Recall | F1-Score | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | **95.36%** | **95.32%** | **98.00%** | **96.64%** | Primary Supervised Threat Detector |
| **Random Forest Classifier** | 87.04% | 81.64% | 98.65% | 89.34% | High-Recall Baseline Classifier |
| **Isolation Forest** | 32.24% | 51.94% | 7.61% | 13.27% | Unsupervised Zero-Day Anomaly Detector |
| **Hybrid Consensus Ensemble** | **95.82%** | **95.20%** | **98.71%** | **96.91%** | Production Decision Engine |

---

## 📊 UNSW-NB15 Benchmark Dataset Overview

### Origin & Background
The **UNSW-NB15** dataset was created by the **Cyber Range Lab of the Australian Centre for Cyber Security (ACCS)** at the University of New South Wales (UNSW Canberra). It was generated using the IXIA PerfectStorm tool to capture a realistic synthesis of modern normal network activity combined with synthetic modern attack behaviors.

### Dataset Partitioning & Split
The project utilizes the official benchmark split of UNSW-NB15:
* **Training Partition (`UNSW_NB15_training-set.csv`)**: 175,341 connection records.
* **Testing Partition (`UNSW_NB15_testing-set.csv`)**: 82,332 connection records.
* **Total Benchmark Corpus**: 257,673 labeled network flow records across 42 predictive features.

### Taxonomy of 9 Attack Families

| Attack Family | Category Description & Threat Behavior | UNSW-NB15 Instance Pattern |
| :--- | :--- | :--- |
| **Fuzzers** | Attempts to discover security vulnerabilities by feeding invalid, unexpected, or random data inputs to protocols. | High packet rate, randomized TTLs, malformed packet sizes. |
| **Analysis** | Intrusion attempts designed to gather target system details via port scanning, spam, and HTML page probing. | Repeated connection attempts across specific HTTP/DNS services. |
| **Backdoors** | Stealthy mechanisms bypassing normal authentication to maintain unauthorized remote access. | Low-frequency persistent TCP streams with unusual window sizes (`swin`/`dwin`). |
| **DoS (Denial of Service)** | Attacks flooding target servers with volumetric traffic to exhaust memory, CPU, or network bandwidth. | Extreme packet transmission rates (`rate` > 5,000 pkts/sec) & high source load (`sload`). |
| **Exploits** | Targeted attacks exploiting known software vulnerabilities, buffer overflows, or privilege escalation bugs. | Specific payload length signatures (`response_body_len`) and flow commands. |
| **Generic** | Cryptographic or protocol-level attacks targeting block cipher primitives independent of specific application bugs. | Consistent collision patterns across state counters (`ct_state_ttl`). |
| **Reconnaissance** | Active scanning and probing activities gathering host OS, open port, and network topology profiles. | Rapid sequential destination port connections (`ct_src_dport_ltm`). |
| **Shellcode** | Small executable payloads injected to spawn interactive command-line shell sessions on vulnerable target hosts. | Short connection duration (`dur`) paired with specific byte ratios (`sbytes`/`dbytes`). |
| **Worms** | Self-propagating malicious software replicating autonomously across host network segments. | High destination host count (`ct_dst_ltm`) over short time windows. |

### Why UNSW-NB15 over Legacy Datasets
Unlike legacy intrusion datasets such as **KDD99** or **NSL-KDD** (which were synthesized in 1999 and suffer from redundant records, outdated attack patterns, and unrealistic traffic distribution artifacts), **UNSW-NB15**:
1. Reflects modern enterprise transport protocols (TCP, UDP, ICMP, ARP, OSPF, SCTP).
2. Contains realistic packet inter-arrival times (`sinpkt`/`dinpkt`) and jitter statistics (`sjit`/`djit`).
3. Incorporates complex low-frequency attack patterns and zero-day anomaly distributions typical of modern network environments.

---

## UNSW-NB15 Feature Reference Schema

The system processes 42 network parameters categorized into:

* **Basic Flow Features**: `dur` (Duration), `proto` (Protocol), `service` (Service), `state` (State), `spkts` (Src Packets), `dpkts` (Dst Packets), `sbytes` (Src Bytes), `dbytes` (Dst Bytes).
* **Packet & Rate Statistics**: `rate` (Packet Rate), `sttl` (Src TTL), `dttl` (Dst TTL), `sload` (Src Load), `dload` (Dst Load), `sloss` (Src Packet Loss), `dloss` (Dst Packet Loss), `sinpkt` (Src Inter-packet Time), `dinpkt` (Dst Inter-packet Time), `sjit` (Src Jitter), `djit` (Dst Jitter), `swin` (Src Window), `dwin` (Dst Window), `stcpb` (Src TCP Sequence Num), `dtcpb` (Dst TCP Sequence Num), `tcprtt` (TCP RTT), `synack` (SYN-ACK Delay), `ackdat` (ACK Delay), `smean` (Mean Src Pkt Size), `dmean` (Mean Dst Pkt Size).
* **Content & Host Features**: `trans_depth` (HTTP Depth), `response_body_len` (HTTP Body Len), `ct_srv_src` (Service-Src Count), `ct_state_ttl` (State-TTL Count), `ct_dst_ltm` (Dst LTM Count), `ct_src_dport_ltm` (Src-Dport Count), `ct_dst_sport_ltm` (Dst-Sport Count), `ct_dst_src_ltm` (Dst-Src Count), `is_ftp_login` (FTP Login Flag), `ct_ftp_cmd` (FTP Cmd Count), `ct_flw_http_mthd` (HTTP Method Count), `ct_src_ltm` (Src LTM Count), `ct_srv_dst` (Service-Dst Count), `is_sm_ips_ports` (Same IP/Port Flag).

---

## Repository Structure

```
XAI-ZERODAY-IDS/
├── data/                         # UNSW-NB15 raw datasets
│   ├── UNSW_NB15_testing-set.csv
│   └── UNSW_NB15_training-set.csv
├── images/                       # Model explainability diagrams
├── notebooks/                    # Exploratory Data Analysis & Model Weights
│   ├── models/                   # Serialized model artifacts (.pkl)
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── XAI_Anlaysis.ipynb
├── results/                      # Evaluation telemetry & confusion matrix plots
├── src/                          # Core Python Engine Modules
│   ├── __init__.py
│   ├── config.py                 # Paths & feature definitions
│   ├── explain.py                # SHAP & LIME XAI engines
│   ├── generator.py              # Dynamic synthetic traffic generator
│   ├── hybrid.py                 # Hybrid consensus decision engine
│   ├── loader.py                 # Model & preprocessor artifact loader
│   ├── model.py                  # Model training logic
│   └── preprocessing.py          # Data cleaning & scaling pipeline
├── ui/                           # Streamlit Web Dashboard
│   └── app.py                    # SOC application interface
├── train_models.py               # Model training entry point script
├── sample_network_data.csv       # Sample batch network logs
├── requirements.txt              # Dependency specifications
├── LICENSE                       # MIT License file
└── README.md                     # Project documentation
```

---

## Step-by-Step Installation & Execution Guide

### Prerequisites
* **Python 3.10+** installed.
* **Git** installed.

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Yojasree1905/XAI-ZERODAY-IDS.git
cd XAI-ZERODAY-IDS
```

---

### Step 2: Create & Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: (Optional) Retrain Models
To train model weights on the UNSW-NB15 dataset:

```bash
python train_models.py
```

---

### Step 5: Launch the Streamlit Web Application
Run the interactive application dashboard:

```bash
streamlit run ui/app.py
```

After launching, open your browser and navigate to:
`http://localhost:8501`

---

## 📄 License & Author

Developed for real-time explainable cybersecurity research and multi-model intrusion detection optimization.

* **Developer**: Yojasree (`Yojasree1905`)
* **License**: MIT License — open for academic & commercial reuse. See [LICENSE](LICENSE) for details.
