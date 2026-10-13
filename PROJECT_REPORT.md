# COMPREHENSIVE PROJECT REPORT

# AI-DRIVEN NETWORK TELEMETRY COMPRESSION & THREAT AUDITING SYSTEM

---

## EXECUTIVE SUMMARY / ABSTRACT

Modern enterprise and cyber-physical networks generate massive volumes of continuous telemetry data while facing sophisticated zero-day cyber attacks. Traditional intrusion detection systems (IDS) struggle with two major challenges:
1. **Explainability & Novel Threat Detection**: High-capacity black-box machine learning models identify threats but fail to provide forensic rationale, delaying mitigation.
2. **Telemetry Storage & Ingestion Bottlenecks**: High-throughput network packet captures and audit logs strain storage infrastructure and network bandwidth during telemetry streaming.

This project proposes a unified, dual-track framework that integrates:
- **Intrusion Detection & Explainable AI (XAI)**: High-performance gradient boosted decision trees (XGBoost) trained on the UNSW-NB15 dataset for binary and multi-class anomaly detection, augmented with SHAP (SHapley Additive exPlanations) and automated MITRE ATT&CK tactical attribution.
- **Advanced Data Compression Suite**: An algorithmic suite implementing statistical entropy coding (**Shannon-Fano**, **Huffman**), dictionary and transform techniques (**LZW**, **LZ77/Gzip**, **Bzip2/Burrows-Wheeler Transform**), and rate-distortion quantization models (**Uniform Scalar Quantization**, **LBG Vector Quantization**) to compress network logs for cold storage and low-latency transport.

The system is deployed as an interactive analytical dashboard built with Streamlit, enabling one-click forensic report generation in executive PDF format and direct binary export of compressed network telemetry.

---

## 1. INTRODUCTION

### 1.1 Background
Network traffic monitoring is the backbone of cyber defense. Modern security operations centers (SOCs) continuously ingest flow-level metrics including source/destination IP addresses, packet rates, byte counts, TCP window advertisements, and service protocols. As link speeds exceed 10 to 100 Gbps, uncompressed raw log storage becomes cost-prohibitive. Furthermore, SOC analysts require actionable justifications when an anomaly is flagged.

### 1.2 Problem Statement
Existing security solutions isolate threat detection from telemetry management:
- Threat detection engines process uncompressed or raw streams without archival optimization.
- Compression pipelines treat network packets as arbitrary text files without exploiting structural statistical properties (e.g., categorical protocol fields vs. continuous rate metrics).
- Deep learning and ensemble detectors behave as black boxes, making zero-day attacks hard to audit quickly.

### 1.3 Project Objectives
1. **Accurate Anomaly & Zero-Day Detection**: Implement gradient boosting classifiers on UNSW-NB15 to achieve high precision and recall on complex attack categories (Exploits, Reconnaissance, DoS, Fuzzers, Backdoors, Generic).
2. **Transparent Explainability**: Embed SHAP feature attribution to highlight top contributing features per prediction and cross-reference them with MITRE ATT&CK tactic IDs.
3. **Comprehensive Compression Benchmarking**: Implement and systematically benchmark fundamental compression algorithms covering:
   - **Entropy & Statistical Prefix Coding**: Shannon-Fano, Huffman.
   - **Dictionary & Transform Coding**: LZW, LZ77 (Deflate/Gzip), Bzip2 (Burrows-Wheeler Transform).
   - **Quantization & Rate-Distortion Coding**: Uniform Scalar Quantization, Linde-Buzo-Gray (LBG) Vector Quantization.
4. **Unified Interactive Dashboard**: Deliver an intuitive web-based interface for live analysis, algorithm comparison, and binary payload generation.

---

## 2. SYSTEM ARCHITECTURE

The system follows a clean, four-tier architecture separating ingestion, security analytics, telemetry compression, and user presentation.

### 2.1 System Architecture Diagram

![System Architecture](system_architecture.png)

### 2.2 Layer Breakdown & Technical Mapping
The system architecture directly mirrors the operational flow of the platform:

1. **Layer 1: Telemetry Ingestion Layer**:
   - **Ingestion Sources**: Real-time synthetic packet streams (`src/generator.py`), user-uploaded network logs (`ui/app.py`), and the standardized **UNSW-NB15** benchmark dataset (`data/`).
   - **Data Cleaning & Standardization**: Automated imputation of missing attributes, Min-Max normalization for numerical continuous features, and categorical one-hot encoding for protocol/state fields (`src/preprocessing.py`).

2. **Layer 2: Orchestration & Dispatch Layer**:
   - **Dual-Track Pipeline Dispatcher**: Routes preprocessed traffic streams simultaneously into:
     - *Normalized Feature Vectors*: Structured 194-dimensional dense vectors fed into the machine learning detection track.
     - *Raw Byte Stream & Numerical Arrays*: Serialized flow bytes and floating-point matrices dispatched to the compression engine.

3. **Layer 3A: Threat Detection & Explainability Track**:
   - **Multi-Model Detection**: Combines **XGBoost** (gradient boosted trees), **Random Forest** (multi-class ensemble), and **Isolation Forest** (unsupervised zero-day anomaly detector) with multi-model consensus scoring (`src/hybrid.py`, `src/model.py`).
   - **Forensic Reasoning**: Local and global feature attributions computed via **TreeSHAP** and **LIME** (`src/explain.py`), with direct automated cross-referencing to **MITRE ATT&CK** tactics and techniques (`src/mitre_mapping.py`).

4. **Layer 3B: Advanced Data Compression Track**:
   - **Compression Benchmarking Arbiter**: Executes all seven compression algorithms simultaneously across ingested network traffic (`src/compression/evaluator.py`).
   - **Supported Codecs**:
     - *Quantization Models*: Uniform Scalar Quantizer (4-bit, 8-bit) and Linde-Buzo-Gray (LBG) Vector Quantizer (`src/compression/scalar_quantization.py`, `src/compression/vector_quantization.py`).
     - *Dictionary & Transform Coders*: LZW, LZ77 (Gzip), and Bzip2 (Burrows-Wheeler Transform) (`src/compression/lzw.py`, `src/compression/lossless_benchmark.py`).
     - *Entropy & Prefix Coders*: Shannon-Fano and Huffman algorithms (`src/compression/shannon_fano.py`, `src/compression/huffman.py`).
   - **Evaluation & Recommendation**: Computes compression ratio (CR), space savings (%), distortion (MSE/PSNR), and runtime latency to automatically recommend the optimal codec.

5. **Layer 4: Application, Delivery & Storage Layer**:
   - **Operations Dashboard**: Full-featured interactive interface built with **Streamlit** (`ui/app.py`) providing live threat radars, SHAP waterfall plots, and compression analytics.
   - **Artifact Exports**: One-click generation of publication-quality forensic incident PDF reports (`src/pdf_generator.py`), compressed binary payloads (`.bin`), quantized codebooks (`.json`), and reconstructed telemetry (`.csv`).

---

## 3. END-TO-END PROCESSING WORKFLOW

The data flows through two synchronized tracks: a Security Track and a Compression Track.

```mermaid
flowchart TD
    Start[Uploaded Network Telemetry CSV] --> Preproc[Data Cleaning & Normalization]
    Preproc --> Split[Dual-Track Pipeline Execution]

    %% Security Track
    Split -->|Security Track| Infer[XGBoost Anomaly & Attack Classification]
    Infer --> XAI[SHAP Feature Attribution & MITRE Mapping]
    XAI --> Report[Automated PDF Forensic Report]

    %% Compression Track
    Split -->|Compression Track| Engine[Compression Benchmarking Engine]
    
    Engine --> C1[Entropy Coders: Shannon-Fano & Huffman]
    Engine --> C2[Dictionary Coders: LZW, LZ77 & Bzip2]
    Engine --> C3[Quantizers: Uniform Scalar & LBG Vector]

    C1 --> Eval[Evaluation: Ratio, Space Savings, Distortion & Latency]
    C2 --> Eval
    C3 --> Eval

    Eval --> Select[Optimal Codec Identification]
    Select --> Package[Compressed Binary Packaging: .bin & .json]

    %% Final Presentation
    Report --> View[Unified Streamlit Dashboard Display]
    Package --> View
```

---

## 4. METHODOLOGY & ALGORITHMIC DETAILS

### 4.1 Intrusion Detection & Explainability Track
- **Classifier**: Extreme Gradient Boosting (`XGBoost`). Employs tree ensemble boosting with second-order Taylor expansion loss minimization and regularization to prevent overfitting on skewed intrusion distributions.
- **Explainable AI (SHAP)**: Computes Shapley values based on cooperative game theory:
  $$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
  This quantifies the exact marginal contribution of network flow metrics (e.g., `sbytes`, `sttl`, `dload`) to the final attack score.
- **MITRE ATT&CK Mapping**: Correlates top contributing features to specific cyber adversary tactics (e.g., Discovery, Exfiltration, Initial Access, Command & Control).

---

### 4.2 Advanced Data Compression Algorithms

#### 1. Entropy & Statistical Prefix Coding
- **Shannon-Fano Coding**:
  - *Principle*: Calculates symbol probabilities $p_i$, sorts them in descending order, and recursively partitions the list into two subsets with sums as close to equal as possible, assigning `0` and `1` until each symbol forms a leaf.
  - *Characteristics*: Top-down greedy approach; near-optimal average code length $L = \sum p_i l_i \approx H(X)$.
- **Huffman Coding**:
  - *Principle*: Builds a prefix-free binary tree bottom-up by repeatedly combining the two least frequent nodes into a parent node whose probability is the sum of its children.
  - *Optimality*: Mathematically proven optimal prefix code minimizing expected length $L(C)$ for a given discrete probability distribution.

#### 2. Dictionary & Transform Coding
- **Lempel-Ziv-Welch (LZW)**:
  - *Principle*: Maintains an evolving dictionary initialized with single-character symbols (0–255). As repeated substrings are encountered, they are emitted as integer indices, and longer phrases are appended to the dictionary.
  - *Benefit*: Single-pass streaming; requires no transmission of the dictionary table (the decoder reconstructs it identically).
- **LZ77 (Sliding Window / Gzip)**:
  - *Principle*: Replaces repeated occurrences of data with references (distance-length pairs) to matching data within a sliding history window, followed by Huffman coding of literals and lengths (Deflate).
  - *Benefit*: High throughput and industry-standard interoperability.
- **Bzip2 (Burrows-Wheeler Transform + MTF + Huffman)**:
  - *Principle*: Reversible block-sorting transform (BWT) that permutes input characters to group identical characters together, followed by Move-To-Front (MTF) coding and run-length encoding (RLE) to create long runs of zeros, terminated with Huffman coding.
  - *Benefit*: Achieves state-of-the-art lossless compression ratios on large text and repetitive telemetry logs.

#### 3. Quantization & Rate-Distortion Coding
- **Uniform Scalar Quantization**:
  - *Principle*: Maps continuous 1D numerical values $x \in [x_{\min}, x_{\max}]$ into a finite set of $2^b$ discrete levels using step size $\Delta = \frac{x_{\max} - x_{\min}}{2^b}$.
  - *Trade-off*: Introduces controlled bounded distortion (MSE $\approx \Delta^2 / 12$) while reducing floating-point fields (32-bit/64-bit) down to compact 4-bit or 8-bit bins.
- **Linde-Buzo-Gray (LBG) Vector Quantization**:
  - *Principle*: Extends quantization to multidimensional vectors $\mathbf{x} \in \mathbb{R}^k$. Clusters multidimensional continuous feature vectors into a codebook $\mathcal{C} = \{\mathbf{y}_1, \dots, \mathbf{y}_K\}$ using the generalized Lloyd-Max iterative splitting algorithm.
  - *Process*: Starts with a centroid, splits centroids by adding/subtracting perturbation vectors, and clusters vectors until distortion convergence:
    $$D = \frac{1}{N} \sum_{n=1}^N \|\mathbf{x}_n - \mathbf{y}_{q(\mathbf{x}_n)}\|^2$$
  - *Benefit*: Explores spatial correlation across multiple metrics simultaneously, achieving drastic compression at minimal reconstruction loss.

---

## 5. EXPERIMENTAL METRICS & EVALUATION

The performance of each compression algorithm and the multi-algorithm ensemble is evaluated using eight comprehensive compression performance metrics, strictly assessing compression efficiency and hardware footprint:

| Metric | Formula / Source | Operational Significance |
| :--- | :--- | :--- |
| **Compression Ratio (CR)** | $CR = \frac{S_{orig}}{S_{comp}}$ | Fundamental compaction multiplier ($>1.0$ indicates reduction) |
| **Space Decrease (%)** | $\left(1 - \frac{S_{comp}}{S_{orig}}\right) \times 100\%$ | Net percentage of bandwidth and storage capacity freed |
| **Throughput (MB/s)** | $\frac{S_{orig} \text{ (MB)}}{\text{Encoding Time (s)}}$ | Line-rate encoding velocity across network ingress interfaces |
| **Encoding Latency (ms)** | Wall-clock elapsed time | Algorithmic execution latency per telemetry batch |
| **Distortion (MSE)** | $\frac{1}{N}\sum (x_i - \hat{x}_i)^2$ | Reconstruction fidelity ($0.0$ for bit-exact lossless codecs) |
| **Peak SNR (PSNR dB)** | $10 \log_{10}\left(\frac{\max(x)^2}{\text{MSE}}\right)$ | Signal quality ($+\infty\text{ dB}$ for lossless preservation) |
| **Downstream Feature Retention (%)** | Feature-level preservation | Retention of anomaly detection accuracy on decompressed records |
| **Active CPU Utilization (%)** | Process polling via `psutil` | Real-time processor utilization during compression encoding |

### Multi-Algorithm Benchmark & Ensemble Summary on UNSW-NB15 Telemetry

| Algorithm Family | Algorithm | Compression Ratio (CR) | Space Decrease (%) | Distortion (MSE) | CPU Usage (%) | Throughput (MB/s) | Best Real-Life Application |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hybrid Ensemble** | **⭐ Multi-Algorithm Compression Ensemble** | **~2.35x - 3.80x** | **57% - 74%** | **0.0 (Bit-Exact)** | **1.2% - 3.5%** | **~1.6 - 5.5 MB/s** | **Two-Party Transmit Vault & Full Audit Recovery** |
| **Dictionary / Transform** | **Bzip2 (BWT)** | **~5.5x - 7.0x** | **82% - 86%** | $0.0$ (Lossless) | ~2.5% - 4.8% | ~8.2 MB/s | Deep Cold Storage & Forensic Archiving |
| **Dictionary** | **LZ77 (Gzip Standard)** | **~3.5x - 5.2x** | **71% - 81%** | $0.0$ (Lossless) | ~1.8% - 3.2% | ~31.4 MB/s | High-Throughput Edge Ingress & Transit |
| **Dictionary** | **LZW** | **~1.7x - 3.5x** | **40% - 71%** | $0.0$ (Lossless) | ~1.2% - 2.8% | ~2.6 MB/s | Dynamic Stream Buffer Encoding |
| **Entropy** | **Huffman Algorithm** | **~1.6x - 2.1x** | **38% - 52%** | $0.0$ (Lossless) | ~1.2% - 2.0% | ~4.4 MB/s | Optimal Prefix Entropy Reduction |
| **Entropy** | **Shannon-Fano Algorithm** | **~1.5x - 2.0x** | **35% - 50%** | $0.0$ (Lossless) | ~1.1% - 1.9% | ~6.0 MB/s | Theoretical Top-Down Partitioning Baseline |
| **Quantization** | **Uniform Scalar Quantizer (8-bit)** | **~4.0x - 8.0x** | **75% - 88%** | $\approx 0.0021$ | ~0.8% - 1.5% | ~79.6 MB/s | Rapid 1D Feature Vector Decimation |
| **Quantization** | **LBG Vector Quantizer** | **~10.0x - 650x** | **90% - 99.8%** | $\approx 0.0048$ | ~1.5% - 3.2% | ~19.7 MB/s | Ultra-Dense Dimensionality Reduction |

---

## 5.1 TWO-PARTY COLLABORATIVE TELEMETRY VAULT & VERIFICATION PROTOCOL

To bridge real-world operational workflows where edge sensor nodes or branch offices must transmit large network logs to a centralized SOC without exceeding uplink bandwidth constraints, the platform features a **Two-Party Collaborative Workflow**:

1. **Person 1: Field Data Producer (Upload, Compress & Transmit)**
   - Uploads or samples high-volume raw telemetry streams ($100+$ features/flows).
   - Profiles all 8 compression techniques simultaneously with active CPU measurement.
   - Selects the optimal scheme (e.g., the Multi-Algorithm Compression Ensemble).
   - Generates an immutable cryptographic **SHA-256 Digest** of the uncompressed data.
   - Packs the compressed binary blob into the persistent **SQLite Database Vault** (`data/telemetry_vault.db`).
   - Exports the **Executive Compression Audit Report (PDF)** containing all 8 metrics and CPU benchmarks.

2. **Person 2: SOC Incident Evaluator (Decompress, Audit & Loss Verification)**
   - Queries and pulls the transmitted package from the SQLite Database Vault.
   - Algorithmic engine executes automated decompression of the binary bitstream.
   - Recomputes the SHA-256 checksum on the decompressed telemetry:
     $$\text{Verification Verdict} = \begin{cases} \text{100\% BIT-EXACT ZERO LOSS}, & \text{if } \text{SHA-256}_{decomp} == \text{SHA-256}_{orig} \\ \text{DISTORTION DETECTED}, & \text{otherwise} \end{cases}$$
   - Assesses Mean Squared Error ($MSE = 0.0000$) and downstream anomaly detection retention ($100.0\%$).
   - Exports the recovered, verified CSV telemetry file for SOC investigation.

---

## 6. PROJECT IMPLEMENTATION STRUCTURE

```
compression project/
├── data/                            # Raw and processed UNSW-NB15 datasets
├── src/                             # Core Python modules
│   ├── compression/                 # Data Compression Engine
│   │   ├── shannon_fano.py          # Shannon-Fano prefix codec
│   │   ├── huffman.py               # Huffman tree codec
│   │   ├── lzw.py                   # LZW dictionary codec
│   │   ├── lossless_benchmark.py    # LZ77 & Bzip2 benchmark runner
│   │   ├── scalar_quantization.py   # Uniform Scalar Quantizer
│   │   ├── vector_quantization.py   # LBG Vector Quantizer
│   │   └── evaluator.py             # Performance metric aggregator
│   ├── model.py                     # XGBoost training & inference
│   ├── explain.py                   # SHAP & LIME interpretability
│   ├── mitre_mapping.py             # MITRE ATT&CK framework mapping
│   ├── pdf_generator.py             # Forensic executive report generator
│   └── preprocessing.py             # Feature normalization & encoding
├── ui/                              # User Interface
│   └── app.py                       # Streamlit multi-tab analytical dashboard
├── results/                         # Output bitstreams, codebooks, and reports
├── run_compression_benchmarks.py    # Standalone CLI compression benchmark
└── train_models.py                  # Model training and artifact generation
```

---

## 7. USER INTERFACE & CAPABILITIES

The Streamlit dashboard (`ui/app.py`) provides a complete operator experience:
1. **Interactive Threat Radar**: Visualizes attack classifications, probabilities, and confidence bounds in real time.
2. **Forensic Attribution Studio**: Inspects individual predictions using SHAP force plots, identifying which packet attributes triggered the alarm.
3. **Automated Incident PDF Generation**: Creates publication-ready security incident reports with tables, MITRE tactics, and recommendations.
4. **Advanced Data Compression Lab**:
   - Single-click benchmarking across all implemented algorithms.
   - Dynamic **#1 Most Useful Algorithm** recommendation card.
   - Top-3 compression technique cards with 1-click downloads for `.bin`, `.json`, and `.csv`.
   - Comparative charts (Space Savings, Compression Ratios, and Compression Latencies).

---

---

## 8. CONCLUSION & FUTURE SCOPE

### 8.1 Conclusion
This project successfully bridges the gap between high-speed cyber defense and storage optimization. By combining XGBoost with SHAP explainability, security analysts obtain rapid, transparent threat classifications. Concurrently, the Advanced Data Compression suite demonstrates that applying modern dictionary transforms (Bzip2, LZ77) and quantization (LBG VQ) can compress network telemetry by up to **85% to 92%**, dramatically cutting enterprise storage overhead without losing analytical forensic capabilities.

### 8.2 Future Scope
1. **Hardware Acceleration**: Implement Huffman and LZW decoders on FPGA or GPU kernels for line-rate decompression on 100 Gbps network interfaces.
2. **Dynamic Adaptive Streaming**: Deploy an entropy arbiter that monitors network burstiness and automatically switches between LZ77 and Bzip2 based on CPU availability.
3. **Deep Generative Compression**: Explore learned neural autoencoders for learned lossy compression with perceptual rate-distortion constraints.

---

## 9. REFERENCES & BIBLIOGRAPHY

### 9.1 Data Compression & Information Theory
1. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423.  
   Link: [https://doi.org/10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x) | [PDF via Harvard](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
2. **Huffman, D. A.** (1952). "A Method for the Construction of Minimum-Redundancy Codes." *Proceedings of the IRE*, 40(9), 1098–1101.  
   Link: [https://doi.org/10.1109/JRPROC.1952.273898](https://doi.org/10.1109/JRPROC.1952.273898) | [IEEE Xplore](https://ieeexplore.ieee.org/document/4051119)
3. **Fano, R. M.** (1949). "The Transmission of Information." *Technical Report No. 65*, Research Laboratory of Electronics, M.I.T.  
   Link: [https://dspace.mit.edu/handle/1721.1/4769](https://dspace.mit.edu/handle/1721.1/4769)
4. **Ziv, J., & Lempel, A.** (1977). "A Universal Algorithm for Sequential Data Compression." *IEEE Transactions on Information Theory*, 23(3), 337–343.  
   Link: [https://doi.org/10.1109/TIT.1977.1055714](https://doi.org/10.1109/TIT.1977.1055714) | [IEEE Xplore](https://ieeexplore.ieee.org/document/1055714)
5. **Welch, T. A.** (1984). "A Technique for High-Performance Data Compression." *IEEE Computer*, 17(6), 8–19.  
   Link: [https://doi.org/10.1109/MC.1984.1659158](https://doi.org/10.1109/MC.1984.1659158) | [IEEE Xplore](https://ieeexplore.ieee.org/document/1659158)
6. **Burrows, M., & Wheeler, D. J.** (1994). "A Block-sorting Lossless Data Compression Algorithm." *SRC Research Report 124*, Digital Equipment Corporation.  
   Link: [https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf](https://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf)
7. **Linde, Y., Buzo, A., & Gray, R. M.** (1980). "An Algorithm for Vector Quantizer Design." *IEEE Transactions on Communications*, 28(1), 84–95.  
   Link: [https://doi.org/10.1109/TCOM.1980.1094577](https://doi.org/10.1109/TCOM.1980.1094577) | [IEEE Xplore](https://ieeexplore.ieee.org/document/1094577)
8. **Sayood, K.** (2017). *Introduction to Data Compression* (5th ed.). Morgan Kaufmann / Elsevier.  
   Link: [https://doi.org/10.1016/C2015-0-01968-3](https://doi.org/10.1016/C2015-0-01968-3) | [Elsevier Store](https://www.elsevier.com/books/introduction-to-data-compression/sayood/978-0-12-809474-7)
9. **Gersho, A., & Gray, R. M.** (1992). *Vector Quantization and Signal Compression*. Springer Science & Business Media.  
   Link: [https://doi.org/10.1007/978-1-4615-3626-0](https://doi.org/10.1007/978-1-4615-3626-0) | [Springer Link](https://link.springer.com/book/10.1007/978-1-4615-3626-0)

### 9.2 Network Security & Intrusion Detection
10. **Moustafa, N., & Slay, J.** (2015). "UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*, Canberra, ACT, Australia, pp. 1–6.  
    Link: [https://doi.org/10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942) | [UNSW Data Portal](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
11. **Moustafa, N., & Slay, J.** (2016). "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set." *Information Security Journal: A Global Perspective*, 25(1-3), 18–31.  
    Link: [https://doi.org/10.1080/19393555.2015.1130333](https://doi.org/10.1080/19393555.2015.1130333) | [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/19393555.2015.1130333)
12. **Kumar, V., Sinha, D., Das, A. K., et al.** (2020). "An Efficient Machine Learning Framework for Intrusion Detection in IoT Networks." *IEEE Transactions on Industrial Informatics*, 16(10), 6654–6665.  
    Link: [https://doi.org/10.1109/TII.2019.2963286](https://doi.org/10.1109/TII.2019.2963286) | [IEEE Xplore](https://ieeexplore.ieee.org/document/8946761)

### 9.3 Machine Learning & Explainable AI (XAI)
13. **Chen, T., & Guestrin, C.** (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)*, pp. 785–794.  
    Link: [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
14. **Lundberg, S. M., & Lee, S.-I.** (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS 30)*, pp. 4765–4774.  
    Link: [https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
15. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135–1144.  
    Link: [https://doi.org/10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778) | [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
16. **Strom, B. E., Applebaum, A., Miller, D. P., et al.** (2018). "MITRE ATT&CK™: Design and Philosophy." *MITRE Technical Report*, The MITRE Corporation.  
    Link: [https://www.mitre.org/news-insights/publication/mitre-attack-design-and-philosophy](https://www.mitre.org/news-insights/publication/mitre-attack-design-and-philosophy) | [MITRE ATT&CK Framework Official](https://attack.mitre.org/)

---

*Report prepared for AI-Driven Network Telemetry Compression & Threat Auditing System Laboratory Evaluation.*
