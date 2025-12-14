# graph-based-ids-5g
Graph Neural Network-based Intrusion Detection for 5G Networks

# Graph-based Intrusion Detection System (IDS) for 5G Networks

This repository contains an end-to-end **graph-based intrusion detection pipeline** for 5G/SDN environments using **Graph Neural Networks (GNNs)**. The system is built and executed on **Kaggle**, leveraging the **ITU-ML5G-PS-006 dataset** and implementing **GraphSAGE** and **Graph Attention Network (GAT)** models.

---

## Project Overview

The goal of this project is to detect network intrusions in 5G traffic by:

* Preprocessing large-scale flow-level data
* Constructing **flow–host bipartite graphs**
* Training scalable **GraphSAGE** and **GAT** models
* Handling class imbalance using synthetic anomaly generation
* Performing comprehensive evaluation with accuracy, F1-score, and confusion matrices

This work is suitable for **research, FYPs, and experimental IDS pipelines**.

---

## Dataset

* **Dataset Name:** ITU-ML5G-PS-006
* **Source:** Zenodo
* **Link:** [https://zenodo.org/records/13939009](https://zenodo.org/records/13939009)


The dataset contains millions of network flow records designed for **5G/SDN intrusion detection**.

**Citation:**

> Chatzimiltis, S. (2024). *ITU-ML5G-PS-006: Intrusion and Vulnerability Detection in Software-Defined Networks (SDN)* [Dataset]. Zenodo.  
> https://doi.org/10.5281/zenodo.13939009

The dataset is publicly available on Zenodo and is **not redistributed** in this repository due to size constraints.


---

## Project Structure

```
.
├── preprocessing.py          # Data preprocessing and cleaning
├── graph_construction.py     # Graph building utilities
├── graphs/                   # Saved PyTorch Geometric graph objects (.pt)
├── models/
│   ├── graphsage          # Large-scale GraphSAGE model
│   └── gat                # Robust GAT model
├── results/
│   ├── confusion_matrices/   # Saved confusion matrices
│   └── training_plots/       # Loss / Accuracy / F1 plots
├── README.md                 # Project documentation
```

*(In Kaggle, all outputs are saved under `/kaggle/working/`)*

---

## Requirements

This project is designed to run on **Kaggle Notebooks**.

### Python Version

* Python 3.8+

### Libraries

* torch
* torch_geometric
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## Setup Instructions (Kaggle)

### Create a Kaggle Notebook

* Go to Kaggle → Notebooks → New Notebook
* Enable **Internet** access

### Add Dataset

* Add the dataset manually or via Kaggle Dataset upload
* Dataset path used in code:

```
/kaggle/input/itu-ml5g-ps-006/Train_ULAK.csv
```


### Install Required Packages

Run the following in a notebook cell:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install torch_geometric
```

---

## Step 1: Data Preprocessing

The preprocessing pipeline:

* Cleans column names
* Auto-detects the label column
* Removes duplicates
* Handles missing and infinite values
* One-hot encodes categorical features (excluding IP/MAC)
* Normalizes numeric features

Output:

```
/kaggle/working/itu_ml5g_cleaned2.csv
```

Run:

```python
python preprocess_itu_ml5g.py
```

---

## Step 2: Graph Construction

The system constructs a **bipartite graph**:

* **Flow nodes:** Network flows
* **Host nodes:** Synthetic host representations

Each flow node is connected to multiple hosts to improve graph connectivity.

Multiple graph scales are supported:

| Mode    | Flows | Hosts |
| ------- | ----- | ----- |
| Minimal | 200   | 10    |
| Small   | 500   | 25    |
| Medium  | 1,000 | 50    |
| Large   | 5,000 | 200   |

Graphs are saved as:

```
/kaggle/working/itu_large_graph.pt
```

---

## Step 3: Large-Scale GraphSAGE IDS

### Model Architecture

* 4-layer GraphSAGE
* Hidden dimensions: 128 → 64 → 32
* Batch normalization
* Dropout regularization

### Training Strategy

* Class-weighted loss
* Synthetic anomaly balancing
* Learning rate scheduling
* Early stopping

### Train / Validation / Test Split

* 70% / 15% / 15%
* Stratified by class

### Output

* Best model: `best_large_graphsage.pth`
* Final model: `large_graphsage_ids.pth`
* Training curves and confusion matrix

---

## Step 4: Graph Attention Network (GAT)

### Model Features

* Multi-head attention
* Batch normalization
* Dropout regularization
* Robust to noisy connections

### Architecture

* 3 GAT layers + linear classifier
* Heads: 4 → 4 → 2

### Outputs

* `best_gat.pth`
* Confusion matrix
* Classification report

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score (macro & weighted)
* Confusion Matrix

Evaluation is performed **only on flow nodes** (host nodes are masked).

---

## Running the Full Pipeline

On Kaggle, simply run all notebook cells in order:

1. Install dependencies
2. Preprocess dataset
3. Build graphs
4. Train GraphSAGE
5. Evaluate GraphSAGE
6. Train GAT
7. Evaluate GAT

---

## Results 

* Balanced classification performance
* Improved anomaly recall using graph context
* Stable convergence on large graphs (5k+ flows)

Exact results depend on random seeds and sampling.

---

## Author

**Yusra Nadeem,**
**Amna Akram and**
**Saba Bano**
---


## License

This project is intended for **academic and research use** only.



