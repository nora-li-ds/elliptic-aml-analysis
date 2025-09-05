# Chasing Dirty Coins: An AML Graph Project  (•̀ᴗ•́)و

Can we **catch money launderers** using **machine learning + network science**?

---

## Why this project?
Fraudsters are smart. Data scientists must be smarter.  
This project uses the **Elliptic Bitcoin Dataset** to:

- Visualize how dirty money flows through networks  
- Build graph features (centrality, embeddings, temporal splits)  
- Train models to classify suspicious transactions  
- Explore GNNs & anomaly detection for **unknown-class fraud**  
- This project is part of my portfolio on FinCrime Data Science, connecting academic research with industry AML use cases.

## Quick Facts 
 **Dataset**: 203k transactions, 234k edges, 166 features  
 **Challenge**: Only ~3% illicit → heavy imbalance  
 **Graph**: 49 connected components, focus on LCC  
 **Feature Engineering**: Dropped 46 highly correlated columns  
 **Split**: Time-based (70/10/20) to avoid data leakage  

---

## 📊 Results (so far)

Here are some key insights and visualizations from the first two notebooks:

### 1. Class Distribution
Illicit transactions are **heavily underrepresented** (~3%), while most transactions are unknown.  
This highlights the **imbalance challenge** typical in fraud detection.  

![Class Distribution](reports/figures/class_distribution.png)

---

### 2. Transaction Network Structure
We analyze the graph of 203k nodes and 234k edges:  

| Top-100 Hubs (Graph Subnetwork) | Degree vs PageRank (log-log) |
|---------------------------------|-------------------------------|
| ![Top-100 Hubs](reports/figures/top100_hubs.png) | ![Degree vs PageRank](reports/figures/scatter_degree_pagerank.png) |

**What these show:**
- **Top-100 Hubs:** highlights the most connected transactions and their 1-hop neighborhoods, showing how laundering activity often clusters around a few "super-nodes".  
- **Degree vs PageRank:** strong positive correlation; high-degree nodes tend to dominate PageRank influence, but outliers may indicate hidden intermediaries.  

👉 These visualizations confirm the **scale-free nature** of the transaction graph and motivate the need for advanced graph-based models.
---

### 3. Feature Correlation Deduplication
We removed **46 highly correlated features** (threshold > 0.98), reducing redundancy and improving model stability.  

| Before Deduplication | After Deduplication |
|----------------------|---------------------|
| ![Before](reports/figures/corr_heatmap_before.png) | ![After](reports/figures/corr_heatmap_after.png) |

---

### 4. Time-based Split (70/10/20)
We split the dataset by **time steps** to avoid data leakage:  

- **Train:** 29,936 licit / 2,659 illicit  
- **Valid:** 4,096 licit / 560 illicit  
- **Test:** 7,987 licit / 1,326 illicit  

This ensures future transactions are never used to predict the past.  

---

> Note: Due to file size limits, raw data files are **not included** in this repository.  
> Please download `elliptic_txs_features.csv` from the original Kaggle dataset and place it in `data/raw/`.

## Data Setup

To run this project, you need to manually download the dataset:

1. Go to [Kaggle: Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  
2. Download the `.zip` file and extract it  
3. Place the extracted `.csv` files into:

```
data/raw/
```

---

## 🧠 What's inside?

```
elliptic-aml-analysis/
├── data/
│   ├── elliptic_bitcoin_dataset/        <- CSVs (features, labels, edges)
│   └── raw/                             <- Original downloaded zip
├── notebooks/
│   ├── 01_exploration.ipynb             <- Data deep dive & network intuition
│   ├── 02_preprocessing.ipynb           <- Feature crafting & graph building
│   ├── 03_modeling.ipynb                <- ML pipelines & classification
│   ├── 04_graph_models.ipynb            <- Graph embeddings & GNN experiments
│   └── 05_anomaly_detection.ipynb       <- Semi-supervised & outlier detection
├── scripts/
│   ├── utils.py                         <- Custom helpers & metrics
│   ├── preprocessing.py                 <- Cleaning, feature selection, splitting
│   ├── models.py                        <- Training pipelines & evaluation
│   └── visualization.py                 <- Plots (heatmaps, ROC, confusion matrix)
├── reports/
│   ├── figures/                         <- Key visualizations
│   └── summary.md                       <- Reflective notes & insights
├── requirements.txt                     <- Dependencies
├── .gitignore                           <- Keeps repo clean
└── README.md                            <- You are here

```

---

## Notebook Overview

### **01_exploration.ipynb** – *Data Deep Dive & Network Intuition*
- Load dataset, inspect distributions, and explore temporal structure  
- Build a **transaction graph** with NetworkX  
- Calculate basic graph statistics (degree, connected components)  
- First visualizations of illicit vs licit transaction patterns  

---

### **02_preprocessing.ipynb** – *Feature Crafting & Graph Building*
- Clean raw CSVs and handle missing values  
- Merge features, labels, and edge lists  
- Engineer time-based and aggregated graph features  
- Output processed dataset for modeling  

---

### **03_modeling.ipynb** – *ML Pipelines & Classification*
- Train baseline models: Logistic Regression, Random Forest, XGBoost  
- Evaluate with Precision, Recall, F1, AUC  
- Feature importance analysis for explainability  
- Build reproducible sklearn pipeline  

---

### **04_graph_models.ipynb** – *Graph-based ML & GNN Experiments* *(planned)*
- Generate graph embeddings (Node2Vec, DeepWalk)  
- Hybrid modeling: embeddings + tabular features  
- Implement **GNNs** (GraphSAGE / GCN) in PyTorch Geometric or DGL  
- Compare ML-only vs graph-enhanced models  
- Visualize embeddings with t-SNE  

---

### **05_anomaly_detection.ipynb** – *Unknown-class & Rare-pattern Detection* *(planned)*
- Semi-supervised learning (Label Propagation, Label Spreading)  
- Outlier detection on embeddings (Isolation Forest, LOF)  
- Time-aware anomaly scoring to detect evolving laundering patterns  
- Identify high-risk **unlabeled** transactions  

---

## About the Dataset

- **Nodes:** 203,769 transactions  
- **Edges:** 234,355 transaction links  
- **Features:** 166 anonymized attributes per node  
- **Labels:**  
  - `1` = illicit  
  - `2` = licit  
  - `0` = unknown  

---

## How to Run It Locally

1. **Create environment**
```bash
python3 -m venv elliptic-env
source elliptic-env/bin/activate
pip install -r requirements.txt
```

2. **Start Jupyter Notebook**
```bash
jupyter notebook
```
Then open `01_exploration.ipynb` and follow the sequence.

---
## 🚧 Project Status (ง •̀_•́)ง

- ✅ 01_exploration.ipynb – completed (graph structure, class distribution, LCC visualization)  
- ✅ 02_preprocessing.ipynb – completed (feature cleaning, deduplication: dropped 46 correlated columns, time-based split)  
- ⏳ 03_modeling.ipynb – in progress  
- 🔜 04_graph_models.ipynb – planned  
- 🔜 05_anomaly_detection.ipynb – planned  

---

## License
MIT License — Use it, extend it, and fight crime with it.  

> **"Graph theory meets criminal intent — one transaction at a time."**
