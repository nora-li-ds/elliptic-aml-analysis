# 🕵️‍♀️ Chasing Dirty Coins: An AML Graph Project

Welcome to **Elliptic AML Analysis** — where we dive into the blockchain shadows, build networks of suspicious transactions, and try to answer one burning question:

> **Can we catch money launderers using machine learning and network science?**

This project is part of a broader portfolio to explore **financial crime**, **graph modeling**, and **predictive analytics** — with real-world **anti-money laundering (AML)** applications.

---

## 🚩 Why this project?

Fraudsters are getting smarter.  
It’s time for data scientists to catch up.

Using the **Elliptic Bitcoin Dataset**, we will:

- Visualize how dirty money flows through networks  
- Engineer meaningful graph features  
- Train models that classify suspicious transactions  
- Extend to advanced modeling like **Graph Neural Networks (GNNs)**  
- Detect illicit patterns in **unlabeled transactions**  

> Note: Due to file size limits, raw data files are **not included** in this repository.  
> Please download `elliptic_txs_features.csv` from the original Kaggle dataset and place it in `data/raw/`.

---

## 📥 Data Setup

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

## 📚 Notebook Overview

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

## 📦 About the Dataset

- **Nodes:** 203,769 transactions  
- **Edges:** 234,355 transaction links  
- **Features:** 166 anonymized attributes per node  
- **Labels:**  
  - `1` = illicit  
  - `2` = licit  
  - `0` = unknown  

---

## 🚀 How to Run It Locally

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

## 📌 What's Next?
- Add graph metrics (degree, centrality, community detection)  
- Run GNN experiments for better detection accuracy  
- Apply anomaly detection to unknown-class transactions  
- Extend to real-world AML case studies  
- Draft PhD research proposal based on findings  

---

## 📚 License
MIT License — Use it, extend it, and fight crime with it.  

> **"Graph theory meets criminal intent — one transaction at a time."**
