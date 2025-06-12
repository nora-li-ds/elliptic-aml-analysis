# 🕵️‍♀️ Chasing Dirty Coins: An AML Graph Project

Welcome to **Elliptic AML Analysis** — where we dive into the blockchain shadows, build networks of suspicious transactions, and try to answer one burning question:

> *Can we catch money launderers using machine learning and network science?*

This project is part of a broader portfolio to explore financial crime, graph modeling, and predictive analytics — with real-world anti-money laundering (AML) applications.

---


## 🚩 Why this project?

Because fraudsters are getting smarter, and it’s time for data scientists to catch up.  
Using the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), we’ll:

- Visualize how dirty money flows through networks
- Engineer meaningful graph features
- Train models that classify suspicious transactions
- Lay the foundation for research and advanced modeling like GNNs

**Note**: Due to file size limits, raw data files are not included in the repository. Please download `elliptic_txs_features.csv` from the original [Kaggle dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place it in `data/raw/`.

---

### 📥 Data Setup

To run this project, you need to manually download the dataset:

1. Go to [Kaggle: Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
2. Download the `.zip` file and extract it
3. Place the extracted `.csv` files into this folder:

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
│   └── 03_modeling.ipynb                <- ML pipelines & classification
├── scripts/
│   └── utils.py                         <- Custom helpers & metrics
├── reports/
│   ├── figures/                         <- Your best graphs go here
│   └── summary.md                       <- Reflective notes & insights
├── requirements.txt                     <- Everything you need to install
├── .gitignore                           <- Keeps things clean
└── README.md                            <- You are here
```

---

## 📦 About the Dataset

- 🧩 **Nodes**: 203,769 transactions
- 🔗 **Edges**: 234,355 transaction links
- 📐 **Features**: 166 anonymized attributes per node
- 🔍 **Labels**:  
  - `1` = illicit  
  - `2` = licit  
  - `0` = unknown

Get the dataset from Kaggle and unzip it into `data/elliptic_bitcoin_dataset/`.

---

## 🚀 How to Run It Locally

### 1. Create environment

```bash
python3 -m venv elliptic-env
source elliptic-env/bin/activate
pip install -r requirements.txt
```

### 2. Start exploring

```bash
jupyter notebook
```

Then open the notebooks and begin with `01_exploration.ipynb`.

---

## 📌 What’s Next?

- [ ] Add graph metrics (degree, centrality, etc.)
- [ ] Visualize communities & outliers
- [ ] Build classification models
- [ ] Try Graph Neural Networks (PyTorch Geometric or DGL)
- [ ] Write a PhD research proposal based on your findings
- [ ] Maybe catch a fraudster or two 🕵️‍♂️

---

## 📚 License

This project is open-sourced under the MIT License. Use it. Extend it. Fight crime with it.

---

🎯 *"Graph theory meets criminal intent — one transaction at a time."*

