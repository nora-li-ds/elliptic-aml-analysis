# preprocessing.py

"""Feature cleaning, correlation filtering, and train/valid/test split."""

"""
preprocessing.py
Helper functions for data cleaning, feature engineering, and dataset splits.
"""

import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler

def drop_high_corr(df, threshold=0.98, exclude_cols=None):
    """
    Drop features with correlation above threshold.
    """
    if exclude_cols is None:
        exclude_cols = []
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_clean = df.drop(columns=to_drop)
    return df_clean, to_drop

def time_based_split(df, timestep_col="timestep", ratios=(0.7, 0.8)):
    """
    Split dataset by timestep into train, valid, test.
    ratios: (train_quantile, valid_quantile)
    """
    df = df.sort_values(timestep_col).reset_index(drop=True)
    ts = df[timestep_col].values
    t1, t2 = np.quantile(ts, ratios[0]), np.quantile(ts, ratios[1])

    train_idx = df[timestep_col] <= t1
    valid_idx = (df[timestep_col] > t1) & (df[timestep_col] <= t2)
    test_idx  = df[timestep_col] > t2

    return df[train_idx], df[valid_idx], df[test_idx]

def scale_features(train, valid, test, feature_cols):
    """
    Fit scaler on train, transform train/valid/test.
    """
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    return (
        scaler.transform(train[feature_cols]),
        scaler.transform(valid[feature_cols]),
        scaler.transform(test[feature_cols]),
        scaler
    )
