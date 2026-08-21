"""
evaluation.py
=============
Runs all nowcasting models via POOS and reports results.

Models
------
  1. AR benchmark        — AR(2) on GDP lags
  2. RF benchmark        — Random Forest on GDP lags
  3. LASSO               — LASSO on simple average (X1)
  4. LASSO lags          — LASSO on simple average + lags (X2)
  5. RF avg              — Random Forest on simple average + lags (X2)
  6. LASSO U-MIDAS       — LASSO on U-MIDAS (X3)
  7. RF U-MIDAS          — Random Forest on U-MIDAS (X3)
  8. Ensemble            — Average of models 3–7 (excluding benchmarks)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import pipeline.poos as poos
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.gdp_data import load_filled_data, load_gdp, load_gdp_with_flash
from database.client import get_backend_client
from pipeline.feature_matrix import (
    build_X1, build_X2, build_X3, build_X4,
    build_X_AR,
)

NUM_TEST = 100


# =============================================================================
# STEP 1 — Load data and build feature matrices
# =============================================================================

print("=" * 60)
print("Step 1: Loading data and building feature matrices")
print("=" * 60)

client = get_backend_client()
df_md, df_qd = load_filled_data(client)
gdp_actual_series = load_gdp(client)["GDPC1_t"]
gdp_flash_series  = load_gdp_with_flash(client)

X_ar, y_ar = build_X_AR(gdp_actual_series, gdp_actual_series, n_lags=2)
X1,   y1   = build_X1(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X2,   y2   = build_X2(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X3,   y3   = build_X3(df_md, df_qd, gdp_flash_series, gdp_actual_series)
X4,   y4   = build_X4(df_md, df_qd, gdp_flash_series, gdp_actual_series)

# Columns to exclude — GDP components/accounting identities released with GDP
LEAKING_SERIES = {"OUTNFB", "OUTBS"}

def remove_leaking_columns(X: pd.DataFrame, leaking_series: set = LEAKING_SERIES) -> pd.DataFrame:
    cols_to_drop = [
        col for col in X.columns
        if col.split("_")[0] in leaking_series
    ]

    print(f"Dropping {len(cols_to_drop)} leaking columns: {cols_to_drop}")
    return X.drop(columns=cols_to_drop)


# ── Apply to all feature matrices ─────────────────────────────────────────────
X1 = remove_leaking_columns(X1)
X2 = remove_leaking_columns(X2)
X3 = remove_leaking_columns(X3)
X4 = remove_leaking_columns(X4)
# X_ar only uses GDP lags — no macro variables to drop

import matplotlib.pyplot as plt
import pandas as pd

def plot_gdp_correlations(X, y, title="Correlation with GDP Growth", top_n=50):
    """
    Plot barchart of correlations between each column in X and y (GDP growth).
    Shows top_n most correlated (by absolute value).
    """
    correlations = X.corrwith(y).dropna().sort_values(key=abs, ascending=False)
    top_corr = correlations.head(top_n)

    colors = ["steelblue" if c > 0 else "tomato" for c in top_corr.values]

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.3)))
    bars = ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1], edgecolor="white")

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Pearson Correlation with GDP Growth")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(f"pipeline/plots/{safe_title}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to pipeline/plots/{safe_title}.png")

    return correlations

os.makedirs("pipeline/plots", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca(X, y, n_components=10):
    """
    Run PCA on X and compute correlation of each PC with y (GDP growth).

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    n_components : int

    Returns
    -------
    pca         : fitted PCA object
    scores      : pd.DataFrame — PC scores (n_quarters, n_components)
    loadings    : pd.DataFrame — loadings (n_features, n_components)
    pc_gdp_corr : pd.Series   — correlation of each PC with GDP
    """
    # ── Align X and y ─────────────────────────────────────────────────────────
    df = pd.concat([X, y.rename("GDP")], axis=1).dropna()
    X_clean = df.drop(columns="GDP")
    y_clean = df["GDP"]

    # ── Standardise ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    pca = PCA(n_components=n_components)
    scores_arr = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    scores   = pd.DataFrame(scores_arr, index=X_clean.index, columns=pc_cols)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_clean.columns,
        columns=pc_cols
    )

    # ── Correlation of each PC with GDP ───────────────────────────────────────
    pc_gdp_corr = scores.corrwith(y_clean).rename("corr_with_GDP")

    print("=== PCA Summary ===")
    print(f"{'PC':<6} {'Var Explained':>14} {'Cumulative':>12} {'Corr w GDP':>12}")
    print("-" * 48)
    cumvar = 0
    for i, (var, corr) in enumerate(zip(pca.explained_variance_ratio_, pc_gdp_corr)):
        cumvar += var
        print(f"PC{i+1:<4} {var*100:>13.2f}%  {cumvar*100:>11.2f}%  {corr:>12.4f}")

    return pca, scores, loadings, pc_gdp_corr


def plot_pc_loadings(loadings, pc="PC1", top_n=30, save_dir="pipeline/plots"):
    """
    Plot the top_n largest (by absolute value) feature loadings for a given PC.

    Parameters
    ----------
    loadings : pd.DataFrame — output from run_pca()
    pc       : str          — e.g. "PC1", "PC2"
    top_n    : int          — number of features to show
    save_dir : str          — directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)

    pc_loadings = loadings[pc].sort_values(key=abs, ascending=False).head(top_n)
    colors = ["steelblue" if v > 0 else "tomato" for v in pc_loadings.values]

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
    ax.barh(pc_loadings.index[::-1], pc_loadings.values[::-1],
            color=colors[::-1], edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title(f"Top {top_n} Loadings — {pc}")
    ax.set_xlabel("Loading")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"loadings_{pc}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")


def plot_pc_gdp_correlation(pc_gdp_corr, save_dir="pipeline/plots"):
    """
    Bar chart of each PC's correlation with GDP growth.
    """
    os.makedirs(save_dir, exist_ok=True)

    colors = ["steelblue" if v > 0 else "tomato" for v in pc_gdp_corr.values]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pc_gdp_corr.index, pc_gdp_corr.values, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_title("Correlation of Each PC with GDP Growth")
    ax.set_ylabel("Pearson Correlation")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "pc_gdp_correlations.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_path}")

def plot_gdp_correlations_lead(X, y, title="Correlation with GDP Growth (t+1)", top_n=50):
    """
    Correlation between X at time t and GDP at time t+1.
    i.e. how well does X today predict GDP next quarter.
    """
    y_lead = y.shift(-1)  # shift y back by 1 — now y[t] = GDP at t+1

    correlations = X.corrwith(y_lead).dropna().sort_values(key=abs, ascending=False)
    top_corr = correlations.head(top_n)

    colors = ["steelblue" if c > 0 else "tomato" for c in top_corr.values]

    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.3)))
    ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Pearson Correlation with GDP Growth (t+1)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    os.makedirs("pipeline/plots", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("+", "plus")
    plt.savefig(f"pipeline/plots/{safe_title}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return correlations

if __name__ == "__main__":
    # QD columns only (no _md suffix, no lag columns)
    qd_cols = [c for c in X1.columns if not c.endswith("_md") and "_lag" not in c]
    X1_qd = X1[qd_cols]

    correlations = plot_gdp_correlations(X1_qd, y1, title="QD Correlation with GDP Growth", top_n=50)

    print("\nTop 20 most correlated QD columns:")
    print(correlations.head(20).to_string())

    print("\nTop 20 most negatively correlated:")
    print(correlations.tail(20).to_string())

    # Run PCA on X1 (simple avg) — change to X3 etc. as needed
    pca, scores, loadings, pc_gdp_corr = run_pca(X1, y1, n_components=10)

    # Plot PC-GDP correlations
    plot_pc_gdp_correlation(pc_gdp_corr)

    # Plot loadings for specific PCs
    plot_pc_loadings(loadings, pc="PC1", top_n=30)
    plot_pc_loadings(loadings, pc="PC2", top_n=30)
    plot_pc_loadings(loadings, pc="PC3", top_n=30)

    # Which PC correlates most with GDP?
    best_pc = pc_gdp_corr.abs().idxmax()
    print(f"\nPC most correlated with GDP: {best_pc} "
          f"(r = {pc_gdp_corr[best_pc]:.4f})")
    plot_pc_loadings(loadings, pc=best_pc, top_n=30,)

    # Usage
    correlations_lead = plot_gdp_correlations_lead(X1, y1, top_n=50)
    print(correlations_lead.head(20).to_string())

