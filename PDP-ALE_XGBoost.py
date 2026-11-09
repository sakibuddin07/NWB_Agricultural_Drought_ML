"""
PDP and ALE Analysis for the Bayesian-Optimized XGBoost Model
============================================================
This script examines the model-estimated relationships between the
environmental predictors and the Vegetation Health Index (VHI)
using Partial Dependence Plots (PDP) and Accumulated Local Effects
(ALE).

The five most important predictors are selected based on the
feature importance scores of the optimized XGBoost model.

For each selected predictor, PDP and ALE curves are generated
to examine the direction, nonlinearity, and potential threshold
patterns in the model-estimated responses.

Requirements:
- best_xgb: Bayesian-optimized XGBoost model
- X_train: Training dataset used to fit the model

Main outputs:
- Combined PDP-ALE plots for the five selected predictors.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from matplotlib.lines import Line2D

# Feature importances from optimised XGBoost
importances = best_xgb.feature_importances_
columns_used = X_train.columns[:len(importances)]
feature_importance_df = pd.DataFrame({
    'feature': columns_used,
    'importance': importances
}).sort_values('importance', ascending=False)
top_features = feature_importance_df.head(5)['feature'].tolist()
print("Top 5 features:", top_features)

# Define BOLD color pair (same for all plots)
color_pdp = '#FF1744'  # Bold Red
color_ale = '#2979FF'  # Bold Blue

# Create figure with 1x5 subplots
fig, axes = plt.subplots(1, 5, figsize=(21, 7), sharey=False, facecolor='white')

for idx, feature in enumerate(top_features):
    ax = axes[idx]
    feature_idx = X_train.columns.get_loc(feature)

    # ---------- PDP ----------
    pd_result = partial_dependence(
        best_xgb,
        X_train,
        features=[feature_idx],
        grid_resolution=50
    )

    pdp_values = pd_result['average'][0]
    pdp_grid   = pd_result['grid_values'][0]

    ax.plot(pdp_grid, pdp_values, color=color_pdp, linewidth=3, label='PDP', linestyle='-')
    ax.set_xlabel(feature, fontsize=24, fontweight='bold')
    ax.set_ylabel('Partial Dependence', color=color_pdp, fontsize=24, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_pdp, labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.grid(True, alpha=0.3)

    # ---------- ALE ----------
    ax2 = ax.twinx()

    feature_values = X_train[feature].values
    n_bins         = 50
    quantiles      = np.linspace(0, 100, n_bins + 1)
    bin_edges      = np.percentile(feature_values, quantiles)

    ale_values  = []
    bin_centers = []

    for i in range(len(bin_edges) - 1):
        mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
        if np.sum(mask) > 0:
            X_bin   = X_train[mask].copy()
            X_upper = X_bin.copy()
            X_lower = X_bin.copy()
            X_upper[feature] = bin_edges[i + 1]
            X_lower[feature] = bin_edges[i]

            pred_upper   = best_xgb.predict(X_upper)
            pred_lower   = best_xgb.predict(X_lower)
            local_effect = np.mean(pred_upper - pred_lower)
            ale_values.append(local_effect)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    if len(ale_values) > 0:
        ale_accumulated  = np.cumsum(ale_values)
        ale_accumulated -= np.mean(ale_accumulated)

        ax2.plot(bin_centers, ale_accumulated, color=color_ale, linewidth=3, label='ALE', linestyle='--')
        ax2.set_ylabel('ALE', color=color_ale, fontsize=24, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_ale, labelsize=20)
        ax2.axhline(y=0, color=color_ale, linestyle=':', alpha=0.5, linewidth=2)

# Single legend outside subplots
legend_elements = [
    Line2D([0], [0], color=color_pdp, lw=3, label='PDP', linestyle='-'),
    Line2D([0], [0], color=color_ale, lw=3, label='ALE', linestyle='--')
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=20, 
           frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1, 1.12))

plt.tight_layout()
plt.suptitle('Bayesian Optimized XGBoost', fontsize=30, fontweight='bold', y=1.05, x=0.11)
plt.savefig(r'D:\Research\VHI_NWB\Review\Figure\Summer\PDP-ALE_XGBoost_Summer.png',
            dpi=600, bbox_inches='tight')
plt.show()
