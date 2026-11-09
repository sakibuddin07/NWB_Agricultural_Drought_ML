"""
SHAP Analysis for the Optimized XGBoost Model.

This script:
1. Uses the independent test dataset for SHAP analysis.
2. Computes SHAP values using TreeExplainer.
3. Ranks predictors by mean absolute SHAP values.
4. Generates a SHAP beeswarm summary plot.

Requirements:
- best_xgb: Bayesian-optimized XGBoost model
- X_test: Independent test dataset

The model and test dataset should be generated using
the XGBoost modeling script before running this file.
"""

# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')

# PLOT SETTINGS
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18

# OUTPUT SETTINGS
OUT_DIR = r'D:\Research\VHI_NWB\Review\Figure\Summer'

N_SHAP = 1000       # Maximum number of test samples
N_TOP = 10          # Number of features displayed
SEED = 42

# STEP 1 — SAMPLE FROM THE TEST DATASET
np.random.seed(SEED)

n_test = len(X_test)

if n_test >= N_SHAP:
    sample_idx = np.random.choice(
        n_test, N_SHAP, replace=False
    )
    X_shap = X_test.iloc[sample_idx].reset_index(drop=True)

    print(f"Sampled {N_SHAP} observations from test set.")
else:
    X_shap = X_test.reset_index(drop=True)

    print(f"Using all {n_test} test observations.")

# STEP 2 — COMPUTE SHAP VALUES
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_shap)

sv = shap_values.values
feature_names = list(X_shap.columns)

print(f"SHAP values computed: {sv.shape}")

# STEP 3 — RANK FEATURES BY MEAN ABSOLUTE SHAP
mean_abs_shap = np.abs(sv).mean(axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_abs_shap
}).sort_values(
    'Mean |SHAP|',
    ascending=False
).reset_index(drop=True)

top_features = importance_df.head(N_TOP)['Feature'].tolist()

top_indices = [
    feature_names.index(feature)
    for feature in top_features
]

print(f"\nTop {N_TOP} predictors by mean absolute SHAP value:")
print(importance_df.head(N_TOP).to_string(index=False))

# STEP 4 — SHAP BEESWARM SUMMARY PLOT
print("\nGenerating SHAP beeswarm plot...")

plt.figure(figsize=(8, 5))

shap.summary_plot(
    sv[:, top_indices],
    X_shap[top_features],
    feature_names=top_features,
    plot_type='dot',
    max_display=N_TOP,
    show=False,
    color_bar=True
)

plt.title(
    'XGBoost',
    fontsize=20,
    fontweight='bold',
    pad=12
)

plt.xlabel(
    'SHAP Value (impact on VHI prediction)',
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}\\SHAP_Summary_Beeswarm_XGBoost_Summer.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("\nSHAP beeswarm plot saved successfully.")

# SUMMARY
print("\n" + "=" * 55)
print("SHAP ANALYSIS COMPLETE")
print("=" * 55)
print(f"Samples used: {len(X_shap)}")
print(f"Features displayed: {len(top_features)}")
print(f"Output directory: {OUT_DIR}")
print("Output: SHAP_Summary_Beeswarm_XGBoost_Summer.png")
print("=" * 55)
