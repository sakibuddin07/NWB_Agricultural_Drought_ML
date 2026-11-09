"""
10-Zone Spatial Hold-Out Evaluation of the Optimized XGBoost Model
==================================================================

This script assesses the spatial generalization performance of
the Bayesian-optimized XGBoost model across ten geographically
defined zones.

The study area is divided into ten spatial zones using K-means
clustering based on geographic coordinates. In each iteration,
one zone is held out for testing, while the remaining nine zones
are used to train a new XGBoost model.

The optimized hyperparameters obtained from the main modeling
procedure are retained for all ten evaluations.

Model performance is assessed using:
- Coefficient of determination (R²)
- Root mean square error (RMSE)
- Mean absolute error (MAE)
- Lin's concordance correlation coefficient (CCC)

The script generates:
- Zone-wise model performance plots
- A spatial distribution map of the ten zones
- Zone-wise performance results
- Summary statistics across all ten evaluations

Requirements:
- df: Complete spatial dataset containing VHI, X, Y,
  predictor variables, and spatial_group.
- best_xgb: Bayesian-optimized XGBoost model.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# =====================================================
# CONFIGURATION
# =====================================================
N_ZONES = 10          # Divide study area into 10 zones
SEED = 42
OUT_DIR = r'D:\Research\VHI_NWB\Review\Figure\Summer'

# =====================================================
# CREATE 10 SPATIAL ZONES FROM FULL DATASET
# =====================================================
print("="*58)
print("  CREATING 10 SPATIAL ZONES (KMEANS CLUSTERING)")
print("="*58)

# Use coordinates from FULL dataset to create zones
coords = df[['X', 'Y']].values

# Create 10 distinct spatial zones
kmeans_zones = KMeans(n_clusters=N_ZONES, random_state=SEED, n_init=10)
df['spatial_zone'] = kmeans_zones.fit_predict(coords)
zones = df['spatial_zone'].values

print(f"Total samples: {len(df):,}")
print(f"Number of zones: {N_ZONES}")
print("\nZone sizes:")
zone_counts = df['spatial_zone'].value_counts().sort_index()
for zone, count in zone_counts.items():
    print(f"  Zone {zone}: {count:,} samples ({count/len(df)*100:.1f}%)")
print("="*58)

# PREPARE FEATURES (using same columns as before)
X_full = df.drop(columns=['VHI', 'X', 'Y', 'spatial_group', 'spatial_zone'])
y_full = df['VHI']

# EXTRACT BEST PARAMS FROM OPTIMIZED MODEL
params = best_xgb.get_params()
print("\nUsing optimized parameters from Bayesian optimization")
print("="*58)

# 10-FOLD SPATIAL CV (Hold out 1 zone per fold)
fold_results = []

for fold, test_zone in enumerate(range(N_ZONES), start=1):
    
    # Create train/test indices based on zone membership
    test_idx = np.where(zones == test_zone)[0]
    train_idx = np.where(zones != test_zone)[0]
    
    X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]
    
    # Train a fresh model with best parameters
    fold_model = XGBRegressor(**params)
    fold_model.fit(X_tr, y_tr)
    
    # Predict on test zone
    y_pred = fold_model.predict(X_te)
    
    # Calculate metrics
    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    
    # Concordance Correlation Coefficient (CCC)
    mu_t = np.mean(y_te)
    mu_p = np.mean(y_pred)
    var_t = np.var(y_te)
    var_p = np.var(y_pred)
    cov = np.mean((y_te.values - mu_t) * (y_pred - mu_p))
    ccc = (2 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2)
    
    fold_results.append({
        'Fold': fold,
        'Test_Zone': test_zone,
        'n_train': len(y_tr),
        'n_test': len(y_te),
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CCC': ccc
    })
    
    print(f"Fold {fold:>2} | Test Zone={test_zone} | "
          f"n_train={len(y_tr):>5} | n_test={len(y_te):>5} | "
          f"R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | CCC={ccc:.4f}")

# SUMMARY STATISTICS
results_df = pd.DataFrame(fold_results)

metrics = ['R²', 'RMSE', 'MAE', 'CCC']
summary = pd.DataFrame({
    'Metric': metrics,
    'Mean': [results_df[m].mean() for m in metrics],
    'Std': [results_df[m].std() for m in metrics],
    'Min': [results_df[m].min() for m in metrics],
    'Max': [results_df[m].max() for m in metrics],
})

print("\n" + "="*58)
print(f"  SPATIAL CV SUMMARY ({N_ZONES}-ZONE HOLD-OUT)")
print("="*58)
print(summary.to_string(index=False))
print("="*58)

# VISUALIZATION 1: Metrics per Zone (2x2 grid)
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
axes1 = axes1.flatten()

colors = ['#1a6faf', '#d62728', '#2ca02c', '#9467bd']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes1[i]
    vals = results_df[metric].values
    mean_val = vals.mean()
    std_val = vals.std()
    
    ax.bar(range(1, N_ZONES+1), vals, color=color, 
           alpha=0.75, edgecolor='black', linewidth=0.6)
    ax.axhline(mean_val, color='black', linewidth=1.8, 
               linestyle='--', label=f'Mean = {mean_val:.4f}')
    ax.fill_between([0.5, N_ZONES+0.5], mean_val - std_val, mean_val + std_val,
                    color='gray', alpha=0.15, label=f'±1 SD = {std_val:.4f}')
    
    ax.set_title(f'{metric} per Zone', fontweight='bold')
    ax.set_xlabel('Test Zone', fontsize=16)
    ax.set_ylabel(metric, fontsize=16)
    ax.set_xticks(range(1, N_ZONES+1))
    ax.legend(fontsize=12, loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle(f'XGBoost',
             fontsize=25, fontweight='bold', y=0.95)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\SpatialCV_10Zone_Metrics_XGBoost_Summer.png", dpi=600, bbox_inches='tight')
plt.show()
print("\nSaved: SpatialCV_10Zone_Metrics_XGBoost_Summer.png")

# VISUALIZATION 2: Spatial Map of Zones
fig2, ax2 = plt.subplots(figsize=(10, 8))

cmap = plt.cm.get_cmap('tab10', N_ZONES)

sc = ax2.scatter(df['X'].values, df['Y'].values, 
                 c=df['spatial_zone'].values, cmap='tab10',
                 s=3, alpha=0.6, vmin=0, vmax=N_ZONES-1)

cbar = plt.colorbar(sc, ax=ax2, ticks=range(N_ZONES))
cbar.set_label('Spatial Zone', fontsize=18)
cbar.ax.set_yticklabels([f'Zone {i}' for i in range(N_ZONES)])

ax2.set_xlabel('X Coordinate', fontsize=18)
ax2.set_ylabel('Y Coordinate', fontsize=18)
ax2.set_title(f'XGBoost',
              fontsize=25, fontweight='bold')
ax2.set_aspect('equal', adjustable='box')
ax2.grid(alpha=0.2, linestyle='--')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\SpatialCV_10Zone_Map_XGBoost_Summer.png", dpi=600, bbox_inches='tight')
plt.show()
print("Saved: SpatialCV_10Zone_Map_XGBoost_Summer.png")

# ZONE PERFORMANCE RANKING
print("\n" + "="*58)
print("  ZONE PERFORMANCE RANKING (by R²)")
print("="*58)

best_zone = results_df.loc[results_df['R²'].idxmax()]
worst_zone = results_df.loc[results_df['R²'].idxmin()]

print(f"\nBest performing test zone:  Zone {int(best_zone['Test_Zone'])}")
print(f"  R² = {best_zone['R²']:.4f}, RMSE = {best_zone['RMSE']:.4f}")
print(f"\nWorst performing test zone: Zone {int(worst_zone['Test_Zone'])}")
print(f"  R² = {worst_zone['R²']:.4f}, RMSE = {worst_zone['RMSE']:.4f}")

# SAVE RESULTS
results_df.to_csv(f"{OUT_DIR}\\SpatialCV_10Zone_Results_XGBoost_Summer.csv", index=False)
summary.to_csv(f"{OUT_DIR}\\SpatialCV_10Zone_Summary_XGBoost_Summer.csv", index=False)
