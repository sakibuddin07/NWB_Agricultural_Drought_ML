"""
ExtraTrees model for spatial prediction of the Vegetation Health Index (VHI).

Includes:
1. K-means-based spatial grouping
2. Spatially separated 80:20 train-test split
3. Base ExtraTrees model
4. Bayesian hyperparameter optimization
5. Five-fold GroupKFold spatial cross-validation
6. Model performance evaluation

Input: A CSV dataset containing VHI, X, Y, and predictor variables.
"""
# IMPORTS
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor

from skopt import BayesSearchCV
from skopt.space import Integer, Real

# SPATIAL GROUPING (KMEANS)

coords = df[['X', 'Y']].values

kmeans = KMeans(n_clusters=200, random_state=42, n_init='auto')
df['spatial_group'] = kmeans.fit_predict(coords)

groups = df['spatial_group']

# FEATURES & TARGET
X = df.drop(columns=['VHI', 'X', 'Y', 'spatial_group'])
y = df['VHI']
,

# TRAIN / TEST SPLIT (80-20)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"Training samples: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test samples: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Unique spatial groups in train: {groups_train.nunique()}")
print(f"Unique spatial groups in test: {groups.iloc[test_idx].nunique()}")

# METRICS FUNCTION
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# BASE EXTRA TREES MODEL
et_base = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

et_base.fit(X_train, y_train)

# PREDICTIONS (BASE)
y_train_pred = et_base.predict(X_train)
y_test_pred = et_base.predict(X_test)

results_base = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'R²': [
        calculate_metrics(y_train, y_train_pred)[0],
        calculate_metrics(y_test, y_test_pred)[0],
    ],
    'RMSE': [
        calculate_metrics(y_train, y_train_pred)[1],
        calculate_metrics(y_test, y_test_pred)[1],
    ],
    'MAE': [
        calculate_metrics(y_train, y_train_pred)[2],
        calculate_metrics(y_test, y_test_pred)[2],
    ]
})

print("\n" + "="*58)
print("BASE EXTRA TREES MODEL PERFORMANCE (80-20 SPLIT)")
print("="*58)
print(results_base.to_string(index=False))
print("="*58)

# BAYESIAN OPTIMIZATION
search_spaces = {
    'n_estimators': Integer(200, 1200),
    'max_depth': Integer(3, 12),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.3, 1.0)
}

et_model = ExtraTreesRegressor(
    random_state=42,
    n_jobs=-1
)

spatial_cv = GroupKFold(n_splits=5)

bayes_search = BayesSearchCV(
    estimator=et_model,
    search_spaces=search_spaces,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=spatial_cv,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

bayes_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

best_et = bayes_search.best_estimator_

print("\n" + "="*58)
print("BEST PARAMETERS FROM BAYESIAN OPTIMIZATION")
print("="*58)

for k, v in bayes_search.best_params_.items():
    print(f"  {k}: {v}")

print("="*58)

# OPTIMIZED MODEL PREDICTIONS
y_train_pred_opt = best_et.predict(X_train)
y_test_pred_opt = best_et.predict(X_test)

results_opt = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'R²': [
        calculate_metrics(y_train, y_train_pred_opt)[0],
        calculate_metrics(y_test, y_test_pred_opt)[0],
    ],
    'RMSE': [
        calculate_metrics(y_train, y_train_pred_opt)[1],
        calculate_metrics(y_test, y_test_pred_opt)[1],
    ],
    'MAE': [
        calculate_metrics(y_train, y_train_pred_opt)[2],
        calculate_metrics(y_test, y_test_pred_opt)[2],
    ]
})

print("\n" + "="*58)
print("OPTIMIZED EXTRA TREES MODEL PERFORMANCE (80-20 SPLIT)")
print("="*58)
print(results_opt.to_string(index=False))
print("="*58)
