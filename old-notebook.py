# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: dmml
#     language: python
#     name: dmml
# ---

# %% [markdown]
# # Stat models
#
# This notebook contains the classical machine learning models. Below are the models it will fit:
#
# - Linear Regression
# - KNN Regression
# - Decision Tree Regression

# %%
import os

# Save the best KNN model to avoid rerunning grid search
# Try to load pre-trained KNN model if it exists
import pickle

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LinearRegression

# Make predictions and evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tensorflow import keras
from tensorflow.keras import layers

# %%
df1 = pd.read_csv("processed/usa_real_estate.csv")
df2 = pd.read_csv("processed/zipcodes.csv")

# %%
df1.info()

# %%
df1.describe()

# %%
df1.head()

# %%
df2.info()

# %%
df2.describe()

# %%
df2.head()

# %% [markdown]
# ## Linear Regression

# %%
# Check column names
print("df1 columns:", df1.columns.tolist())
print("\ndf2 columns:", df2.columns.tolist())
print("\ndf2 sample:")
df2.head()

# %%
# Merge datasets on zip_code
# We'll use demographic and amenity features from df2, and house characteristics from df1
df_merged = df1.merge(
    df2[
        [
            "zipcode",
            "bank",
            "bus",
            "hospital",
            "mall",
            "park",
            "restaurant",
            "school",
            "station",
            "supermarket",
            "Total Population",
            "Median Age",
            "Per Capita Income",
            "Total Families Below Poverty",
            "Total Housing Units",
            "Total Labor Force",
            "Unemployed Population",
            "Total School Age Population",
            "Total School Enrollment",
            "Median Commute Time",
        ]
    ].drop_duplicates(subset="zipcode"),
    left_on="zip_code",
    right_on="zipcode",
    how="inner",
)

print(f"Original df1 shape: {df1.shape}")
print(f"Merged dataframe shape: {df_merged.shape}")
print(f"\nMissing values:\n{df_merged.isnull().sum()}")
df_merged.head()

# %%
# Prepare features for Linear Regression
# Using house characteristics + demographic + amenity features (excluding price-related features)
feature_columns = [
    # House characteristics
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    # Amenities
    "bank",
    "bus",
    "hospital",
    "mall",
    "park",
    "restaurant",
    "school",
    "station",
    "supermarket",
    # Demographics
    "Total Population",
    "Median Age",
    "Per Capita Income",
    "Total Families Below Poverty",
    "Total Housing Units",
    "Total Labor Force",
    "Unemployed Population",
    "Total School Age Population",
    "Total School Enrollment",
    "Median Commute Time",
]

X = df_merged[feature_columns]
y = df_merged["price"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures being used:")
for i, col in enumerate(feature_columns, 1):
    print(f"{i}. {col}")

# %%
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# %%
# Standardize the features (important for linear regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized successfully")

# %%
# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully!")
print(f"Number of features used: {len(feature_columns)}")


# %%

y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("=" * 60)
print("LINEAR REGRESSION MODEL PERFORMANCE")
print("=" * 60)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:,.2f}")
print(f"  MAE: ${train_mae:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:,.2f}")
print(f"  MAE: ${test_mae:,.2f}")
print("=" * 60)

# %%
# Analyze feature importance (coefficients)
feature_importance = pd.DataFrame(
    {"Feature": feature_columns, "Coefficient": lr_model.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

print("\nTop 10 Most Important Features (by absolute coefficient):")
print(feature_importance.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
colors = ["green" if x > 0 else "red" for x in top_features["Coefficient"]]
plt.barh(range(len(top_features)), top_features["Coefficient"], color=colors)
plt.yticks(range(len(top_features)), top_features["Feature"])
plt.xlabel("Coefficient Value")
plt.title("Top 15 Feature Coefficients in Linear Regression Model")
plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
# Visualize predictions vs actual values
plt.figure(figsize=(12, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.3, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Training Set\nR² = {train_r2:.4f}")
plt.ticklabel_format(style="plain", axis="both")

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Testing Set\nR² = {test_r2:.4f}")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# %%
# Residual analysis
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure(figsize=(12, 5))

# Residual plot for training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, residuals_train, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Training Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

# Residual plot for testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals_test, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Testing Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

print(f"\nResidual Statistics (Test Set):")
print(f"  Mean: ${residuals_test.mean():,.2f}")
print(f"  Std Dev: ${residuals_test.std():,.2f}")
print(f"  Min: ${residuals_test.min():,.2f}")
print(f"  Max: ${residuals_test.max():,.2f}")

# %% [markdown]
# ## Model Improvements
#
# Now let's try several techniques to improve model performance:
# 1. **Stratified Sampling**: Use stratified train-test split based on price bins
# 2. **K-Fold Cross-Validation**: Evaluate model with cross-validation

# %% [markdown]
# ### 1. Stratified Sampling
#
# Let's use stratified sampling for the train-test split based on price bins to ensure balanced representation across price ranges.

# %%
# Create price bins for stratification
n_bins = 10
stratify_labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

print(f"Created {len(stratify_labels.unique())} price bins for stratification")
print(f"\nPrice bin distribution:")
print(pd.Series(stratify_labels).value_counts().sort_index())

# Perform stratified split using all features
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_labels
)

print(f"\nStratified split completed:")
print(f"  Training set size: {X_train_strat.shape[0]}")
print(f"  Testing set size: {X_test_strat.shape[0]}")

# Scale features
scaler_strat = StandardScaler()
X_train_strat_scaled = scaler_strat.fit_transform(X_train_strat)
X_test_strat_scaled = scaler_strat.transform(X_test_strat)

# Train model with stratified data
lr_strat = LinearRegression()
lr_strat.fit(X_train_strat_scaled, y_train_strat)

# Evaluate
y_train_pred_strat = lr_strat.predict(X_train_strat_scaled)
y_test_pred_strat = lr_strat.predict(X_test_strat_scaled)

train_r2_strat = r2_score(y_train_strat, y_train_pred_strat)
test_r2_strat = r2_score(y_test_strat, y_test_pred_strat)
train_rmse_strat = np.sqrt(mean_squared_error(y_train_strat, y_train_pred_strat))
test_rmse_strat = np.sqrt(mean_squared_error(y_test_strat, y_test_pred_strat))
train_mae_strat = mean_absolute_error(y_train_strat, y_train_pred_strat)
test_mae_strat = mean_absolute_error(y_test_strat, y_test_pred_strat)

print("\n" + "=" * 70)
print("STRATIFIED SAMPLING RESULTS")
print("=" * 70)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_strat:.6f}")
print(f"  RMSE: ${train_rmse_strat:,.2f}")
print(f"  MAE: ${train_mae_strat:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_strat:.6f}")
print(f"  RMSE: ${test_rmse_strat:,.2f}")
print(f"  MAE: ${test_mae_strat:,.2f}")
print(f"\nComparison with original model:")
print(f"  R² difference: {(test_r2_strat - test_r2):.6f}")
print(f"  RMSE difference: ${(test_rmse_strat - test_rmse):,.2f}")
print("=" * 70)

# %% [markdown]
# ### 2. K-Fold Cross-Validation
#
# Let's evaluate the model using k-fold cross-validation to get a more robust performance estimate.

# %%

# Create pipeline with scaler and model
pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

# Perform k-fold cross-validation with multiple metrics
k = 10
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"Performing {k}-fold cross-validation...")
print("This will take a few moments...\n")

# Cross-validate with multiple scoring metrics
scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
cv_results = cross_validate(
    pipeline, X, y, cv=kfold, scoring=scoring, return_train_score=True, n_jobs=-1
)

# Calculate metrics
cv_train_r2 = cv_results["train_r2"]
cv_test_r2 = cv_results["test_r2"]
cv_train_rmse = np.sqrt(-cv_results["train_neg_mean_squared_error"])
cv_test_rmse = np.sqrt(-cv_results["test_neg_mean_squared_error"])
cv_train_mae = -cv_results["train_neg_mean_absolute_error"]
cv_test_mae = -cv_results["test_neg_mean_absolute_error"]

print("=" * 70)
print(f"{k}-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)
print(f"\nR² Score:")
print(f"  Training:   {cv_train_r2.mean():.6f} (+/- {cv_train_r2.std():.6f})")
print(f"  Validation: {cv_test_r2.mean():.6f} (+/- {cv_test_r2.std():.6f})")
print(f"\nRMSE:")
print(f"  Training:   ${cv_train_rmse.mean():,.2f} (+/- ${cv_train_rmse.std():,.2f})")
print(f"  Validation: ${cv_test_rmse.mean():,.2f} (+/- ${cv_test_rmse.std():,.2f})")
print(f"\nMAE:")
print(f"  Training:   ${cv_train_mae.mean():,.2f} (+/- ${cv_train_mae.std():,.2f})")
print(f"  Validation: ${cv_test_mae.mean():,.2f} (+/- ${cv_test_mae.std():,.2f})")
print(f"\nFold-by-fold R² scores:")
for i, score in enumerate(cv_test_r2, 1):
    print(f"  Fold {i:2d}: {score:.6f}")
print("=" * 70)

# %%
# Visualize cross-validation results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² scores across folds
axes[0].plot(
    range(1, k + 1), cv_train_r2, "o-", label="Training", linewidth=2, markersize=8
)
axes[0].plot(
    range(1, k + 1), cv_test_r2, "s-", label="Validation", linewidth=2, markersize=8
)
axes[0].axhline(
    y=cv_test_r2.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: {cv_test_r2.mean():.4f}",
)
axes[0].fill_between(
    range(1, k + 1),
    cv_test_r2.mean() - cv_test_r2.std(),
    cv_test_r2.mean() + cv_test_r2.std(),
    alpha=0.2,
    color="red",
)
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("R² Score")
axes[0].set_title("R² Score Across Folds")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RMSE across folds
axes[1].plot(
    range(1, k + 1), cv_train_rmse, "o-", label="Training", linewidth=2, markersize=8
)
axes[1].plot(
    range(1, k + 1), cv_test_rmse, "s-", label="Validation", linewidth=2, markersize=8
)
axes[1].axhline(
    y=cv_test_rmse.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: ${cv_test_rmse.mean():,.0f}",
)
axes[1].fill_between(
    range(1, k + 1),
    cv_test_rmse.mean() - cv_test_rmse.std(),
    cv_test_rmse.mean() + cv_test_rmse.std(),
    alpha=0.2,
    color="red",
)
axes[1].set_xlabel("Fold")
axes[1].set_ylabel("RMSE ($)")
axes[1].set_title("RMSE Across Folds")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(style="plain", axis="y")

# MAE across folds
axes[2].plot(
    range(1, k + 1), cv_train_mae, "o-", label="Training", linewidth=2, markersize=8
)
axes[2].plot(
    range(1, k + 1), cv_test_mae, "s-", label="Validation", linewidth=2, markersize=8
)
axes[2].axhline(
    y=cv_test_mae.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: ${cv_test_mae.mean():,.0f}",
)
axes[2].fill_between(
    range(1, k + 1),
    cv_test_mae.mean() - cv_test_mae.std(),
    cv_test_mae.mean() + cv_test_mae.std(),
    alpha=0.2,
    color="red",
)
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("MAE ($)")
axes[2].set_title("MAE Across Folds")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].ticklabel_format(style="plain", axis="y")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Summary Comparison
#
# Let's compare all the approaches we've tried:

# %%
# Create summary comparison table
summary_data = {
    "Approach": [
        "Original (All Features)",
        "With Stratified Sampling",
        f"{k}-Fold Cross-Validation (Mean)",
    ],
    "Features": [len(feature_columns), len(feature_columns), len(feature_columns)],
    "Test R²": [test_r2, test_r2_strat, cv_test_r2.mean()],
    "Test RMSE": [test_rmse, test_rmse_strat, cv_test_rmse.mean()],
    "Test MAE": [test_mae, test_mae_strat, cv_test_mae.mean()],
}

summary_df = pd.DataFrame(summary_data)

print("=" * 90)
print("FINAL SUMMARY - ALL APPROACHES")
print("=" * 90)
print(summary_df.to_string(index=False))
print("=" * 90)

# Find best approach
best_idx = summary_df["Test R²"].idxmax()
print(f"\nBest Approach: {summary_df.loc[best_idx, 'Approach']}")
print(f"  R² Score: {summary_df.loc[best_idx, 'Test R²']:.6f}")
print(f"  RMSE: ${summary_df.loc[best_idx, 'Test RMSE']:,.2f}")
print(f"  MAE: ${summary_df.loc[best_idx, 'Test MAE']:,.2f}")
print(f"  Number of Features: {summary_df.loc[best_idx, 'Features']}")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² comparison
axes[0].bar(
    range(len(summary_df)),
    summary_df["Test R²"],
    color=["#1f77b4", "#2ca02c", "#d62728"],
)
axes[0].set_xticks(range(len(summary_df)))
axes[0].set_xticklabels(summary_df["Approach"], rotation=45, ha="right")
axes[0].set_ylabel("R² Score")
axes[0].set_title("R² Score Comparison")
axes[0].grid(True, alpha=0.3, axis="y")

# RMSE comparison
axes[1].bar(
    range(len(summary_df)),
    summary_df["Test RMSE"],
    color=["#1f77b4", "#2ca02c", "#d62728"],
)
axes[1].set_xticks(range(len(summary_df)))
axes[1].set_xticklabels(summary_df["Approach"], rotation=45, ha="right")
axes[1].set_ylabel("RMSE ($)")
axes[1].set_title("RMSE Comparison")
axes[1].ticklabel_format(style="plain", axis="y")
axes[1].grid(True, alpha=0.3, axis="y")

# MAE comparison
axes[2].bar(
    range(len(summary_df)),
    summary_df["Test MAE"],
    color=["#1f77b4", "#2ca02c", "#d62728"],
)
axes[2].set_xticks(range(len(summary_df)))
axes[2].set_xticklabels(summary_df["Approach"], rotation=45, ha="right")
axes[2].set_ylabel("MAE ($)")
axes[2].set_title("MAE Comparison")
axes[2].ticklabel_format(style="plain", axis="y")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## KNN Regression
#
# Now let's implement K-Nearest Neighbors regression and compare it with Linear Regression.

# %% [markdown]
# ### Load Pre-trained KNN Model (Optional)
#
# If you've already run the grid search and saved the model, you can load it here to skip the training process.
#
# import os
# %%

USE_SAVED_MODEL = True  # Set to False to retrain from scratch

model_path = "models/best_knn_model.pkl"
scaler_path = "models/scaler.pkl"

if USE_SAVED_MODEL and os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Loading saved KNN model and scaler...")

    with open(model_path, "rb") as f:
        best_knn_model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load the scaled data if needed
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Model and scaler loaded successfully!")
    print(f"\nModel parameters:")
    print(f"  k = {best_knn_model.n_neighbors}")
    print(f"  weights = {best_knn_model.weights}")
    print(f"  metric = {best_knn_model.metric}")
    print("\n⚠️  Skip the grid search cells below and jump to the evaluation cells.")

    SKIP_GRID_SEARCH = True
else:
    print("No saved model found. Will train from scratch.")
    print("Run the cells below to perform grid search and train the model.")
    SKIP_GRID_SEARCH = False


# %%

# Train a basic KNN model with default parameters (k=5)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_knn = knn_model.predict(X_train_scaled)
y_test_pred_knn = knn_model.predict(X_test_scaled)

# Calculate metrics
train_r2_knn = r2_score(y_train, y_train_pred_knn)
test_r2_knn = r2_score(y_test, y_test_pred_knn)
train_rmse_knn = np.sqrt(mean_squared_error(y_train, y_train_pred_knn))
test_rmse_knn = np.sqrt(mean_squared_error(y_test, y_test_pred_knn))
train_mae_knn = mean_absolute_error(y_train, y_train_pred_knn)
test_mae_knn = mean_absolute_error(y_test, y_test_pred_knn)

print("=" * 70)
print("KNN REGRESSION MODEL PERFORMANCE (k=5)")
print("=" * 70)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_knn:.6f}")
print(f"  RMSE: ${train_rmse_knn:,.2f}")
print(f"  MAE: ${train_mae_knn:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_knn:.6f}")
print(f"  RMSE: ${test_rmse_knn:,.2f}")
print(f"  MAE: ${test_mae_knn:,.2f}")
print("\n" + "=" * 70)

# %% [markdown]
# ### Hyperparameter Tuning with GridSearchCV
#
# Let's find the optimal value of k using GridSearchCV.

# %%
# Define parameter grid
param_grid = {
    "n_neighbors": [3, 5, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

# Only run grid search if model wasn't loaded
if not SKIP_GRID_SEARCH:
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        KNeighborsRegressor(), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )

    print("Performing GridSearchCV for KNN hyperparameter tuning...")
    print("This may take a few minutes...\n")

    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)

    print("\n" + "=" * 70)
    print("GRIDSEARCHCV RESULTS")
    print("=" * 70)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R² score: {grid_search.best_score_:.6f}")
    print("=" * 70)
else:
    print("⏭️  Skipping grid search - using loaded model")
    print("(Set USE_SAVED_MODEL = False in the cell above to retrain)")

# %%
# Get the best model (skip if already loaded)
if not SKIP_GRID_SEARCH:
    best_knn_model = grid_search.best_estimator_

# Make predictions with the best model
y_train_pred_knn_best = best_knn_model.predict(X_train_scaled)
y_test_pred_knn_best = best_knn_model.predict(X_test_scaled)

# Calculate metrics
train_r2_knn_best = r2_score(y_train, y_train_pred_knn_best)
test_r2_knn_best = r2_score(y_test, y_test_pred_knn_best)
train_rmse_knn_best = np.sqrt(mean_squared_error(y_train, y_train_pred_knn_best))
test_rmse_knn_best = np.sqrt(mean_squared_error(y_test, y_test_pred_knn_best))
train_mae_knn_best = mean_absolute_error(y_train, y_train_pred_knn_best)
test_mae_knn_best = mean_absolute_error(y_test, y_test_pred_knn_best)

print("=" * 70)
print("OPTIMIZED KNN REGRESSION MODEL PERFORMANCE")
print("=" * 70)
if not SKIP_GRID_SEARCH:
    print(f"\nBest Parameters: {grid_search.best_params_}")
else:
    print(f"\nModel Parameters (loaded from file):")
    print(f"  n_neighbors: {best_knn_model.n_neighbors}")
    print(f"  weights: {best_knn_model.weights}")
    print(f"  metric: {best_knn_model.metric}")
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_knn_best:.6f}")
print(f"  RMSE: ${train_rmse_knn_best:,.2f}")
print(f"  MAE: ${train_mae_knn_best:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_knn_best:.6f}")
print(f"  RMSE: ${test_rmse_knn_best:,.2f}")
print(f"  MAE: ${test_mae_knn_best:,.2f}")
if not SKIP_GRID_SEARCH:
    print(f"\nImprovement over default KNN (k=5):")
    print(f"  R² improvement: {(test_r2_knn_best - test_r2_knn):.6f}")
    print(f"  RMSE improvement: ${(test_rmse_knn - test_rmse_knn_best):,.2f}")
print("=" * 70)


# %%

# Only save if we just trained the model (not if we loaded it)
if not SKIP_GRID_SEARCH:
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the best KNN model
    model_path = "models/best_knn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_knn_model, f)

    # Save the scaler as well (needed for predictions)
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Best KNN model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    print(f"\nModel parameters:")
    print(f"  k = {grid_search.best_params_['n_neighbors']}")
    print(f"  weights = {grid_search.best_params_['weights']}")
    print(f"  metric = {grid_search.best_params_['metric']}")
else:
    print("⏭️  Model already loaded from file - no need to save again")

# %%
# Visualize GridSearchCV results (only if grid search was run)
if not SKIP_GRID_SEARCH:
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    # Group by n_neighbors and calculate mean test score
    k_performance = (
        cv_results_df.groupby("param_n_neighbors")["mean_test_score"]
        .agg(["mean", "std", "max"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mean R² score by k value
    axes[0].errorbar(
        k_performance["param_n_neighbors"],
        k_performance["mean"],
        yerr=k_performance["std"],
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    axes[0].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[0].set_ylabel("Mean R² Score (5-Fold CV)", fontsize=12)
    axes[0].set_title("KNN Performance vs. k Value", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        x=grid_search.best_params_["n_neighbors"],
        color="red",
        linestyle="--",
        label=f"Best k={grid_search.best_params_['n_neighbors']}",
    )
    axes[0].legend()

    # Heatmap of performance by weights and metric
    pivot_data = cv_results_df.pivot_table(
        values="mean_test_score",
        index="param_weights",
        columns="param_metric",
        aggfunc="mean",
    )
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        ax=axes[1],
        cbar_kws={"label": "Mean R² Score"},
    )
    axes[1].set_title(
        "Mean Performance by Weights and Distance Metric",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Distance Metric", fontsize=12)
    axes[1].set_ylabel("Weights", fontsize=12)

    plt.tight_layout()
    plt.show()

    print(f"\nBest configuration found:")
    print(f"  k = {grid_search.best_params_['n_neighbors']}")
    print(f"  weights = {grid_search.best_params_['weights']}")
    print(f"  metric = {grid_search.best_params_['metric']}")
else:
    print("⏭️  Skipping grid search visualization - model was loaded from file")
    print("\nModel configuration:")
    print(f"  k = {best_knn_model.n_neighbors}")
    print(f"  weights = {best_knn_model.weights}")
    print(f"  metric = {best_knn_model.metric}")

# %%
# Visualize predictions vs actual values for KNN
plt.figure(figsize=(12, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_knn_best, alpha=0.3, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"KNN Training Set\nR² = {train_r2_knn_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_knn_best, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"KNN Testing Set\nR² = {test_r2_knn_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# %%
# Residual analysis for KNN
residuals_train_knn = y_train - y_train_pred_knn_best
residuals_test_knn = y_test - y_test_pred_knn_best

plt.figure(figsize=(12, 5))

# Residual plot for training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_knn_best, residuals_train_knn, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("KNN Training Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

# Residual plot for testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred_knn_best, residuals_test_knn, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("KNN Testing Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

print(f"\nResidual Statistics (Test Set - KNN):")
print(f"  Mean: ${residuals_test_knn.mean():,.2f}")
print(f"  Std Dev: ${residuals_test_knn.std():,.2f}")
print(f"  Min: ${residuals_test_knn.min():,.2f}")
print(f"  Max: ${residuals_test_knn.max():,.2f}")

# %% [markdown]
# ### Comparison: Linear Regression vs KNN
#
# Let's compare the performance of Linear Regression and KNN models.

# %%
# Create comprehensive comparison table
comparison_data = {
    "Model": [
        "Linear Regression",
        "Linear Regression (Stratified)",
        "Linear Regression (10-Fold CV)",
        "KNN (k=5, default)",
        "KNN (Optimized)",
    ],
    "Test R²": [
        test_r2,
        test_r2_strat,
        cv_test_r2.mean(),
        test_r2_knn,
        test_r2_knn_best,
    ],
    "Test RMSE": [
        test_rmse,
        test_rmse_strat,
        cv_test_rmse.mean(),
        test_rmse_knn,
        test_rmse_knn_best,
    ],
    "Test MAE": [
        test_mae,
        test_mae_strat,
        cv_test_mae.mean(),
        test_mae_knn,
        test_mae_knn_best,
    ],
}

comparison_df = pd.DataFrame(comparison_data)

print("=" * 90)
print("MODEL COMPARISON - LINEAR REGRESSION vs KNN")
print("=" * 90)
print(comparison_df.to_string(index=False))
print("=" * 90)

# Find best model
best_model_idx = comparison_df["Test R²"].idxmax()
print(f"\n🏆 Best Model: {comparison_df.loc[best_model_idx, 'Model']}")
print(f"   R² Score: {comparison_df.loc[best_model_idx, 'Test R²']:.6f}")
print(f"   RMSE: ${comparison_df.loc[best_model_idx, 'Test RMSE']:,.2f}")
print(f"   MAE: ${comparison_df.loc[best_model_idx, 'Test MAE']:,.2f}")

if best_model_idx == 4:  # KNN Optimized
    print(f"\n   Best Parameters:")
    print(f"   - k = {best_knn_model.n_neighbors}")
    print(f"   - weights = {best_knn_model.weights}")
    print(f"   - metric = {best_knn_model.metric}")

# %%
# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models = comparison_df["Model"].tolist()
colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]

# R² comparison
axes[0].barh(range(len(models)), comparison_df["Test R²"], color=colors)
axes[0].set_yticks(range(len(models)))
axes[0].set_yticklabels(models)
axes[0].set_xlabel("R² Score", fontsize=12)
axes[0].set_title("R² Score Comparison", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(comparison_df["Test R²"]):
    axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

# RMSE comparison
axes[1].barh(range(len(models)), comparison_df["Test RMSE"], color=colors)
axes[1].set_yticks(range(len(models)))
axes[1].set_yticklabels(models)
axes[1].set_xlabel("RMSE ($)", fontsize=12)
axes[1].set_title("RMSE Comparison (Lower is Better)", fontsize=14, fontweight="bold")
axes[1].ticklabel_format(style="plain", axis="x")
axes[1].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(comparison_df["Test RMSE"]):
    axes[1].text(v + 5000, i, f"${v:,.0f}", va="center", fontsize=9)

# MAE comparison
axes[2].barh(range(len(models)), comparison_df["Test MAE"], color=colors)
axes[2].set_yticks(range(len(models)))
axes[2].set_yticklabels(models)
axes[2].set_xlabel("MAE ($)", fontsize=12)
axes[2].set_title("MAE Comparison (Lower is Better)", fontsize=14, fontweight="bold")
axes[2].ticklabel_format(style="plain", axis="x")
axes[2].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(comparison_df["Test MAE"]):
    axes[2].text(v + 3000, i, f"${v:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Decision Tree Regression
#
# Now let's implement Decision Tree Regressor and compare it with the previous models.

# %%

# Train a basic Decision Tree model with default parameters
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_dt = dt_model.predict(X_train_scaled)
y_test_pred_dt = dt_model.predict(X_test_scaled)

# Calculate metrics
train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)
train_rmse_dt = np.sqrt(mean_squared_error(y_train, y_train_pred_dt))
test_rmse_dt = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
train_mae_dt = mean_absolute_error(y_train, y_train_pred_dt)
test_mae_dt = mean_absolute_error(y_test, y_test_pred_dt)

print("=" * 70)
print("DECISION TREE REGRESSION MODEL PERFORMANCE (Default Parameters)")
print("=" * 70)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_dt:.6f}")
print(f"  RMSE: ${train_rmse_dt:,.2f}")
print(f"  MAE: ${train_mae_dt:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_dt:.6f}")
print(f"  RMSE: ${test_rmse_dt:,.2f}")
print(f"  MAE: ${test_mae_dt:,.2f}")
print("\n" + "=" * 70)

# %% [markdown]
# ### Hyperparameter Tuning for Decision Tree
#
# Let's optimize the Decision Tree using GridSearchCV to find the best hyperparameters.

# %%
# Define parameter grid for Decision Tree
dt_param_grid = {
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

# Create GridSearchCV object
dt_grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

print("Performing GridSearchCV for Decision Tree hyperparameter tuning...")
print("This may take a few minutes...\n")

# Fit the grid search
dt_grid_search.fit(X_train_scaled, y_train)

print("\n" + "=" * 70)
print("GRIDSEARCHCV RESULTS - DECISION TREE")
print("=" * 70)
print(f"\nBest parameters: {dt_grid_search.best_params_}")
print(f"Best cross-validation R² score: {dt_grid_search.best_score_:.6f}")
print("=" * 70)

# %%
# Get the best model
best_dt_model = dt_grid_search.best_estimator_

# Make predictions with the best model
y_train_pred_dt_best = best_dt_model.predict(X_train_scaled)
y_test_pred_dt_best = best_dt_model.predict(X_test_scaled)

# Calculate metrics
train_r2_dt_best = r2_score(y_train, y_train_pred_dt_best)
test_r2_dt_best = r2_score(y_test, y_test_pred_dt_best)
train_rmse_dt_best = np.sqrt(mean_squared_error(y_train, y_train_pred_dt_best))
test_rmse_dt_best = np.sqrt(mean_squared_error(y_test, y_test_pred_dt_best))
train_mae_dt_best = mean_absolute_error(y_train, y_train_pred_dt_best)
test_mae_dt_best = mean_absolute_error(y_test, y_test_pred_dt_best)

print("=" * 70)
print("OPTIMIZED DECISION TREE REGRESSION MODEL PERFORMANCE")
print("=" * 70)
print(f"\nBest Parameters:")
for param, value in dt_grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_dt_best:.6f}")
print(f"  RMSE: ${train_rmse_dt_best:,.2f}")
print(f"  MAE: ${train_mae_dt_best:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_dt_best:.6f}")
print(f"  RMSE: ${test_rmse_dt_best:,.2f}")
print(f"  MAE: ${test_mae_dt_best:,.2f}")
print(f"\nImprovement over default Decision Tree:")
print(f"  R² improvement: {(test_r2_dt_best - test_r2_dt):.6f}")
print(f"  RMSE improvement: ${(test_rmse_dt - test_rmse_dt_best):,.2f}")
print("=" * 70)

# %%
# Analyze feature importance for Decision Tree
feature_importance_dt = pd.DataFrame(
    {"Feature": feature_columns, "Importance": best_dt_model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Most Important Features (Decision Tree):")
print(feature_importance_dt.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features_dt = feature_importance_dt.head(15)
colors_dt = plt.cm.viridis(np.linspace(0, 1, len(top_features_dt)))
plt.barh(range(len(top_features_dt)), top_features_dt["Importance"], color=colors_dt)
plt.yticks(range(len(top_features_dt)), top_features_dt["Feature"])
plt.xlabel("Feature Importance", fontsize=12)
plt.title(
    "Top 15 Feature Importances in Decision Tree Model", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

# %%
# Visualize predictions vs actual values for Decision Tree
plt.figure(figsize=(12, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred_dt_best, alpha=0.3, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Decision Tree Training Set\nR² = {train_r2_dt_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred_dt_best, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Decision Tree Testing Set\nR² = {test_r2_dt_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# %%
# Residual analysis for Decision Tree
residuals_train_dt = y_train - y_train_pred_dt_best
residuals_test_dt = y_test - y_test_pred_dt_best

plt.figure(figsize=(12, 5))

# Residual plot for training set
plt.subplot(1, 2, 1)
plt.scatter(y_train_pred_dt_best, residuals_train_dt, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Decision Tree Training Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

# Residual plot for testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred_dt_best, residuals_test_dt, alpha=0.3, s=1)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Decision Tree Testing Set Residuals")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

print(f"\nResidual Statistics (Test Set - Decision Tree):")
print(f"  Mean: ${residuals_test_dt.mean():,.2f}")
print(f"  Std Dev: ${residuals_test_dt.std():,.2f}")
print(f"  Min: ${residuals_test_dt.min():,.2f}")
print(f"  Max: ${residuals_test_dt.max():,.2f}")

# %% [markdown]
# ### Final Comparison: All Models
#
# Let's create a comprehensive comparison of all models including Decision Tree.

# %%
# Create comprehensive comparison table including Decision Tree
final_comparison_data = {
    "Model": [
        "Linear Regression",
        "Linear Regression (Stratified)",
        "Linear Regression (10-Fold CV)",
        "KNN (k=5, default)",
        "KNN (Optimized)",
        "Decision Tree (default)",
        "Decision Tree (Optimized)",
    ],
    "Test R²": [
        test_r2,
        test_r2_strat,
        cv_test_r2.mean(),
        test_r2_knn,
        test_r2_knn_best,
        test_r2_dt,
        test_r2_dt_best,
    ],
    "Test RMSE": [
        test_rmse,
        test_rmse_strat,
        cv_test_rmse.mean(),
        test_rmse_knn,
        test_rmse_knn_best,
        test_rmse_dt,
        test_rmse_dt_best,
    ],
    "Test MAE": [
        test_mae,
        test_mae_strat,
        cv_test_mae.mean(),
        test_mae_knn,
        test_mae_knn_best,
        test_mae_dt,
        test_mae_dt_best,
    ],
}

final_comparison_df = pd.DataFrame(final_comparison_data)

print("=" * 100)
print("FINAL MODEL COMPARISON - ALL MODELS")
print("=" * 100)
print(final_comparison_df.to_string(index=False))
print("=" * 100)

# Find best model
best_final_idx = final_comparison_df["Test R²"].idxmax()
print(f"\n🏆 BEST OVERALL MODEL: {final_comparison_df.loc[best_final_idx, 'Model']}")
print(f"   R² Score: {final_comparison_df.loc[best_final_idx, 'Test R²']:.6f}")
print(f"   RMSE: ${final_comparison_df.loc[best_final_idx, 'Test RMSE']:,.2f}")
print(f"   MAE: ${final_comparison_df.loc[best_final_idx, 'Test MAE']:,.2f}")

if best_final_idx == 4:  # KNN Optimized
    print(f"\n   Best Parameters:")
    print(f"   - k = {best_knn_model.n_neighbors}")
    print(f"   - weights = {best_knn_model.weights}")
    print(f"   - metric = {best_knn_model.metric}")
elif best_final_idx == 6:  # Decision Tree Optimized
    print(f"\n   Best Parameters:")
    for param, value in dt_grid_search.best_params_.items():
        print(f"   - {param} = {value}")

# %%
# Visualize final model comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

models_final = final_comparison_df["Model"].tolist()
colors_final = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

# R² comparison
axes[0].barh(
    range(len(models_final)), final_comparison_df["Test R²"], color=colors_final
)
axes[0].set_yticks(range(len(models_final)))
axes[0].set_yticklabels(models_final, fontsize=10)
axes[0].set_xlabel("R² Score", fontsize=12)
axes[0].set_title("R² Score Comparison - All Models", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_comparison_df["Test R²"]):
    axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

# RMSE comparison
axes[1].barh(
    range(len(models_final)), final_comparison_df["Test RMSE"], color=colors_final
)
axes[1].set_yticks(range(len(models_final)))
axes[1].set_yticklabels(models_final, fontsize=10)
axes[1].set_xlabel("RMSE ($)", fontsize=12)
axes[1].set_title(
    "RMSE Comparison - All Models (Lower is Better)", fontsize=14, fontweight="bold"
)
axes[1].ticklabel_format(style="plain", axis="x")
axes[1].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_comparison_df["Test RMSE"]):
    axes[1].text(v + 3000, i, f"${v:,.0f}", va="center", fontsize=9)

# MAE comparison
axes[2].barh(
    range(len(models_final)), final_comparison_df["Test MAE"], color=colors_final
)
axes[2].set_yticks(range(len(models_final)))
axes[2].set_yticklabels(models_final, fontsize=10)
axes[2].set_xlabel("MAE ($)", fontsize=12)
axes[2].set_title(
    "MAE Comparison - All Models (Lower is Better)", fontsize=14, fontweight="bold"
)
axes[2].ticklabel_format(style="plain", axis="x")
axes[2].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_comparison_df["Test MAE"]):
    axes[2].text(v + 2000, i, f"${v:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

# %%
# Create a summary comparison of model types
model_type_comparison = {
    "Model Type": ["Linear Regression", "KNN", "Decision Tree"],
    "Best Test R²": [
        max(test_r2, test_r2_strat, cv_test_r2.mean()),
        max(test_r2_knn, test_r2_knn_best),
        max(test_r2_dt, test_r2_dt_best),
    ],
    "Best Test RMSE": [
        min(test_rmse, test_rmse_strat, cv_test_rmse.mean()),
        min(test_rmse_knn, test_rmse_knn_best),
        min(test_rmse_dt, test_rmse_dt_best),
    ],
    "Best Test MAE": [
        min(test_mae, test_mae_strat, cv_test_mae.mean()),
        min(test_mae_knn, test_mae_knn_best),
        min(test_mae_dt, test_mae_dt_best),
    ],
}

model_type_df = pd.DataFrame(model_type_comparison)

print("\n" + "=" * 100)
print("MODEL TYPE COMPARISON (Best variant of each)")
print("=" * 100)
print(model_type_df.to_string(index=False))
print("=" * 100)

# Visualize model type comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

type_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# R² comparison
axes[0].bar(
    model_type_df["Model Type"], model_type_df["Best Test R²"], color=type_colors
)
axes[0].set_ylabel("R² Score", fontsize=12)
axes[0].set_title("Best R² Score by Model Type", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(model_type_df["Best Test R²"]):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

# RMSE comparison
axes[1].bar(
    model_type_df["Model Type"], model_type_df["Best Test RMSE"], color=type_colors
)
axes[1].set_ylabel("RMSE ($)", fontsize=12)
axes[1].set_title(
    "Best RMSE by Model Type (Lower is Better)", fontsize=14, fontweight="bold"
)
axes[1].ticklabel_format(style="plain", axis="y")
axes[1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(model_type_df["Best Test RMSE"]):
    axes[1].text(i, v + 3000, f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")

# MAE comparison
axes[2].bar(
    model_type_df["Model Type"], model_type_df["Best Test MAE"], color=type_colors
)
axes[2].set_ylabel("MAE ($)", fontsize=12)
axes[2].set_title(
    "Best MAE by Model Type (Lower is Better)", fontsize=14, fontweight="bold"
)
axes[2].ticklabel_format(style="plain", axis="y")
axes[2].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(model_type_df["Best Test MAE"]):
    axes[2].text(i, v + 2000, f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison with Price-Related Features
#
# Now let's compare how the models perform when they have access to price-related data. We'll add "Median Rent" as an additional feature and retrain all models to see how this affects performance.

# %%
# Merge datasets with Median Rent included
df_merged_rent = df1.merge(
    df2[
        [
            "zipcode",
            "bank",
            "bus",
            "hospital",
            "mall",
            "park",
            "restaurant",
            "school",
            "station",
            "supermarket",
            "Total Population",
            "Median Age",
            "Per Capita Income",
            "Total Families Below Poverty",
            "Total Housing Units",
            "Total Labor Force",
            "Unemployed Population",
            "Total School Age Population",
            "Total School Enrollment",
            "Median Commute Time",
            "Median Rent",
        ]
    ].drop_duplicates(subset="zipcode"),
    left_on="zip_code",
    right_on="zipcode",
    how="inner",
)

print(f"Original df1 shape: {df1.shape}")
print(f"Merged dataframe with Median Rent shape: {df_merged_rent.shape}")
print(f"\nMissing values:\n{df_merged_rent.isnull().sum()}")

# Prepare features including Median Rent
feature_columns_rent = [
    # House characteristics
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    # Amenities
    "bank",
    "bus",
    "hospital",
    "mall",
    "park",
    "restaurant",
    "school",
    "station",
    "supermarket",
    # Demographics
    "Total Population",
    "Median Age",
    "Per Capita Income",
    "Total Families Below Poverty",
    "Total Housing Units",
    "Total Labor Force",
    "Unemployed Population",
    "Total School Age Population",
    "Total School Enrollment",
    "Median Commute Time",
    # Price-related feature
    "Median Rent",
]

X_rent = df_merged_rent[feature_columns_rent]
y_rent = df_merged_rent["price"]

print(f"\nFeatures shape: {X_rent.shape}")
print(f"Target shape: {y_rent.shape}")
print(
    f"\n✨ NEW: Added 'Median Rent' feature (total features: {len(feature_columns_rent)})"
)
print(f"Previous feature count: {len(feature_columns)}")

# %%
# Split data and standardize features
X_train_rent, X_test_rent, y_train_rent, y_test_rent = train_test_split(
    X_rent, y_rent, test_size=0.2, random_state=42
)

scaler_rent = StandardScaler()
X_train_rent_scaled = scaler_rent.fit_transform(X_train_rent)
X_test_rent_scaled = scaler_rent.transform(X_test_rent)

print(f"Training set size: {X_train_rent.shape[0]}")
print(f"Testing set size: {X_test_rent.shape[0]}")
print("Features standardized successfully")

# %% [markdown]
# ### Linear Regression with Median Rent

# %%
# Train Linear Regression model with Median Rent
lr_model_rent = LinearRegression()
lr_model_rent.fit(X_train_rent_scaled, y_train_rent)

# Make predictions
y_train_pred_lr_rent = lr_model_rent.predict(X_train_rent_scaled)
y_test_pred_lr_rent = lr_model_rent.predict(X_test_rent_scaled)

# Calculate metrics
train_r2_lr_rent = r2_score(y_train_rent, y_train_pred_lr_rent)
test_r2_lr_rent = r2_score(y_test_rent, y_test_pred_lr_rent)
train_rmse_lr_rent = np.sqrt(mean_squared_error(y_train_rent, y_train_pred_lr_rent))
test_rmse_lr_rent = np.sqrt(mean_squared_error(y_test_rent, y_test_pred_lr_rent))
train_mae_lr_rent = mean_absolute_error(y_train_rent, y_train_pred_lr_rent)
test_mae_lr_rent = mean_absolute_error(y_test_rent, y_test_pred_lr_rent)

print("=" * 70)
print("LINEAR REGRESSION WITH MEDIAN RENT")
print("=" * 70)
print(f"\nTraining Set:")
print(f"  R² Score: {train_r2_lr_rent:.6f}")
print(f"  RMSE: ${train_rmse_lr_rent:,.2f}")
print(f"  MAE: ${train_mae_lr_rent:,.2f}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_lr_rent:.6f}")
print(f"  RMSE: ${test_rmse_lr_rent:,.2f}")
print(f"  MAE: ${test_mae_lr_rent:,.2f}")
print(f"\nImprovement over original Linear Regression:")
print(f"  R² improvement: {(test_r2_lr_rent - test_r2):.6f}")
print(f"  RMSE improvement: ${(test_rmse - test_rmse_lr_rent):,.2f}")
print("=" * 70)

# %%
# Stratified sampling with Median Rent
n_bins = 10
stratify_labels_rent = pd.qcut(y_rent, q=n_bins, labels=False, duplicates="drop")

X_train_strat_rent, X_test_strat_rent, y_train_strat_rent, y_test_strat_rent = (
    train_test_split(
        X_rent, y_rent, test_size=0.2, random_state=42, stratify=stratify_labels_rent
    )
)

scaler_strat_rent = StandardScaler()
X_train_strat_rent_scaled = scaler_strat_rent.fit_transform(X_train_strat_rent)
X_test_strat_rent_scaled = scaler_strat_rent.transform(X_test_strat_rent)

lr_strat_rent = LinearRegression()
lr_strat_rent.fit(X_train_strat_rent_scaled, y_train_strat_rent)

y_train_pred_strat_rent = lr_strat_rent.predict(X_train_strat_rent_scaled)
y_test_pred_strat_rent = lr_strat_rent.predict(X_test_strat_rent_scaled)

train_r2_strat_rent = r2_score(y_train_strat_rent, y_train_pred_strat_rent)
test_r2_strat_rent = r2_score(y_test_strat_rent, y_test_pred_strat_rent)
train_rmse_strat_rent = np.sqrt(
    mean_squared_error(y_train_strat_rent, y_train_pred_strat_rent)
)
test_rmse_strat_rent = np.sqrt(
    mean_squared_error(y_test_strat_rent, y_test_pred_strat_rent)
)
train_mae_strat_rent = mean_absolute_error(y_train_strat_rent, y_train_pred_strat_rent)
test_mae_strat_rent = mean_absolute_error(y_test_strat_rent, y_test_pred_strat_rent)

print("=" * 70)
print("STRATIFIED SAMPLING WITH MEDIAN RENT")
print("=" * 70)
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_strat_rent:.6f}")
print(f"  RMSE: ${test_rmse_strat_rent:,.2f}")
print(f"  MAE: ${test_mae_strat_rent:,.2f}")
print(f"\nImprovement over original stratified:")
print(f"  R² improvement: {(test_r2_strat_rent - test_r2_strat):.6f}")
print(f"  RMSE improvement: ${(test_rmse_strat - test_rmse_strat_rent):,.2f}")
print("=" * 70)

# %%
# K-Fold Cross-Validation with Median Rent
pipeline_rent = Pipeline(
    [("scaler", StandardScaler()), ("regressor", LinearRegression())]
)

k = 10
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"Performing {k}-fold cross-validation with Median Rent...")

scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
cv_results_rent = cross_validate(
    pipeline_rent,
    X_rent,
    y_rent,
    cv=kfold,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,
)

cv_test_r2_rent = cv_results_rent["test_r2"]
cv_test_rmse_rent = np.sqrt(-cv_results_rent["test_neg_mean_squared_error"])
cv_test_mae_rent = -cv_results_rent["test_neg_mean_absolute_error"]

print("=" * 70)
print(f"{k}-FOLD CV WITH MEDIAN RENT")
print("=" * 70)
print(
    f"\nValidation R²: {cv_test_r2_rent.mean():.6f} (+/- {cv_test_r2_rent.std():.6f})"
)
print(
    f"Validation RMSE: ${cv_test_rmse_rent.mean():,.2f} (+/- ${cv_test_rmse_rent.std():,.2f})"
)
print(
    f"Validation MAE: ${cv_test_mae_rent.mean():,.2f} (+/- ${cv_test_mae_rent.std():,.2f})"
)
print(f"\nImprovement over original CV:")
print(f"  R² improvement: {(cv_test_r2_rent.mean() - cv_test_r2.mean()):.6f}")
print(f"  RMSE improvement: ${(cv_test_rmse.mean() - cv_test_rmse_rent.mean()):,.2f}")
print("=" * 70)

# %% [markdown]
# ### KNN Regression with Median Rent

# %%
# Train basic KNN with Median Rent
knn_model_rent = KNeighborsRegressor(n_neighbors=5)
knn_model_rent.fit(X_train_rent_scaled, y_train_rent)

y_train_pred_knn_rent = knn_model_rent.predict(X_train_rent_scaled)
y_test_pred_knn_rent = knn_model_rent.predict(X_test_rent_scaled)

train_r2_knn_rent = r2_score(y_train_rent, y_train_pred_knn_rent)
test_r2_knn_rent = r2_score(y_test_rent, y_test_pred_knn_rent)
train_rmse_knn_rent = np.sqrt(mean_squared_error(y_train_rent, y_train_pred_knn_rent))
test_rmse_knn_rent = np.sqrt(mean_squared_error(y_test_rent, y_test_pred_knn_rent))
train_mae_knn_rent = mean_absolute_error(y_train_rent, y_train_pred_knn_rent)
test_mae_knn_rent = mean_absolute_error(y_test_rent, y_test_pred_knn_rent)

print("=" * 70)
print("KNN (k=5) WITH MEDIAN RENT")
print("=" * 70)
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_knn_rent:.6f}")
print(f"  RMSE: ${test_rmse_knn_rent:,.2f}")
print(f"  MAE: ${test_mae_knn_rent:,.2f}")
print(f"\nImprovement over original KNN (k=5):")
print(f"  R² improvement: {(test_r2_knn_rent - test_r2_knn):.6f}")
print(f"  RMSE improvement: ${(test_rmse_knn - test_rmse_knn_rent):,.2f}")
print("=" * 70)

# %%
# GridSearchCV for KNN with Median Rent
param_grid = {
    "n_neighbors": [3, 5, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

grid_search_rent = GridSearchCV(
    KNeighborsRegressor(), param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
)

print("Performing GridSearchCV for KNN with Median Rent...")
print("This may take a few minutes...\n")

grid_search_rent.fit(X_train_rent_scaled, y_train_rent)

print("\n" + "=" * 70)
print("GRIDSEARCHCV RESULTS - KNN WITH MEDIAN RENT")
print("=" * 70)
print(f"\nBest parameters: {grid_search_rent.best_params_}")
print(f"Best cross-validation R² score: {grid_search_rent.best_score_:.6f}")
print("=" * 70)

# %%
# Evaluate best KNN model with Median Rent
best_knn_model_rent = grid_search_rent.best_estimator_

y_train_pred_knn_best_rent = best_knn_model_rent.predict(X_train_rent_scaled)
y_test_pred_knn_best_rent = best_knn_model_rent.predict(X_test_rent_scaled)

train_r2_knn_best_rent = r2_score(y_train_rent, y_train_pred_knn_best_rent)
test_r2_knn_best_rent = r2_score(y_test_rent, y_test_pred_knn_best_rent)
train_rmse_knn_best_rent = np.sqrt(
    mean_squared_error(y_train_rent, y_train_pred_knn_best_rent)
)
test_rmse_knn_best_rent = np.sqrt(
    mean_squared_error(y_test_rent, y_test_pred_knn_best_rent)
)
train_mae_knn_best_rent = mean_absolute_error(y_train_rent, y_train_pred_knn_best_rent)
test_mae_knn_best_rent = mean_absolute_error(y_test_rent, y_test_pred_knn_best_rent)

print("=" * 70)
print("OPTIMIZED KNN WITH MEDIAN RENT")
print("=" * 70)
print(f"\nBest Parameters: {grid_search_rent.best_params_}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_knn_best_rent:.6f}")
print(f"  RMSE: ${test_rmse_knn_best_rent:,.2f}")
print(f"  MAE: ${test_mae_knn_best_rent:,.2f}")
print(f"\nImprovement over original optimized KNN:")
print(f"  R² improvement: {(test_r2_knn_best_rent - test_r2_knn_best):.6f}")
print(f"  RMSE improvement: ${(test_rmse_knn_best - test_rmse_knn_best_rent):,.2f}")
print("=" * 70)

# %%
# Save the KNN model with Median Rent (with different name)
os.makedirs("models", exist_ok=True)

model_path_rent = "models/best_knn_model_with_rent.pkl"
with open(model_path_rent, "wb") as f:
    pickle.dump(best_knn_model_rent, f)

scaler_path_rent = "models/scaler_with_rent.pkl"
with open(scaler_path_rent, "wb") as f:
    pickle.dump(scaler_rent, f)

print(f"✅ Best KNN model (with Median Rent) saved to: {model_path_rent}")
print(f"✅ Scaler (with Median Rent) saved to: {scaler_path_rent}")
print(f"\nModel parameters:")
print(f"  k = {grid_search_rent.best_params_['n_neighbors']}")
print(f"  weights = {grid_search_rent.best_params_['weights']}")
print(f"  metric = {grid_search_rent.best_params_['metric']}")

# %% [markdown]
# ### Decision Tree Regression with Median Rent

# %%
# Train basic Decision Tree with Median Rent
dt_model_rent = DecisionTreeRegressor(random_state=42)
dt_model_rent.fit(X_train_rent_scaled, y_train_rent)

y_train_pred_dt_rent = dt_model_rent.predict(X_train_rent_scaled)
y_test_pred_dt_rent = dt_model_rent.predict(X_test_rent_scaled)

train_r2_dt_rent = r2_score(y_train_rent, y_train_pred_dt_rent)
test_r2_dt_rent = r2_score(y_test_rent, y_test_pred_dt_rent)
train_rmse_dt_rent = np.sqrt(mean_squared_error(y_train_rent, y_train_pred_dt_rent))
test_rmse_dt_rent = np.sqrt(mean_squared_error(y_test_rent, y_test_pred_dt_rent))
train_mae_dt_rent = mean_absolute_error(y_train_rent, y_train_pred_dt_rent)
test_mae_dt_rent = mean_absolute_error(y_test_rent, y_test_pred_dt_rent)

print("=" * 70)
print("DECISION TREE (Default) WITH MEDIAN RENT")
print("=" * 70)
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_dt_rent:.6f}")
print(f"  RMSE: ${test_rmse_dt_rent:,.2f}")
print(f"  MAE: ${test_mae_dt_rent:,.2f}")
print(f"\nImprovement over original Decision Tree:")
print(f"  R² improvement: {(test_r2_dt_rent - test_r2_dt):.6f}")
print(f"  RMSE improvement: ${(test_rmse_dt - test_rmse_dt_rent):,.2f}")
print("=" * 70)

# %%
# GridSearchCV for Decision Tree with Median Rent
dt_param_grid = {
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

dt_grid_search_rent = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

print("Performing GridSearchCV for Decision Tree with Median Rent...")
print("This may take a few minutes...\n")

dt_grid_search_rent.fit(X_train_rent_scaled, y_train_rent)

print("\n" + "=" * 70)
print("GRIDSEARCHCV RESULTS - DECISION TREE WITH MEDIAN RENT")
print("=" * 70)
print(f"\nBest parameters: {dt_grid_search_rent.best_params_}")
print(f"Best cross-validation R² score: {dt_grid_search_rent.best_score_:.6f}")
print("=" * 70)

# %%
# Evaluate best Decision Tree model with Median Rent
best_dt_model_rent = dt_grid_search_rent.best_estimator_

y_train_pred_dt_best_rent = best_dt_model_rent.predict(X_train_rent_scaled)
y_test_pred_dt_best_rent = best_dt_model_rent.predict(X_test_rent_scaled)

train_r2_dt_best_rent = r2_score(y_train_rent, y_train_pred_dt_best_rent)
test_r2_dt_best_rent = r2_score(y_test_rent, y_test_pred_dt_best_rent)
train_rmse_dt_best_rent = np.sqrt(
    mean_squared_error(y_train_rent, y_train_pred_dt_best_rent)
)
test_rmse_dt_best_rent = np.sqrt(
    mean_squared_error(y_test_rent, y_test_pred_dt_best_rent)
)
train_mae_dt_best_rent = mean_absolute_error(y_train_rent, y_train_pred_dt_best_rent)
test_mae_dt_best_rent = mean_absolute_error(y_test_rent, y_test_pred_dt_best_rent)

print("=" * 70)
print("OPTIMIZED DECISION TREE WITH MEDIAN RENT")
print("=" * 70)
print(f"\nBest Parameters:")
for param, value in dt_grid_search_rent.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nTesting Set:")
print(f"  R² Score: {test_r2_dt_best_rent:.6f}")
print(f"  RMSE: ${test_rmse_dt_best_rent:,.2f}")
print(f"  MAE: ${test_mae_dt_best_rent:,.2f}")
print(f"\nImprovement over original optimized Decision Tree:")
print(f"  R² improvement: {(test_r2_dt_best_rent - test_r2_dt_best):.6f}")
print(f"  RMSE improvement: ${(test_rmse_dt_best - test_rmse_dt_best_rent):,.2f}")
print("=" * 70)

# %% [markdown]
# ### Final Comparison: With vs Without Median Rent
#
# Let's create comprehensive comparisons showing the impact of adding the Median Rent feature.

# %%
# Create comprehensive comparison table
comparison_with_rent = {
    "Model": [
        "Linear Regression",
        "Linear Regression (Stratified)",
        "Linear Regression (10-Fold CV)",
        "KNN (k=5, default)",
        "KNN (Optimized)",
        "Decision Tree (default)",
        "Decision Tree (Optimized)",
    ],
    "Without Median Rent - R²": [
        test_r2,
        test_r2_strat,
        cv_test_r2.mean(),
        test_r2_knn,
        test_r2_knn_best,
        test_r2_dt,
        test_r2_dt_best,
    ],
    "With Median Rent - R²": [
        test_r2_lr_rent,
        test_r2_strat_rent,
        cv_test_r2_rent.mean(),
        test_r2_knn_rent,
        test_r2_knn_best_rent,
        test_r2_dt_rent,
        test_r2_dt_best_rent,
    ],
    "R² Improvement": [
        test_r2_lr_rent - test_r2,
        test_r2_strat_rent - test_r2_strat,
        cv_test_r2_rent.mean() - cv_test_r2.mean(),
        test_r2_knn_rent - test_r2_knn,
        test_r2_knn_best_rent - test_r2_knn_best,
        test_r2_dt_rent - test_r2_dt,
        test_r2_dt_best_rent - test_r2_dt_best,
    ],
    "Without Median Rent - RMSE": [
        test_rmse,
        test_rmse_strat,
        cv_test_rmse.mean(),
        test_rmse_knn,
        test_rmse_knn_best,
        test_rmse_dt,
        test_rmse_dt_best,
    ],
    "With Median Rent - RMSE": [
        test_rmse_lr_rent,
        test_rmse_strat_rent,
        cv_test_rmse_rent.mean(),
        test_rmse_knn_rent,
        test_rmse_knn_best_rent,
        test_rmse_dt_rent,
        test_rmse_dt_best_rent,
    ],
    "RMSE Improvement": [
        test_rmse - test_rmse_lr_rent,
        test_rmse_strat - test_rmse_strat_rent,
        cv_test_rmse.mean() - cv_test_rmse_rent.mean(),
        test_rmse_knn - test_rmse_knn_rent,
        test_rmse_knn_best - test_rmse_knn_best_rent,
        test_rmse_dt - test_rmse_dt_rent,
        test_rmse_dt_best - test_rmse_dt_best_rent,
    ],
}

comparison_with_rent_df = pd.DataFrame(comparison_with_rent)

print("=" * 120)
print("IMPACT OF ADDING MEDIAN RENT FEATURE")
print("=" * 120)
print("\nR² Scores:")
print("-" * 120)
print(
    comparison_with_rent_df[
        ["Model", "Without Median Rent - R²", "With Median Rent - R²", "R² Improvement"]
    ].to_string(index=False)
)
print("\n" + "=" * 120)
print("\nRMSE Values:")
print("-" * 120)
print(
    comparison_with_rent_df[
        [
            "Model",
            "Without Median Rent - RMSE",
            "With Median Rent - RMSE",
            "RMSE Improvement",
        ]
    ].to_string(index=False)
)
print("=" * 120)

# Calculate average improvement
avg_r2_improvement = comparison_with_rent_df["R² Improvement"].mean()
avg_rmse_improvement = comparison_with_rent_df["RMSE Improvement"].mean()

print(f"\n📊 AVERAGE IMPROVEMENTS ACROSS ALL MODELS:")
print(f"   Average R² improvement: {avg_r2_improvement:.6f}")
print(f"   Average RMSE improvement: ${avg_rmse_improvement:,.2f}")
print(f"\n🏆 BEST MODEL WITH MEDIAN RENT:")
best_rent_idx = comparison_with_rent_df["With Median Rent - R²"].idxmax()
print(f"   Model: {comparison_with_rent_df.loc[best_rent_idx, 'Model']}")
print(
    f"   R² Score: {comparison_with_rent_df.loc[best_rent_idx, 'With Median Rent - R²']:.6f}"
)
print(
    f"   RMSE: ${comparison_with_rent_df.loc[best_rent_idx, 'With Median Rent - RMSE']:,.2f}"
)

# %%
# Visualize the impact of adding Median Rent
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

models_list = comparison_with_rent_df["Model"].tolist()
colors_without = "#3498db"
colors_with = "#2ecc71"

# 1. R² Score Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(models_list))
width = 0.35
bars1 = ax1.barh(
    x_pos - width / 2,
    comparison_with_rent_df["Without Median Rent - R²"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars2 = ax1.barh(
    x_pos + width / 2,
    comparison_with_rent_df["With Median Rent - R²"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(models_list, fontsize=9)
ax1.set_xlabel("R² Score", fontsize=11)
ax1.set_title("R² Score: With vs Without Median Rent", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3, axis="x")

# 2. RMSE Comparison
ax2 = axes[0, 1]
bars3 = ax2.barh(
    x_pos - width / 2,
    comparison_with_rent_df["Without Median Rent - RMSE"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars4 = ax2.barh(
    x_pos + width / 2,
    comparison_with_rent_df["With Median Rent - RMSE"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(models_list, fontsize=9)
ax2.set_xlabel("RMSE ($)", fontsize=11)
ax2.set_title(
    "RMSE: With vs Without Median Rent (Lower is Better)",
    fontsize=13,
    fontweight="bold",
)
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3, axis="x")
ax2.ticklabel_format(style="plain", axis="x")

# 3. R² Improvement
ax3 = axes[1, 0]
improvement_colors = [
    "#2ecc71" if x > 0 else "#e74c3c" for x in comparison_with_rent_df["R² Improvement"]
]
bars5 = ax3.barh(
    x_pos,
    comparison_with_rent_df["R² Improvement"],
    color=improvement_colors,
    alpha=0.8,
)
ax3.set_yticks(x_pos)
ax3.set_yticklabels(models_list, fontsize=9)
ax3.set_xlabel("R² Improvement", fontsize=11)
ax3.set_title("R² Improvement from Adding Median Rent", fontsize=13, fontweight="bold")
ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
ax3.grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(comparison_with_rent_df["R² Improvement"]):
    ax3.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

# 4. RMSE Improvement
ax4 = axes[1, 1]
rmse_improvement_colors = [
    "#2ecc71" if x > 0 else "#e74c3c"
    for x in comparison_with_rent_df["RMSE Improvement"]
]
bars6 = ax4.barh(
    x_pos,
    comparison_with_rent_df["RMSE Improvement"],
    color=rmse_improvement_colors,
    alpha=0.8,
)
ax4.set_yticks(x_pos)
ax4.set_yticklabels(models_list, fontsize=9)
ax4.set_xlabel("RMSE Improvement ($)", fontsize=11)
ax4.set_title(
    "RMSE Improvement from Adding Median Rent", fontsize=13, fontweight="bold"
)
ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
ax4.grid(True, alpha=0.3, axis="x")
ax4.ticklabel_format(style="plain", axis="x")
# Add value labels
for i, v in enumerate(comparison_with_rent_df["RMSE Improvement"]):
    ax4.text(v + 500, i, f"${v:,.0f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %%
# Summary by model type
model_type_with_rent = {
    "Model Type": ["Linear Regression", "KNN", "Decision Tree"],
    "Best R² (Without)": [
        max(test_r2, test_r2_strat, cv_test_r2.mean()),
        max(test_r2_knn, test_r2_knn_best),
        max(test_r2_dt, test_r2_dt_best),
    ],
    "Best R² (With)": [
        max(test_r2_lr_rent, test_r2_strat_rent, cv_test_r2_rent.mean()),
        max(test_r2_knn_rent, test_r2_knn_best_rent),
        max(test_r2_dt_rent, test_r2_dt_best_rent),
    ],
    "R² Improvement": [
        max(test_r2_lr_rent, test_r2_strat_rent, cv_test_r2_rent.mean())
        - max(test_r2, test_r2_strat, cv_test_r2.mean()),
        max(test_r2_knn_rent, test_r2_knn_best_rent)
        - max(test_r2_knn, test_r2_knn_best),
        max(test_r2_dt_rent, test_r2_dt_best_rent) - max(test_r2_dt, test_r2_dt_best),
    ],
    "Best RMSE (Without)": [
        min(test_rmse, test_rmse_strat, cv_test_rmse.mean()),
        min(test_rmse_knn, test_rmse_knn_best),
        min(test_rmse_dt, test_rmse_dt_best),
    ],
    "Best RMSE (With)": [
        min(test_rmse_lr_rent, test_rmse_strat_rent, cv_test_rmse_rent.mean()),
        min(test_rmse_knn_rent, test_rmse_knn_best_rent),
        min(test_rmse_dt_rent, test_rmse_dt_best_rent),
    ],
    "RMSE Improvement": [
        min(test_rmse, test_rmse_strat, cv_test_rmse.mean())
        - min(test_rmse_lr_rent, test_rmse_strat_rent, cv_test_rmse_rent.mean()),
        min(test_rmse_knn, test_rmse_knn_best)
        - min(test_rmse_knn_rent, test_rmse_knn_best_rent),
        min(test_rmse_dt, test_rmse_dt_best)
        - min(test_rmse_dt_rent, test_rmse_dt_best_rent),
    ],
}

model_type_with_rent_df = pd.DataFrame(model_type_with_rent)

print("\n" + "=" * 100)
print("MODEL TYPE COMPARISON: IMPACT OF MEDIAN RENT (Best variant of each)")
print("=" * 100)
print(model_type_with_rent_df.to_string(index=False))
print("=" * 100)

# Visualize model type comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_types = model_type_with_rent_df["Model Type"].tolist()
x_pos = np.arange(len(model_types))
width = 0.35

# R² comparison by model type
ax1 = axes[0]
bars1 = ax1.bar(
    x_pos - width / 2,
    model_type_with_rent_df["Best R² (Without)"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars2 = ax1.bar(
    x_pos + width / 2,
    model_type_with_rent_df["Best R² (With)"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_types, fontsize=11)
ax1.set_ylabel("R² Score", fontsize=12)
ax1.set_title("Best R² Score by Model Type", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")
# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# RMSE comparison by model type
ax2 = axes[1]
bars3 = ax2.bar(
    x_pos - width / 2,
    model_type_with_rent_df["Best RMSE (Without)"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars4 = ax2.bar(
    x_pos + width / 2,
    model_type_with_rent_df["Best RMSE (With)"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_types, fontsize=11)
ax2.set_ylabel("RMSE ($)", fontsize=12)
ax2.set_title(
    "Best RMSE by Model Type (Lower is Better)", fontsize=14, fontweight="bold"
)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
ax2.ticklabel_format(style="plain", axis="y")
# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.show()

# %%
# Feature importance comparison for models that support it
print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS: MEDIAN RENT CONTRIBUTION")
print("=" * 70)

# Linear Regression coefficients
feature_importance_lr_rent = pd.DataFrame(
    {"Feature": feature_columns_rent, "Coefficient": lr_model_rent.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

print("\nLinear Regression - Top 10 Features (with Median Rent):")
print(feature_importance_lr_rent.head(10).to_string(index=False))
print(
    f"\nMedian Rent rank: {feature_importance_lr_rent[feature_importance_lr_rent['Feature'] == 'Median Rent'].index[0] + 1}"
)
print(
    f"Median Rent coefficient: {feature_importance_lr_rent[feature_importance_lr_rent['Feature'] == 'Median Rent']['Coefficient'].values[0]:.4f}"
)

# Decision Tree feature importance
feature_importance_dt_rent = pd.DataFrame(
    {
        "Feature": feature_columns_rent,
        "Importance": best_dt_model_rent.feature_importances_,
    }
).sort_values("Importance", ascending=False)

print("\n" + "-" * 70)
print("\nDecision Tree - Top 10 Features (with Median Rent):")
print(feature_importance_dt_rent.head(10).to_string(index=False))
print(
    f"\nMedian Rent rank: {feature_importance_dt_rent[feature_importance_dt_rent['Feature'] == 'Median Rent'].index[0] + 1}"
)
print(
    f"Median Rent importance: {feature_importance_dt_rent[feature_importance_dt_rent['Feature'] == 'Median Rent']['Importance'].values[0]:.4f}"
)
print("=" * 70)

# Visualize feature importance with Median Rent highlighted
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Linear Regression coefficients
ax1 = axes[0]
top_features_lr = feature_importance_lr_rent.head(15)
colors_lr = [
    (
        "#ff6b6b"
        if x == "Median Rent"
        else (
            "#4ecdc4"
            if top_features_lr.loc[
                top_features_lr["Feature"] == x, "Coefficient"
            ].values[0]
            > 0
            else "#95e1d3"
        )
    )
    for x in top_features_lr["Feature"]
]
ax1.barh(range(len(top_features_lr)), top_features_lr["Coefficient"], color=colors_lr)
ax1.set_yticks(range(len(top_features_lr)))
ax1.set_yticklabels(top_features_lr["Feature"])
ax1.set_xlabel("Coefficient Value", fontsize=12)
ax1.set_title(
    "Linear Regression: Top 15 Features\n(Median Rent highlighted in red)",
    fontsize=13,
    fontweight="bold",
)
ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
ax1.grid(True, alpha=0.3, axis="x")

# Decision Tree importance
ax2 = axes[1]
top_features_dt = feature_importance_dt_rent.head(15)
colors_dt = [
    "#ff6b6b" if x == "Median Rent" else "#a8e6cf" for x in top_features_dt["Feature"]
]
ax2.barh(range(len(top_features_dt)), top_features_dt["Importance"], color=colors_dt)
ax2.set_yticks(range(len(top_features_dt)))
ax2.set_yticklabels(top_features_dt["Feature"])
ax2.set_xlabel("Feature Importance", fontsize=12)
ax2.set_title(
    "Decision Tree: Top 15 Features\n(Median Rent highlighted in red)",
    fontsize=13,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

# %%
# Final summary insights
print("\n" + "=" * 100)
print("KEY INSIGHTS: IMPACT OF PRICE-RELATED DATA (MEDIAN RENT)")
print("=" * 100)

print("\n1️⃣  PERFORMANCE IMPROVEMENTS:")
print(f"   • Average R² improvement across all models: {avg_r2_improvement:.6f}")
print(f"   • Average RMSE improvement across all models: ${avg_rmse_improvement:,.2f}")
print(
    f"   • All models showed improvement: {all(comparison_with_rent_df['R² Improvement'] > 0)}"
)

best_improvement_idx = comparison_with_rent_df["R² Improvement"].idxmax()
worst_improvement_idx = comparison_with_rent_df["R² Improvement"].idxmin()

print(f"\n2️⃣  MOST IMPROVED MODEL:")
print(f"   • {comparison_with_rent_df.loc[best_improvement_idx, 'Model']}")
print(
    f"   • R² improvement: {comparison_with_rent_df.loc[best_improvement_idx, 'R² Improvement']:.6f}"
)
print(
    f"   • RMSE improvement: ${comparison_with_rent_df.loc[best_improvement_idx, 'RMSE Improvement']:,.2f}"
)

print(f"\n3️⃣  LEAST IMPROVED MODEL:")
print(f"   • {comparison_with_rent_df.loc[worst_improvement_idx, 'Model']}")
print(
    f"   • R² improvement: {comparison_with_rent_df.loc[worst_improvement_idx, 'R² Improvement']:.6f}"
)
print(
    f"   • RMSE improvement: ${comparison_with_rent_df.loc[worst_improvement_idx, 'RMSE Improvement']:,.2f}"
)

print(f"\n4️⃣  MEDIAN RENT FEATURE IMPORTANCE:")
median_rent_lr_rank = (
    feature_importance_lr_rent[
        feature_importance_lr_rent["Feature"] == "Median Rent"
    ].index[0]
    + 1
)
median_rent_dt_rank = (
    feature_importance_dt_rent[
        feature_importance_dt_rent["Feature"] == "Median Rent"
    ].index[0]
    + 1
)
print(
    f"   • Linear Regression: Rank {median_rent_lr_rank} out of {len(feature_columns_rent)} features"
)
print(
    f"   • Decision Tree: Rank {median_rent_dt_rank} out of {len(feature_columns_rent)} features"
)

print(f"\n5️⃣  OVERALL BEST MODEL:")
overall_best_idx = comparison_with_rent_df["With Median Rent - R²"].idxmax()
print(f"   • {comparison_with_rent_df.loc[overall_best_idx, 'Model']} WITH Median Rent")
print(
    f"   • R² Score: {comparison_with_rent_df.loc[overall_best_idx, 'With Median Rent - R²']:.6f}"
)
print(
    f"   • RMSE: ${comparison_with_rent_df.loc[overall_best_idx, 'With Median Rent - RMSE']:,.2f}"
)

print("\n" + "=" * 100)
print("\n✅ CONCLUSION:")
print(
    "   Adding price-related features (Median Rent) significantly improves model performance"
)
print(
    "   across all model types, demonstrating that price-related data is highly predictive"
)
print(
    "   of housing prices. However, this also suggests that models with this feature may"
)
print(
    "   have limited practical utility if median rent data is not readily available for"
)
print("   new predictions.")
print("=" * 100)
