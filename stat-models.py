# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: dmml
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Statistical Models
#
# This notebook contains the training and evaluation of the baseline statistical models.
#
# This notebook will fit the following models:
#     - Linear Regressor (both single and multi-variable)
#     - Bayesion Regressor
#     - Decision tree regressor
#
# Imports

# %%
import os
import pickle

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LinearRegression
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

# %% [markdown]
# Load data

# %%
df1 = pd.read_csv("processed/usa_real_estate.csv")
df2 = pd.read_csv("processed/zipcodes.csv")
# merge datasets ,retaining only latest zip code infromation from df2
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
df_merged_with_price = df1.merge(
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


# %% [markdown]
# Check datasets

# %%
df1.info()
df2.info()
df1.head()
df2.head()

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# Prepare features

# %%
feature_col = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
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
feature_col_rent = [
    "bed",
    "bath",
    "acre_lot",
    "house_size",
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

X = df_merged[feature_col]
y = df_merged["price"]

X_rent = df_merged_with_price[feature_col_rent]
y_rent = df_merged_with_price["price"]
# %% [markdown]
# Training Testing split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Fit linear regression model

# %%
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# %% [markdown]
# Evaluation

# %%
# predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# metrics
train_R2 = r2_score(y_train, y_pred_train)
test_R2 = r2_score(y_test, y_pred_test)
train_MAE = mean_absolute_error(y_train, y_pred_train)
test_MAE = mean_absolute_error(y_test, y_pred_test)
train_RMSE = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))

# print results
print(f"Training R2: {train_R2:.4f}, Testing R2: {test_R2:.4f}")
print(f"Training MAE: {train_MAE:.2f}, Testing MAE: {test_MAE:.2f}")
print(f"Training RMSE: {train_RMSE:.2f}, Testing RMSE: {test_RMSE:.2f}")

# %% [markdown]
# Analyze feature importance

# %%
features = pd.DataFrame(
    {"Feature": feature_col, "Coefficient": model.coef_}
).sort_values("Coefficient", key=abs, ascending=False)

# Plot bar graph
plt.figure(figsize=(14, 8))
top_features = features.head(15)
colors = ["green" if x > 0 else "red" for x in top_features["Coefficient"]]
plt.barh(range(len(top_features)), top_features["Coefficient"], color=colors)
plt.yticks(range(len(top_features)), top_features["Feature"])
plt.xlabel("Coefficient Value")
plt.title("Top 15 Feature Coefficients in Linear Regression Model")
plt.show()


# %% [markdown]
# Visualize predictions vs actual values

# %%
plt.figure(figsize=(12, 5))

# Training set figure
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.3, s=1)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Training Set: Actual vs Predicted Prices\nR² = {train_R2:.4f}")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")

# Testing set figure
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.3, s=1)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Testing Set: Actual vs Predicted Prices\nR² = {test_R2:.4f}")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Model improvements

# %% [markdown]
# # Stratified Sampling

# %%
# Create bins
bins = 10
strat_bins = pd.qcut(y, q=bins, labels=False, duplicates="drop")

# Preform stratified split
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat_bins
)

# normalise data
scaler_strat = StandardScaler()
X_train_strat_scaled = scaler_strat.fit_transform(X_train_strat)
X_test_strat_scaled = scaler_strat.transform(X_test_strat)


# Train model with stratified data
model_strat = LinearRegression()
model_strat.fit(X_train_strat_scaled, y_train_strat)


# %% [markdown]
# Evaluate

# %%
y_strat_pred_train = model_strat.predict(X_train_strat_scaled)
y_strat_pred_test = model_strat.predict(X_test_strat_scaled)

# metrics
test_r2_strat = r2_score(y_test_strat, y_strat_pred_test)
train_r2_strat = r2_score(y_train_strat, y_strat_pred_train)
test_MAE_strat = mean_absolute_error(y_test_strat, y_strat_pred_test)
train_MAE_strat = mean_absolute_error(y_train_strat, y_strat_pred_train)
train_RMSE_strat = np.sqrt(mean_squared_error(y_train_strat, y_strat_pred_train))
test_RMSE_strat = np.sqrt(mean_squared_error(y_test_strat, y_strat_pred_test))

# print results
print(f"Training R2: {train_r2_strat:.4f}, Testing R2: {test_r2_strat:.4f}")
print(f"Training MAE: {train_MAE_strat:.2f}, Testing MAE: {test_MAE_strat:.2f}")
print(f"Training RMSE: {train_RMSE_strat:.2f}, Testing RMSE: {test_RMSE_strat:.2f}")


# %% [markdown]
# # K-Fold cross validation

# %%
pipelines = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

k = 10
kflod = KFold(n_splits=k, shuffle=True, random_state=42)

scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
cv_results = cross_validate(
    pipelines, X, y, cv=kflod, scoring=scoring, return_train_score=True
)
# %% [markdown]
# Evaluate

# %%
train_r2_cv = cv_results["train_r2"]
test_r2_cv = cv_results["test_r2"]
train_mae_cv = -cv_results["train_neg_mean_absolute_error"]
test_mae_cv = -cv_results["test_neg_mean_absolute_error"]
train_rmse_cv = np.sqrt(-cv_results["train_neg_mean_squared_error"])
test_rmse_cv = np.sqrt(-cv_results["test_neg_mean_squared_error"])

# print results
print(
    f"Cross-Validation Training R2: {train_r2_cv.mean():.4f} ± {train_r2_cv.std():.4f}"
)
print(f"Cross-Validation Testing R2: {test_r2_cv.mean():.4f} ± {test_r2_cv.std():.4f}")
print(
    f"Cross-Validation Training MAE: {train_mae_cv.mean():.2f} ± {train_mae_cv.std():.2f}"
)
print(
    f"Cross-Validation Testing MAE: {test_mae_cv.mean():.2f} ± {test_mae_cv.std():.2f}"
)
print(
    f"Cross-Validation Training RMSE: {train_rmse_cv.mean():.2f} ± {train_rmse_cv.std():.2f}"
)
print(
    f"Cross-Validation Testing RMSE: {test_rmse_cv.mean():.2f} ± {test_rmse_cv.std():.2f}"
)
# fold by fold results
for x, score in enumerate(test_r2_cv):
    print(f"Fold {x+1}: Testing R2: {score:.4f}")


# %% [markdown]
# Visualise K-fold cross validation results

# %%
# Visualize cross-validation results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R2 scores across folds
axes[0].plot(
    range(1, k + 1), train_r2_cv, "o-", label="Training", linewidth=2, markersize=8
)
axes[0].plot(
    range(1, k + 1), test_r2_cv, "s-", label="Validation", linewidth=2, markersize=8
)
axes[0].axhline(
    y=test_r2_cv.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: {test_r2_cv.mean():.0f}",
)
axes[0].fill_between(
    range(1, k + 1),
    test_r2_cv.mean() - test_r2_cv.std(),
    test_r2_cv.mean() + test_r2_cv.std(),
    alpha=0.2,
    color="red",
)
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("R2 Score")
axes[0].set_title("R2 Score Across Folds")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RMSE across folds
axes[1].plot(
    range(1, k + 1), train_rmse_cv, "o-", label="Training", linewidth=2, markersize=8
)
axes[1].plot(
    range(1, k + 1), test_rmse_cv, "s-", label="Validation", linewidth=2, markersize=8
)
axes[1].axhline(
    y=test_rmse_cv.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: ${test_rmse_cv.mean():,.0f}",
)
axes[1].fill_between(
    range(1, k + 1),
    test_rmse_cv.mean() - test_rmse_cv.std(),
    test_rmse_cv.mean() + test_rmse_cv.std(),
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
    range(1, k + 1), train_mae_cv, "o-", label="Training", linewidth=2, markersize=8
)
axes[2].plot(
    range(1, k + 1), test_mae_cv, "s-", label="Validation", linewidth=2, markersize=8
)
axes[2].axhline(
    y=test_mae_cv.mean(),
    color="red",
    linestyle="--",
    label=f"Mean Val: ${test_mae_cv.mean():,.0f}",
)
axes[2].fill_between(
    range(1, k + 1),
    test_mae_cv.mean() - test_mae_cv.std(),
    test_mae_cv.mean() + test_mae_cv.std(),
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
# ## KNN Regression

# %% [markdown]
# Prepare features

# %% [markdown]
# Fit model

# %%
knn_model = KNeighborsRegressor()
knn_model.fit(X_train_scaled, y_train)


# %% [markdown]
# Evaluate

# %%
# Make predictions
y_pred_test = knn_model.predict(X_test_scaled)
y_pred_train = knn_model.predict(X_train_scaled)

# Calculate metrics
train_R2_knn = r2_score(y_train, y_pred_train)
test_R2_knn = r2_score(y_test, y_pred_test)
train_MAE_knn = mean_absolute_error(y_train, y_pred_train)
test_MAE_knn = mean_absolute_error(y_test, y_pred_test)
train_RMSE_knn = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_RMSE_knn = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Print results
print(f"Training R2: {train_R2_knn:.4f}, Testing R2: {test_R2_knn:.4f}")
print(f"Training MAE: {train_MAE_knn:.2f}, Testing MAE: {test_MAE_knn:.2f}")
print(f"Training RMSE: {train_RMSE_knn:.2f}, Testing RMSE: {test_RMSE_knn:.2f}")

# %% [markdown]
# # Hyperparameter tuning using gridsearch

# %% [markdown]
# Load model if already saved

# %%
grid_search_model_loaded = False

model_path = "models/best_knn_model.pkl"
scaler_path = "models/scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as model_file:
        best_knn_model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    grid_search_model_loaded = True
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Loaded saved model and scaler.")
else:
    print("No saved model found. Please run grid search to train and save the model.")


# %% [markdown]
# Hyperparameter tuning using gridsearch

# %%
# Define parameter grid
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 20, 25, 30],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

# Preform grid search
grid_search = GridSearchCV(
    KNeighborsRegressor(), param_grid, cv=5, scoring="r2", n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best Hyperparameters:", grid_search.best_params_)


# %% [markdown]
# Evaluation

# %%
# Make predictions with the best model
if not grid_search_model_loaded:
    best_knn_model = grid_search.best_estimator_
y_pred_train_knn_best = best_knn_model.predict(X_train_scaled)
y_pred_test_knn_best = best_knn_model.predict(X_test_scaled)

# Calculate metrics
train_R2_knn_best = r2_score(y_train, y_pred_train_knn_best)
test_R2_knn_best = r2_score(y_test, y_pred_test_knn_best)
train_MAE_knn_best = mean_absolute_error(y_train, y_pred_train_knn_best)
test_MAE_knn_best = mean_absolute_error(y_test, y_pred_test_knn_best)
train_RMSE_knn_best = np.sqrt(mean_squared_error(y_train, y_pred_train_knn_best))
test_RMSE_knn_best = np.sqrt(mean_squared_error(y_test, y_pred_test_knn_best))

# Print results
print(
    f"Training R2 (KNN): {train_R2_knn_best:.4f}, Testing R2 (KNN): {test_R2_knn_best:.4f}"
)
print(
    f"Training MAE (KNN): {train_MAE_knn_best:.2f}, Testing MAE (KNN): {test_MAE_knn_best:.2f}"
)
print(
    f"Training RMSE (KNN): {train_RMSE_knn_best:.2f}, Testing RMSE (KNN): {test_RMSE_knn_best:.2f}"
)


# %% [markdown]
# Save best KNN gridrearch model & scaler

# %%
os.makedirs("models", exist_ok=True)
# save model in models directory
model_path = os.path.join("models", "best_knn_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_knn_model, f)
# save scaler as well
scaler_path = os.path.join("models", "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# print path
print(f"Best KNN model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")

# %% [markdown]
# Visualise results

# %%
# Visualize predictions vs actual values for KNN
plt.figure(figsize=(12, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_knn_best, alpha=0.3, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"KNN Training Set\nR² = {train_R2_knn_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test_knn_best, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"KNN Testing Set\nR² = {test_R2_knn_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualise gridsearch results

# %%
if not grid_search_model_loaded:
    # Create DataFrame from cv results
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    # Group by n_neighbors and weights, then calculate mean test score
    kmean_scores = (
        cv_results_df.groupby(["param_n_neighbors"])["mean_test_score"]
        .agg(["mean", "std", "max"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot mean scores for each K value
    axes[0].errorbar(
        kmean_scores["param_n_neighbors"],
        kmean_scores["mean"],
        yerr=kmean_scores["std"],
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    axes[0].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[0].set_ylabel("Mean R2 Score (5-Fold CV)", fontsize=12)
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
        cbar_kws={"label": "Mean R2 Score"},
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

    print("KNN Hyperparameter Tuning Results Visualized.")
    print(
        "k=",
        grid_search.best_params_["n_neighbors"],
        ", weights=",
        grid_search.best_params_["weights"],
        ", metric=",
        grid_search.best_params_["metric"],
    )
else:
    print("KNN grid search model loaded from disk.")


# %% [markdown]
# ## Decision Tree regression

# %%
# Train model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# %% [markdown]
# Evaluate

# %%
# Make predictions
y_pred_train_dt = dt_model.predict(X_train_scaled)
y_pred_test_dt = dt_model.predict(X_test_scaled)

# Calculate metrics
train_R2_dt = r2_score(y_train, y_pred_train_dt)
test_R2_dt = r2_score(y_test, y_pred_test_dt)
train_MAE_dt = mean_absolute_error(y_train, y_pred_train_dt)
test_MAE_dt = mean_absolute_error(y_test, y_pred_test_dt)
train_RMSE_dt = np.sqrt(mean_squared_error(y_train, y_pred_train_dt))
test_RMSE_dt = np.sqrt(mean_squared_error(y_test, y_pred_test_dt))

# Print results
print(f"Decision Tree - Training R2: {train_R2_dt:.4f}, Testing R2: {test_R2_dt:.4f}")
print(
    f"Decision Tree - Training MAE: {train_MAE_dt:.2f}, Testing MAE: {test_MAE_dt:.2f}"
)
print(
    f"Decision Tree - Training RMSE: {train_RMSE_dt:.2f}, Testing RMSE: {test_RMSE_dt:.2f}"
)


# %% [markdown]
# # Hyperparameter tuning for Decision Tree

# %%
df_pram_grid = {
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

dt_grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42), df_pram_grid, cv=5, scoring="r2", n_jobs=-1
)
dt_grid_search.fit(X_train_scaled, y_train)

# get the best model
best_dt_model = dt_grid_search.best_estimator_


# %% [markdown]
# Evaluation

# %%
y_pred_train_dt_best = best_dt_model.predict(X_train_scaled)
y_pred_test_dt_best = best_dt_model.predict(X_test_scaled)

# Calculate metrics
train_R2_dt_best = r2_score(y_train, y_pred_train_dt_best)
test_R2_dt_best = r2_score(y_test, y_pred_test_dt_best)
train_MAE_dt_best = mean_absolute_error(y_train, y_pred_train_dt_best)
test_MAE_dt_best = mean_absolute_error(y_test, y_pred_test_dt_best)
train_RMSE_dt_best = np.sqrt(mean_squared_error(y_train, y_pred_train_dt_best))
test_RMSE_dt_best = np.sqrt(mean_squared_error(y_test, y_pred_test_dt_best))

# Print results
print(
    f"Decision Tree (Tuned) - Training R2: {train_R2_dt_best:.4f}, Testing R2: {test_R2_dt_best:.4f}"
)
print(
    f"Decision Tree (Tuned) - Training MAE: {train_MAE_dt_best:.2f}, Testing MAE: {test_MAE_dt_best:.2f}"
)
print(
    f"Decision Tree (Tuned) - Training RMSE: {train_RMSE_dt_best:.2f}, Testing RMSE: {test_RMSE_dt_best:.2f}"
)
print("Best Hyperparameters for Decision Tree:", dt_grid_search.best_params_)
print(f"Improvement in Testing R2: {test_R2_dt_best - test_R2_dt:.4f}")


# %% [markdown]
# Visualise

# %% [markdown]
# Most important features

# %%
# Analyze feature importance for Decision Tree
feature_importance_dt = pd.DataFrame(
    {"Feature": feature_col, "Importance": best_dt_model.feature_importances_}
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

# %% [markdown]
# Predictions

# %%
# Visualize predictions vs actual values for Decision Tree
plt.figure(figsize=(12, 5))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train_dt_best, alpha=0.3, s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Decision Tree Training Set\nR² = {train_R2_dt_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

# Testing set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test_dt_best, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Decision Tree Testing Set\nR² = {test_R2_dt_best:.4f}")
plt.ticklabel_format(style="plain", axis="both")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final comparision across all models

# %%
final_results = {
    "Model": [
        "Linear Regression",
        "Linear Regression Stratified",
        "Linear Regression CV",
        "KNN (k=5)",
        "KNN Best",
        "Decision Tree",
        "Decision Tree Best",
    ],
    "Testing R2": [
        test_R2,
        test_r2_strat,
        test_r2_cv.mean(),
        test_R2_knn,
        test_R2_knn_best,
        test_R2_dt,
        test_R2_dt_best,
    ],
    "Testing MAE": [
        test_MAE,
        test_MAE_strat,
        test_mae_cv.mean(),
        test_MAE_knn,
        test_MAE_knn_best,
        test_MAE_dt,
        test_MAE_dt_best,
    ],
    "Testing RMSE": [
        test_RMSE,
        test_RMSE_strat,
        test_rmse_cv.mean(),
        test_RMSE_knn,
        test_RMSE_knn_best,
        test_RMSE_dt,
        test_RMSE_dt_best,
    ],
}
final_results_df = pd.DataFrame(final_results)
print("Comparison of Models:")
print(f'R2 Scores:\n{final_results_df[["Model", "Testing R2"]].to_string(index=False)}')
print(
    f'MAE Scores:\n{final_results_df[["Model", "Testing MAE"]].to_string(index=False)}'
)
print(
    f'RMSE Scores:\n{final_results_df[["Model", "Testing RMSE"]].to_string(index=False)}'
)


# %% [markdown]
# Visualisation

# %%
# Visualize final model comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

models_final = final_results_df["Model"].tolist()
colors_final = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

# R2 comparison
axes[0].barh(range(len(models_final)), final_results["Testing R2"], color=colors_final)
axes[0].set_yticks(range(len(models_final)))
axes[0].set_yticklabels(models_final, fontsize=10)
axes[0].set_xlabel("R2 Score", fontsize=12)
axes[0].set_title("R2 Score Comparison - All Models", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_results["Testing R2"]):
    axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

# RMSE comparison
axes[1].barh(
    range(len(models_final)), final_results["Testing RMSE"], color=colors_final
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
for i, v in enumerate(final_results["Testing RMSE"]):
    axes[1].text(v + 3000, i, f"${v:,.0f}", va="center", fontsize=9)

# MAE comparison
axes[2].barh(range(len(models_final)), final_results["Testing MAE"], color=colors_final)
axes[2].set_yticks(range(len(models_final)))
axes[2].set_yticklabels(models_final, fontsize=10)
axes[2].set_xlabel("MAE ($)", fontsize=12)
axes[2].set_title(
    "MAE Comparison - All Models (Lower is Better)", fontsize=14, fontweight="bold"
)
axes[2].ticklabel_format(style="plain", axis="x")
axes[2].grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_results["Testing MAE"]):
    axes[2].text(v + 2000, i, f"${v:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparision with price related features

# %% [markdown]
# #Linear Regression with Median Rent

# %% [markdown]
# Training Data split

# %%
# Split data
X_train_rent, X_test_rent, y_train_rent, y_test_rent = train_test_split(
    X_rent, y_rent, test_size=0.2, random_state=42
)

# Normalize data
scaler_rent = StandardScaler()
X_train_rent_scaled = scaler_rent.fit_transform(X_train_rent)
X_test_rent_scaled = scaler_rent.transform(X_test_rent)


# %% [markdown]
# Fit model

# %%
model_lr_rent = LinearRegression()
model_lr_rent.fit(X_train_rent_scaled, y_train_rent)


# %% [markdown]
# Evaluate

# %%
# Make predictions
y_pred_train_rent = model_lr_rent.predict(X_train_rent_scaled)
y_pred_test_rent = model_lr_rent.predict(X_test_rent_scaled)

# Calculate metrics
train_R2_rent = r2_score(y_train_rent, y_pred_train_rent)
test_R2_rent = r2_score(y_test_rent, y_pred_test_rent)
train_MAE_rent = mean_absolute_error(y_train_rent, y_pred_train_rent)
test_MAE_rent = mean_absolute_error(y_test_rent, y_pred_test_rent)
train_RMSE_rent = np.sqrt(mean_squared_error(y_train_rent, y_pred_train_rent))
test_RMSE_rent = np.sqrt(mean_squared_error(y_test_rent, y_pred_test_rent))

# Print results
print(f"Training R2: {train_R2_rent:.4f}, Testing R2: {test_R2_rent:.4f}")
print(f"Training MAE: {train_MAE_rent:.2f}, Testing MAE: {test_MAE_rent:.2f}")
print(f"Training RMSE: {train_RMSE_rent:.2f}, Testing RMSE: {test_RMSE_rent:.2f}")


# %% [markdown]
# Stratified sampling

# %%
# Create bins
bins = 10
strat_bins_rent = pd.qcut(y_rent, q=bins, labels=False, duplicates="drop")

# Preform stratified split
X_train_strat_rent, X_test_strat_rent, y_train_strat_rent, y_test_strat_rent = (
    train_test_split(
        X_rent, y_rent, test_size=0.2, random_state=42, stratify=strat_bins_rent
    )
)

# normalise features
scaler_strat = StandardScaler()
X_train_strat_scaled_rent = scaler_strat.fit_transform(X_train_strat_rent)
X_test_strat_scaled_rent = scaler_strat.transform(X_test_strat_rent)

# Train model with stratified data
model_strat = LinearRegression()
model_strat.fit(X_train_strat_scaled_rent, y_train_strat_rent)

# Make predictions
y_strat_pred_train_rent = model_strat.predict(X_train_strat_scaled_rent)
y_strat_pred_test_rent = model_strat.predict(X_test_strat_scaled_rent)

# metrics
test_r2_strat_rent = r2_score(y_test_strat_rent, y_strat_pred_test_rent)
train_r2_strat_rent = r2_score(y_train_strat_rent, y_strat_pred_train_rent)
test_MAE_strat_rent = mean_absolute_error(y_test_strat_rent, y_strat_pred_test_rent)
train_MAE_strat_rent = mean_absolute_error(y_train_strat_rent, y_strat_pred_train_rent)
train_RMSE_strat_rent = np.sqrt(
    mean_squared_error(y_train_strat_rent, y_strat_pred_train_rent)
)
test_RMSE_strat_rent = np.sqrt(
    mean_squared_error(y_test_strat_rent, y_strat_pred_test_rent)
)

# print results
print(f"Training R2: {train_r2_strat_rent:.4f}, Testing R2: {test_r2_strat_rent:.4f}")
print(
    f"Training MAE: {test_MAE_strat_rent:.2f}, Testing MAE: {train_MAE_strat_rent:.2f}"
)
print(
    f"Training RMSE: {train_RMSE_strat_rent:.2f}, Testing RMSE: {test_RMSE_strat_rent:.2f}"
)


# %% [markdown]
# K-Fold cross validation

# %%

# Train model with cross-validation
pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

k = 10
kflod = KFold(n_splits=k, shuffle=True, random_state=42)

scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
cv_results_rent = cross_validate(
    pipeline, X_rent, y_rent, cv=kflod, scoring=scoring, return_train_score=True
)

# make predictions

train_r2_cv_rent = cv_results_rent["train_r2"]
test_r2_cv_rent = cv_results_rent["test_r2"]
train_mae_cv_rent = -cv_results_rent["train_neg_mean_absolute_error"]
test_mae_cv_rent = -cv_results_rent["test_neg_mean_absolute_error"]
train_rmse_cv_rent = np.sqrt(-cv_results_rent["train_neg_mean_squared_error"])
test_rmse_cv_rent = np.sqrt(-cv_results_rent["test_neg_mean_squared_error"])

# print results
print(
    f"Cross-Validation Training R2: {train_r2_cv_rent.mean():.4f} ± {train_r2_cv_rent.std():.4f}"
)
print(
    f"Cross-Validation Testing R2: {test_r2_cv_rent.mean():.4f} ± {test_r2_cv_rent.std():.4f}"
)
print(
    f"Cross-Validation Training MAE: {train_mae_cv_rent.mean():.2f} ± {train_mae_cv_rent.std():.2f}"
)
print(
    f"Cross-Validation Testing MAE: {test_mae_cv_rent.mean():.2f} ± {test_mae_cv_rent.std():.2f}"
)
print(
    f"Cross-Validation Training RMSE: {train_rmse_cv_rent.mean():.2f} ± {train_rmse_cv_rent.std():.2f}"
)
print(
    f"Cross-Validation Testing RMSE: {train_rmse_cv_rent.mean():.2f} ± {train_rmse_cv_rent.std():.2f}"
)
# fold by fold results
for x, score in enumerate(test_r2_cv_rent):
    print(f"Fold {x+1}: Testing R2: {score:.4f}")

# %% [markdown]
# # KNN Regression

# %%
X_train_rent, X_test_rent, y_train_rent, y_test_rent = train_test_split(
    X_rent, y_rent, test_size=0.2, random_state=42
)
# Prepare features
X_train_scaled_rent = scaler.fit_transform(X_train_rent)
X_test_scaled_rent = scaler.transform(X_test_rent)

# Train KNN model
knn_model_rent = KNeighborsRegressor()
knn_model_rent.fit(X_train_scaled_rent, y_train_rent)

# Make predictions
y_pred_test_rent = knn_model_rent.predict(X_test_scaled_rent)
y_pred_train_rent = knn_model_rent.predict(X_train_scaled_rent)

# Calculate metrics
train_R2_knn_rent = r2_score(y_train_rent, y_pred_train_rent)
test_R2_knn_rent = r2_score(y_test_rent, y_pred_test_rent)
train_MAE_knn_rent = mean_absolute_error(y_train_rent, y_pred_train_rent)
test_MAE_knn_rent = mean_absolute_error(y_test_rent, y_pred_test_rent)
train_RMSE_knn_rent = np.sqrt(mean_squared_error(y_train_rent, y_pred_train_rent))
test_RMSE_knn_rent = np.sqrt(mean_squared_error(y_test_rent, y_pred_test_rent))

# Print results
print(f"KNN Training R2: {train_R2_knn_rent:.4f}, Testing R2: {test_R2_knn_rent:.4f}")
print(
    f"KNN Training MAE: {train_MAE_knn_rent:.2f}, Testing MAE: {test_MAE_knn_rent:.2f}"
)
print(
    f"KNN Training RMSE: {train_RMSE_knn_rent:.2f}, Testing RMSE: {test_RMSE_knn_rent:.2f}"
)

# %% [markdown]
# Hyperparameter tuning using gridsearch

# %% [markdown]
# Load model if already saved

# %%
grid_search_model_loaded_rent = False

model_path_rent = "models/best_knn_model_rent.pkl"
scaler_path_rent = "models/scaler_rent.pkl"

if os.path.exists(model_path_rent) and os.path.exists(scaler_path_rent):
    # Load model and scaler
    with open(model_path_rent, "rb") as model_file:
        best_knn_model_rent = pickle.load(model_file)
    with open(scaler_path_rent, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    # Set grid search loaded to true
    grid_search_model_loaded_rent = True
    X_train_scaled_rent = scaler.transform(X_train_rent)
    X_test_scaled_rent = scaler.transform(X_test_rent)

    print("Loaded saved model and scaler.")
else:
    print("No saved model found. Please run grid search to train and save the model.")


# %% [markdown]
# Hyperparameter tuning using gridsearch

# %%
# Define parameter grid
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 20, 25, 30],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

# Preform grid search
grid_search_rent = GridSearchCV(
    KNeighborsRegressor(), param_grid, cv=5, scoring="r2", n_jobs=-1
)
grid_search_rent.fit(X_train_scaled_rent, y_train_rent)

print("Best Hyperparameters:", grid_search.best_params_)

# %% [markdown]
# Evaluation

# %%
# Make predictions with the best model
if not grid_search_model_loaded_rent:
    best_knn_model_rent = grid_search_rent.best_estimator_

y_pred_train_knn_rent = best_knn_model_rent.predict(X_train_scaled_rent)
y_pred_test_knn_rent = best_knn_model_rent.predict(X_test_scaled_rent)

# Calculate metrics
train_R2_knn_best_rent = r2_score(y_train_rent, y_pred_train_knn_rent)
test_R2_knn_best_rent = r2_score(y_test_rent, y_pred_test_knn_rent)
train_MAE_knn_best_rent = mean_absolute_error(y_train_rent, y_pred_train_knn_rent)
test_MAE_knn_best_rent = mean_absolute_error(y_test_rent, y_pred_test_knn_rent)
train_RMSE_knn_best_rent = np.sqrt(
    mean_squared_error(y_train_rent, y_pred_train_knn_rent)
)
test_RMSE_knn_best_rent = np.sqrt(mean_squared_error(y_test_rent, y_pred_test_knn_rent))

# Print results
print(
    f"Training R2 (KNN): {train_R2_knn_best_rent:.4f}, Testing R2 (KNN): {test_R2_knn_best_rent:.4f}"
)
print(
    f"Training MAE (KNN): {train_MAE_knn_best_rent:.2f}, Testing MAE (KNN): {test_MAE_knn_best_rent:.2f}"
)
print(
    f"Training RMSE (KNN): {train_RMSE_knn_best_rent:.2f}, Testing RMSE (KNN): {test_RMSE_knn_best_rent:.2f}"
)

# %% [markdown]
# Save model

# %%
os.makedirs("models", exist_ok=True)
# save model in models directory
model_path_rent = os.path.join("models", "best_knn_model_rent.pkl")
with open(model_path_rent, "wb") as f:
    pickle.dump(best_knn_model_rent, f)
# save scaler as well
scaler_path_rent = os.path.join("models", "scaler_rent.pkl")
with open(scaler_path_rent, "wb") as f:
    pickle.dump(scaler, f)

# print path
print(f"Best KNN model saved to {model_path_rent}")
print(f"Scaler saved to {scaler_path_rent}")


# %% [markdown]
# ## Decision Tree regression

# %%
# Train model
dt_model_rent = DecisionTreeRegressor(random_state=42)
dt_model_rent.fit(X_train_scaled_rent, y_train_rent)

# Make predictions
y_pred_train_dt_rent = dt_model_rent.predict(X_train_scaled_rent)
y_pred_test_dt_rent = dt_model_rent.predict(X_test_scaled_rent)

# Calculate metrics
train_R2_dt_rent = r2_score(y_train_rent, y_pred_train_dt_rent)
test_R2_dt_rent = r2_score(y_test_rent, y_pred_test_dt_rent)
train_MAE_dt_rent = mean_absolute_error(y_train_rent, y_pred_train_dt_rent)
test_MAE_dt_rent = mean_absolute_error(y_test_rent, y_pred_test_dt_rent)
train_RMSE_dt_rent = np.sqrt(mean_squared_error(y_train_rent, y_pred_train_dt_rent))
test_RMSE_dt_rent = np.sqrt(mean_squared_error(y_test_rent, y_pred_test_dt_rent))

# Print results
print(
    f"Decision Tree - Training R2: {train_R2_dt_rent:.4f}, Testing R2: {test_R2_dt_rent:.4f}"
)
print(
    f"Decision Tree - Training MAE: {train_MAE_dt_rent:.2f}, Testing MAE: {test_MAE_dt_rent:.2f}"
)
print(
    f"Decision Tree - Training RMSE: {train_RMSE_dt_rent:.2f}, Testing RMSE: {test_RMSE_dt_rent:.2f}"
)


# %% [markdown]
# Hyperparameter tuning for Decision tree using gridsearch

# %%
df_pram_grid = {
    "max_depth": [5, 10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

dt_grid_search_rent = GridSearchCV(
    DecisionTreeRegressor(random_state=42), df_pram_grid, cv=5, scoring="r2", n_jobs=-1
)
dt_grid_search_rent.fit(X_train_scaled_rent, y_train_rent)

# get the best model
best_dt_model_rent = dt_grid_search_rent.best_estimator_

# Make predictions
y_pred_train_dt_best_rent = best_dt_model_rent.predict(X_train_scaled_rent)
y_pred_test_dt_best_rent = best_dt_model_rent.predict(X_test_scaled_rent)

# Calculate metrics
train_R2_dt_best_rent = r2_score(y_train_rent, y_pred_train_dt_best_rent)
test_R2_dt_best_rent = r2_score(y_test_rent, y_pred_test_dt_best_rent)
train_MAE_dt_best_rent = mean_absolute_error(y_train_rent, y_pred_train_dt_best_rent)
test_MAE_dt_best_rent = mean_absolute_error(y_test_rent, y_pred_test_dt_best_rent)
train_RMSE_dt_best_rent = np.sqrt(
    mean_squared_error(y_train_rent, y_pred_train_dt_best_rent)
)
test_RMSE_dt_best_rent = np.sqrt(
    mean_squared_error(y_test_rent, y_pred_test_dt_best_rent)
)

# Print results
print(
    f"Decision Tree (Tuned) - Training R2: {train_R2_dt_best_rent:.4f}, Testing R2: {test_R2_dt_best_rent:.4f}"
)
print(
    f"Decision Tree (Tuned) - Training MAE: {train_MAE_dt_best_rent:.2f}, Testing MAE: {test_MAE_dt_best_rent:.2f}"
)
print(
    f"Decision Tree (Tuned) - Training RMSE: {train_RMSE_dt_best_rent:.2f}, Testing RMSE: {test_RMSE_dt_best_rent:.2f}"
)
print("Best Hyperparameters for Decision Tree:", dt_grid_search_rent.best_params_)
print(f"Improvement in Testing R2: {test_R2_dt_best_rent - test_R2_dt_rent:.4f}")

# Analyze feature importance for Decision Tree
feature_importance_dt_rent = pd.DataFrame(
    {"Feature": feature_col_rent, "Importance": best_dt_model_rent.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Most Important Features (Decision Tree):")
print(feature_importance_dt_rent.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 8))
top_features_dt_rent = feature_importance_dt_rent.head(15)
colors_dt = plt.cm.viridis(np.linspace(0, 1, len(top_features_dt_rent)))
plt.barh(
    range(len(top_features_dt_rent)),
    top_features_dt_rent["Importance"],
    color=colors_dt,
)
plt.yticks(range(len(top_features_dt_rent)), top_features_dt_rent["Feature"])
plt.xlabel("Feature Importance", fontsize=12)
plt.title(
    "Top 15 Feature Importances in Decision Tree Model", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()


# %% [markdown]
# # Final Comparision With vs Without Rent

# %% [markdown]
#  Comprehensive

# %%
final_resultsvsrent = {
    "Model": [
        "Linear Regression",
        "Linear Regression (Stratified)",
        "Linear Regression (10-Fold CV)",
        "KNN (k=5, default)",
        "KNN (Optimized)",
        "Decision Tree (default)",
        "Decision Tree (Optimized)",
    ],
    "Without Median Rent - R2": [
        test_R2,
        test_r2_strat,
        test_r2_cv.mean(),
        test_R2_knn,
        test_R2_knn_best,
        test_R2_dt,
        test_R2_dt_best,
    ],
    "With Median Rent - R2": [
        test_R2_rent,
        test_r2_strat_rent,
        test_r2_cv_rent.mean(),
        test_R2_knn_rent,
        test_R2_knn_best_rent,
        test_R2_dt_rent,
        test_R2_dt_best_rent,
    ],
    "R2 Improvement": [
        test_R2_rent - test_R2,
        test_r2_strat_rent - test_r2_strat,
        test_r2_cv_rent.mean() - test_r2_cv.mean(),
        test_R2_knn_rent - test_R2_knn,
        test_R2_knn_best_rent - test_R2_knn_best,
        test_R2_dt_rent - test_R2_dt,
        test_R2_dt_best_rent - test_R2_dt_best,
    ],
    "Without Median Rent - RMSE": [
        test_RMSE,
        test_RMSE_strat,
        test_rmse_cv.mean(),
        test_RMSE_knn,
        test_RMSE_knn_best,
        test_RMSE_dt,
        test_RMSE_dt_best,
    ],
    "With Median Rent - RMSE": [
        test_RMSE_rent,
        test_RMSE_strat_rent,
        test_rmse_cv_rent.mean(),
        test_RMSE_knn_rent,
        test_RMSE_knn_best_rent,
        test_RMSE_dt_rent,
        test_RMSE_dt_best_rent,
    ],
    "RMSE Improvement": [
        test_RMSE - test_RMSE_rent,
        test_RMSE_strat - test_RMSE_strat_rent,
        test_rmse_cv.mean() - test_rmse_cv_rent.mean(),
        test_RMSE_knn - test_RMSE_knn_rent,
        test_RMSE_knn_best - test_RMSE_knn_best_rent,
        test_RMSE_dt - test_RMSE_dt_rent,
        test_RMSE_dt_best - test_RMSE_dt_best_rent,
    ],
}
final_resultsvsrent_df = pd.DataFrame(final_resultsvsrent)

print("Comparison of Models:")
print(
    "R2 Scores:\n",
    final_resultsvsrent_df[
        ["Model", "Without Median Rent - R2", "With Median Rent - R2", "R2 Improvement"]
    ].to_string(index=False),
)
print(
    "RMSE Scores:\n",
    final_resultsvsrent_df[
        [
            "Model",
            "Without Median Rent - RMSE",
            "With Median Rent - RMSE",
            "RMSE Improvement",
        ]
    ].to_string(index=False),
)


# %% [markdown]
# Visualisation of diffrence

# %%
# Visualize the impact of adding Median Rent
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

models_list = final_resultsvsrent_df["Model"].tolist()
colors_without = "#3498db"
colors_with = "#2ecc71"

# 1. R2 Score Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(models_list))
width = 0.35
bars1 = ax1.barh(
    x_pos - width / 2,
    final_resultsvsrent_df["Without Median Rent - R2"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars2 = ax1.barh(
    x_pos + width / 2,
    final_resultsvsrent_df["With Median Rent - R2"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax1.set_yticks(x_pos)
ax1.set_yticklabels(models_list, fontsize=9)
ax1.set_xlabel("R2 Score", fontsize=11)
ax1.set_title("R2 Score: With vs Without Median Rent", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3, axis="x")

# 2. RMSE Comparison
ax2 = axes[0, 1]
bars3 = ax2.barh(
    x_pos - width / 2,
    final_resultsvsrent_df["Without Median Rent - RMSE"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars4 = ax2.barh(
    x_pos + width / 2,
    final_resultsvsrent_df["With Median Rent - RMSE"],
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

# 3. R2 Improvement
ax3 = axes[1, 0]
improvement_colors = [
    "#2ecc71" if x > 0 else "#e74c3c" for x in final_resultsvsrent_df["R2 Improvement"]
]
bars5 = ax3.barh(
    x_pos, final_resultsvsrent_df["R2 Improvement"], color=improvement_colors, alpha=0.8
)
ax3.set_yticks(x_pos)
ax3.set_yticklabels(models_list, fontsize=9)
ax3.set_xlabel("R2 Improvement", fontsize=11)
ax3.set_title("R2 Improvement from Adding Median Rent", fontsize=13, fontweight="bold")
ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
ax3.grid(True, alpha=0.3, axis="x")
# Add value labels
for i, v in enumerate(final_resultsvsrent_df["R2 Improvement"]):
    ax3.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

# 4. RMSE Improvement
ax4 = axes[1, 1]
rmse_improvement_colors = [
    "#2ecc71" if x > 0 else "#e74c3c"
    for x in final_resultsvsrent_df["RMSE Improvement"]
]
bars6 = ax4.barh(
    x_pos,
    final_resultsvsrent_df["RMSE Improvement"],
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
for i, v in enumerate(final_resultsvsrent_df["RMSE Improvement"]):
    ax4.text(v + 500, i, f"${v:,.0f}", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# Summary

# %%
final_resultsvsrent_summary = {
    "Model Type": ["Linear Regression", "KNN", "Decision Tree"],
    "Best R2 (Without)": [
        max(test_R2, test_r2_strat, test_r2_cv.mean()),
        max(test_R2_knn, test_R2_knn_best),
        max(test_R2_dt, test_R2_dt_best),
    ],
    "Best R2 (With)": [
        max(test_R2_rent, test_r2_strat_rent, test_r2_cv_rent.mean()),
        max(test_R2_knn_rent, test_R2_knn_best_rent),
        max(test_R2_dt_rent, test_R2_dt_best_rent),
    ],
    "R2 Improvement": [
        max(test_R2_rent, test_r2_strat_rent, test_r2_cv_rent.mean())
        - max(test_R2, test_r2_strat, test_r2_cv.mean()),
        max(test_R2_knn_rent, test_R2_knn_best_rent)
        - max(test_R2_knn, test_R2_knn_best),
        max(test_R2_dt_rent, test_R2_dt_best_rent) - max(test_R2_dt, test_R2_dt_best),
    ],
    "Best RMSE (Without)": [
        min(test_RMSE, test_RMSE_strat, test_rmse_cv.mean()),
        min(test_RMSE_knn, test_RMSE_knn_best),
        min(test_RMSE_dt, test_RMSE_dt_best),
    ],
    "Best RMSE (With)": [
        min(test_RMSE_rent, test_RMSE_strat_rent, test_rmse_cv_rent.mean()),
        min(test_RMSE_knn_rent, test_RMSE_knn_best_rent),
        min(test_RMSE_dt_rent, test_RMSE_dt_best_rent),
    ],
    "RMSE Improvement": [
        min(test_RMSE, test_RMSE_strat, test_rmse_cv.mean())
        - min(test_RMSE_rent, test_RMSE_strat_rent, test_rmse_cv_rent.mean()),
        min(test_RMSE_knn, test_RMSE_knn_best)
        - min(test_RMSE_knn_rent, test_RMSE_knn_best_rent),
        min(test_RMSE_dt, test_RMSE_dt_best)
        - min(test_RMSE_dt_rent, test_RMSE_dt_best_rent),
    ],
}
final_resultsvsrent_summary_df = pd.DataFrame(final_resultsvsrent_summary)

print(final_resultsvsrent_summary_df.to_string(index=False))


# %% [markdown]
# Visualisation

# %%
# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_types = final_resultsvsrent_summary_df["Model Type"].tolist()
x_pos = np.arange(len(model_types))
width = 0.35

# R2 Score Comparison
ax1 = axes[0]
bars1 = ax1.bar(
    x_pos - width / 2,
    final_resultsvsrent_summary_df["Best R2 (Without)"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars2 = ax1.bar(
    x_pos + width / 2,
    final_resultsvsrent_summary_df["Best R2 (With)"],
    width,
    label="With Median Rent",
    color=colors_with,
    alpha=0.8,
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_types, fontsize=11)
ax1.set_ylabel("R2 Score", fontsize=12)
ax1.set_title("Best R2 Score by Model Type", fontsize=14, fontweight="bold")
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

# RMSE Comparison
ax2 = axes[1]
bars3 = ax2.bar(
    x_pos - width / 2,
    final_resultsvsrent_summary_df["Best RMSE (Without)"],
    width,
    label="Without Median Rent",
    color=colors_without,
    alpha=0.8,
)
bars4 = ax2.bar(
    x_pos + width / 2,
    final_resultsvsrent_summary_df["Best RMSE (With)"],
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

# %% [markdown]
# Feature importance visualisation

# %%
# Feature importance comparison
print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS: MEDIAN RENT CONTRIBUTION")
print("=" * 70)


# Linear Regression coefficients
feature_importance_lr_rent = (
    pd.DataFrame({"Feature": feature_col_rent, "Coefficient": model_lr_rent.coef_})
    .sort_values("Coefficient", key=abs, ascending=False)
    .reset_index(drop=True)
)

print("\nLinear Regression - Top 10 Features (with Median Rent):")
print(feature_importance_lr_rent.head(10).to_string(index=False))

if "Median Rent" in feature_importance_lr_rent["Feature"].values:
    lr_pos = (
        feature_importance_lr_rent[
            feature_importance_lr_rent["Feature"] == "Median Rent"
        ].index[0]
        + 1
    )
    lr_coef = feature_importance_lr_rent.loc[
        feature_importance_lr_rent["Feature"] == "Median Rent", "Coefficient"
    ].values[0]
    print(f"\nMedian Rent rank: {lr_pos}")
    print(f"Median Rent coefficient: {lr_coef:.4f}")
else:
    print("\nMedian Rent not found in feature_col_rent.")

# Decision Tree feature importance
feature_importance_dt_rent = (
    pd.DataFrame(
        {
            "Feature": feature_col_rent,
            "Importance": best_dt_model_rent.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "-" * 70)
print("\nDecision Tree - Top 10 Features (with Median Rent):")
print(feature_importance_dt_rent.head(10).to_string(index=False))

if "Median Rent" in feature_importance_dt_rent["Feature"].values:
    dt_pos = (
        feature_importance_dt_rent[
            feature_importance_dt_rent["Feature"] == "Median Rent"
        ].index[0]
        + 1
    )
    dt_imp = feature_importance_dt_rent.loc[
        feature_importance_dt_rent["Feature"] == "Median Rent", "Importance"
    ].values[0]
    print(f"\nMedian Rent rank: {dt_pos}")
    print(f"Median Rent importance: {dt_imp:.4f}")
else:
    print("\nMedian Rent not found in feature_col_rent.")
print("=" * 70)

# Visualize feature importance with Median Rent highlighted
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Linear Regression coefficients plot
ax1 = axes[0]
top_features_lr = feature_importance_lr_rent.head(15)
# build colors: red for Median Rent, teal for positive coeffs, light teal for negative
coef_map = top_features_lr.set_index("Feature")["Coefficient"].to_dict()
colors_lr = []
for f in top_features_lr["Feature"]:
    if f == "Median Rent":
        colors_lr.append("#ff6b6b")
    else:
        colors_lr.append("#4ecdc4" if coef_map[f] > 0 else "#95e1d3")

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

# Decision Tree importance plot
ax2 = axes[1]
top_features_dt = feature_importance_dt_rent.head(15)
colors_dt = [
    "#ff6b6b" if f == "Median Rent" else "#a8e6cf" for f in top_features_dt["Feature"]
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
plt.show()  # re
