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
# # Neural Networks
#
# This notebook trains various standard feed-forward MLP neural networks.
#
# import matplotlib.pyplot as plt
# import numpy as np
# %%
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# Firstly, merge the data on the zip code, and use the demographic/socio-economic features from the zip code data

# %%
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

# %% [markdown]
# Select feature columns. Basically everything but exclude the price related information:

# %%

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
    # Socio-economic
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

# %% [markdown]
# Split the data into train, test, validation using stratified sampling based on the price.
#
# 70/15/15 split
#
# Since the price is continuous, we need to bin it into discrete categories first.

# %%
# Create bins
y_binned = pd.qcut(y, q=10, labels=False, duplicates="drop")

# First split: separate out test set (15%)
X_temp, X_test, y_temp, y_test, y_temp_binned, y_test_binned = train_test_split(
    X, y, y_binned, test_size=0.15, random_state=42, stratify=y_binned
)

# Second split: split remaining data into train (70%) and validation (15%)
# 0.1765 of 85% ≈ 15% of total
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp_binned
)

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")


# %% [markdown]
# Use a standard scalar to normalize the data.
#
# Standard scalar (z-score normalization) is a good choice because house prices vary a lot so min-max might squish lots of values close to each other

# %%
in_scalar = StandardScaler()
X_train_scaled = in_scalar.fit_transform(X_train)
X_test_scaled = in_scalar.transform(X_test)
X_val_scaled = in_scalar.transform(X_val)
out_scalar = StandardScaler()
y_train_scaled = out_scalar.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = out_scalar.transform(y_test.values.reshape(-1, 1))
y_val_scaled = out_scalar.transform(y_val.values.reshape(-1, 1))


# %% [markdown]
# Define all model architectures to compare


# %%
def create_model(architecture, input_dim):
    """
    Create a neural network model with the specified architecture.

    Args:
        architecture: List of layer sizes (excluding input, including output)
        input_dim: Number of input features

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    # Add hidden layers
    for i, units in enumerate(architecture[:-1]):  # All layers except the last
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(0.001),
            )
        )
        # Add dropout for regularization (skip for very small layers)
        if units >= 16:
            dropout_rate = 0.2 if i == 0 else 0.15
            model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(architecture[-1], activation="linear"))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipvalue=1.0),
        loss="huber",
        metrics=["mae"],
    )

    return model


# Define all architectures to test
architectures = {
    "original": [64, 32, 16, 1],
    "deep": [256, 128, 64, 32, 16, 1],
    "simple_256_64": [256, 64, 1],
    "simple_128": [128, 1],
    "simple_64": [64, 1],
    "three_layer": [32, 16, 8, 1],
    "deep_128": [128, 64, 1],
    "double_32": [32, 32, 1],
}

print(f"Will train and compare {len(architectures)} different architectures:")
for name, arch in architectures.items():
    print(f"  {name}: {' -> '.join(map(str, arch))}")


# %% [markdown]
# Check GPU availability

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# %% [markdown]
# Train all models and save results

# %%
import os
from datetime import datetime

# Create directory for saving models
os.makedirs("models/nn", exist_ok=True)

# Store results for comparison
results = {}

# Train each architecture
for arch_name, architecture in architectures.items():
    print(f"\n{'='*60}")
    print(f"Training {arch_name}: {' -> '.join(map(str, architecture))}")
    print(f"{'='*60}\n")

    # Create model
    model = create_model(architecture, X_train.shape[1])

    # Display model summary
    print(f"\n{arch_name} Model Summary:")
    model.summary()

    # Setup callbacks
    model_path = f"models/nn/{arch_name}_model.keras"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    # Train model
    start_time = datetime.now()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=2056,
        verbose=1,
        callbacks=[checkpoint_callback, early_stopping],
    )
    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    # Store results
    results[arch_name] = {
        "architecture": architecture,
        "history": history.history,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "training_time": training_time,
        "model_path": model_path,
        "total_params": model.count_params(),
    }

    print(f"\n{arch_name} Results:")
    print(f"  Test Loss (Huber): {test_loss:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Saved to: {model_path}")

print(f"\n{'='*60}")
print("All models trained successfully!")
print(f"{'='*60}")

# %% [markdown]
# Compare model performance

# %%
# Create comparison dataframe
comparison_df = pd.DataFrame(
    {
        "Architecture": [
            " -> ".join(map(str, results[name]["architecture"])) for name in results
        ],
        "Test Loss": [results[name]["test_loss"] for name in results],
        "Test MAE": [results[name]["test_mae"] for name in results],
        "Training Time (s)": [results[name]["training_time"] for name in results],
        "Total Parameters": [results[name]["total_params"] for name in results],
    },
    index=results.keys(),
)

# Sort by test MAE
comparison_df = comparison_df.sort_values("Test MAE")

print("Model Comparison (sorted by Test MAE):")
print("=" * 80)
print(comparison_df.to_string())
print("=" * 80)

# Find best model
best_model_name = comparison_df.index[0]
print(f"\nBest performing model: {best_model_name}")
print(f"Architecture: {comparison_df.loc[best_model_name, 'Architecture']}")
print(f"Test MAE: {comparison_df.loc[best_model_name, 'Test MAE']:.4f}")

# %% [markdown]
# Visualize training history for all models

# %%
# Plot training and validation loss for all models
num_models = len(results)
cols = 3
rows = (num_models + cols - 1) // cols  # Calculate rows needed

fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
axes = axes.flatten()

for idx, (name, data) in enumerate(results.items()):
    history = data["history"]

    axes[idx].plot(history["loss"], label="Training Loss", linewidth=2)
    axes[idx].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[idx].set_title(
        f"{name}\n{' -> '.join(map(str, data['architecture']))}",
        fontsize=12,
        fontweight="bold",
    )
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Loss (Huber)")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

    # Add final test MAE as text
    axes[idx].text(
        0.02,
        0.98,
        f"Test MAE: {data['test_mae']:.4f}",
        transform=axes[idx].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

# Hide any unused subplots
for idx in range(num_models, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig("models/nn/training_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Training history plot saved to: models/nn/training_comparison.png")

# %% [markdown]
# Compare metrics across models

# %%
# Create bar plots for comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Test MAE comparison
axes[0, 0].bar(range(len(comparison_df)), comparison_df["Test MAE"], color="steelblue")
axes[0, 0].set_xticks(range(len(comparison_df)))
axes[0, 0].set_xticklabels(comparison_df.index, rotation=45, ha="right")
axes[0, 0].set_ylabel("Test MAE")
axes[0, 0].set_title("Test MAE by Model", fontweight="bold")
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Test Loss comparison
axes[0, 1].bar(range(len(comparison_df)), comparison_df["Test Loss"], color="coral")
axes[0, 1].set_xticks(range(len(comparison_df)))
axes[0, 1].set_xticklabels(comparison_df.index, rotation=45, ha="right")
axes[0, 1].set_ylabel("Test Loss (Huber)")
axes[0, 1].set_title("Test Loss by Model", fontweight="bold")
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Training time comparison
axes[1, 0].bar(
    range(len(comparison_df)), comparison_df["Training Time (s)"], color="forestgreen"
)
axes[1, 0].set_xticks(range(len(comparison_df)))
axes[1, 0].set_xticklabels(comparison_df.index, rotation=45, ha="right")
axes[1, 0].set_ylabel("Training Time (seconds)")
axes[1, 0].set_title("Training Time by Model", fontweight="bold")
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Parameter count comparison
axes[1, 1].bar(
    range(len(comparison_df)), comparison_df["Total Parameters"], color="mediumpurple"
)
axes[1, 1].set_xticks(range(len(comparison_df)))
axes[1, 1].set_xticklabels(comparison_df.index, rotation=45, ha="right")
axes[1, 1].set_ylabel("Number of Parameters")
axes[1, 1].set_title("Model Size (Parameters) by Model", fontweight="bold")
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("models/nn/metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Metrics comparison plot saved to: models/nn/metrics_comparison.png")

# %% [markdown]
# Test best model on sample properties

# %%
# Load the best model
best_model = keras.models.load_model(results[best_model_name]["model_path"])

# Test on sample properties
sample_properties = X_test.sample(10, random_state=42)
# sample_properties_scaled = in_scalar.transform(sample_properties)
predicted_prices = best_model.predict(sample_properties, verbose=0)
# predicted_prices = out_scalar.inverse_transform(predicted_prices)

print(f"Predictions using best model: {best_model_name}")
print(
    f"Architecture: {' -> '.join(map(str, results[best_model_name]['architecture']))}\n"
)

for i, prop in enumerate(sample_properties.index):
    true_price = y_test.loc[prop]
    predicted_price = predicted_prices[i][0]
    error = abs(true_price - predicted_price)
    error_pct = (error / true_price) * 100
    print(
        f"Property {i+1} | True: ${true_price:,.2f} | Predicted: ${predicted_price:,.2f} | Error: ${error:,.2f} ({error_pct:.1f}%)"
    )
