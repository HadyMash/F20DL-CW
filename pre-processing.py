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
# # Imports and data loading

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# %%
DATASET_DIR = "2025_05_27_PROACT_ALL_FORMS"
# Load the proact ALSFRS, Demographics and death files
df_ALSFRS = pd.read_csv(f"{DATASET_DIR}/PROACT_ALSFRS.csv")
df_DEMO = pd.read_csv(f"{DATASET_DIR}/PROACT_DEMOGRAPHICS.csv")
df_DEATH = pd.read_csv(f"{DATASET_DIR}/PROACT_DEATHDATA.csv")
df_DEMO = df_DEMO.merge(df_DEATH, on="subject_id", how="left")

# %% [markdown]
# # Pre-processing
#
# This notebook pre-processes the data such that it is in a usable format for the model training. This notebook:
#
# 1. Converts ALSFRS Revised scores to ALSFRS scores
# 2. Normalizes the ALSFRS scores in another column
# 3. Convert DOB to Age
# 4. Remove patients missing both age and DOB
# 5. Consolidate Q5a and Q5b cutting columns into a single Q5_Cutting column with a gastrostomy boolean
# 6. Linearly interpolates missing ALSFRS scores for each patient for time-step based models (this is saved to another csv)
#
# ## ALSFRS Revised conversion
#
# See the EDA notebook for more details about this approach and accuracy

# %%
df_ALSFRS.describe()

# %%
for idx, row in df_ALSFRS.iterrows():
    if pd.isna(row["ALSFRS_Total"]) and pd.notna(row["ALSFRS_R_Total"]):
        r1 = row["R_1_Dyspnea"]
        r2 = row["R_2_Orthopnea"]
        r3 = row["R_3_Respiratory_Insufficiency"]
        avg = (r1 + r2 + r3) / 3  # normal avg

        if np.isnan(avg):
            continue

        avg = round(avg)

        # subtract sum of r1-3 from R_Total
        total = row["ALSFRS_R_Total"]
        total -= r1 + r2 + r3

        # add the average
        total += avg
        df_ALSFRS.loc[idx, "ALSFRS_Total"] = total

# %% [markdown]
# Remove rows where both are null

# %%
print(df_ALSFRS["ALSFRS_Total"].isna().sum())
df_ALSFRS = df_ALSFRS.dropna(subset=["ALSFRS_Total"])
print(df_ALSFRS["ALSFRS_Total"].isna().sum())

# %%
df_ALSFRS.describe()

# %% [markdown]
# Clamp ALSFRS scores between 0 and 40

# %%
df_ALSFRS["ALSFRS_Total"] = df_ALSFRS["ALSFRS_Total"].clip(lower=0, upper=40)

# %%
df_ALSFRS.describe()

# %% [markdown]
# ## Normalize ALSFRS question scores and total
#
# Normalize using min-max normalization since the ALSFRS scores on a fixed scale (0-40)

# %%
min = 0
max = 40

df_ALSFRS["ALSFRS_Total_norm"] = (df_ALSFRS["ALSFRS_Total"] - min) / (max - min)

# %%
df_ALSFRS.describe()

# %% [markdown]
# ## Remove patients missing age and DOB

# %%
df_DEMO.describe()

# %%
df_DEMO = df_DEMO[~(df_DEMO["Age"].isna() & df_DEMO["Date_of_Birth"].isna())]

# %% [markdown]
# ## Convert DOB to Age

# %%
df_DEMO["Age"] = df_DEMO["Age"].fillna(-df_DEMO["Date_of_Birth"] / 365.25)

# %%
print(df_DEMO["Age"].isna().sum())

# %%
df_DEMO.describe()

# %% [markdown]
# ## Consolidate Q5 Cutting columns
#
# Consolidate Q5a_Cutting_without_Gastrostomy and Q5b_Cutting_with_Gastrostomy into a single Q5_Cutting column, and add a boolean gastrostomy column to indicate which was used.

# %%
# Create gastrostomy boolean column
# If Q5b has a value, patient has gastrostomy; otherwise they don't
df_ALSFRS["gastrostomy"] = df_ALSFRS["Q5b_Cutting_with_Gastrostomy"].notna()

# Create consolidated Q5_Cutting column
# Use Q5b if available (patient has gastrostomy), otherwise use Q5a
df_ALSFRS["Q5_Cutting"] = df_ALSFRS["Q5b_Cutting_with_Gastrostomy"].fillna(
    df_ALSFRS["Q5a_Cutting_without_Gastrostomy"]
)

# Drop the original Q5a and Q5b columns
df_ALSFRS = df_ALSFRS.drop(
    columns=["Q5a_Cutting_without_Gastrostomy", "Q5b_Cutting_with_Gastrostomy"]
)

print(f"Gastrostomy distribution:")
print(df_ALSFRS["gastrostomy"].value_counts())
print(f"\nQ5_Cutting missing values: {df_ALSFRS['Q5_Cutting'].isna().sum()}")

# %%
df_ALSFRS.head()

# %%
df_ALSFRS.describe()

# %%
df_ALSFRS.info()

# %% [markdown]
# # Save dataframes

# %%
# make a processed directory if not exists
os.makedirs(f"{DATASET_DIR}/processed", exist_ok=True)

# save the csvs
df_ALSFRS.to_csv(f"{DATASET_DIR}/processed/ALSFRS.csv", index=False)
df_DEMO.to_csv(f"{DATASET_DIR}/processed/DEMOGRAPHICS.csv", index=False)
# %% [markdown]
# # ALSFRS Interpolation

# %% [markdown]
# ## Temporal Data Modeling Approach
#
# Linearly interpolate time-series data for each patient in 1 month (30 day timesteps). If multiple visits exist for the same timestep, the last visit value is used.


# %%
def delta_to_timestep(delta_days):
    """Convert days to month time-step using rounding"""
    if pd.isna(delta_days):
        return None
    return int(round(delta_days / 30.0))


def create_patient_timeseries(patient_data):
    """
    Create a time series for a patient with linear interpolation.

    Steps:
    1. Convert delta to timestep
    2. Keep the latest entry for each timestep
    3. Interpolate missing timesteps

    Args:
        patient_data: DataFrame containing data for a single patient

    Returns:
        tuple: (numpy array with interpolated ALSFRS_Total values, max_timestep)
    """
    # Step 1: Add time-step column
    patient_data = patient_data.copy()
    patient_data["timestep"] = patient_data["ALSFRS_Delta"].apply(delta_to_timestep)

    # Remove rows with missing time-step or negative time-step
    patient_data = patient_data[patient_data["timestep"].notna()]
    patient_data = patient_data[patient_data["timestep"] >= 0]

    if len(patient_data) == 0:
        return np.array([]), -1

    # Step 2: Keep the latest entry for each time-step
    # Sort by ALSFRS_Delta to ensure we keep the latest measurement
    patient_data = patient_data.sort_values("ALSFRS_Delta")
    # Group by timestep and take the last (latest) entry
    patient_data = patient_data.groupby("timestep").last().reset_index()

    # Determine the maximum timestep for this patient
    max_timestep = int(patient_data["timestep"].max())

    # Step 3: Create timeseries array and fill with available data
    timeseries = np.full(max_timestep + 1, np.nan)
    for _, row in patient_data.iterrows():
        timestep = int(row["timestep"])
        timeseries[timestep] = row["ALSFRS_Total"]

    # Perform linear interpolation for missing values
    valid_indices = np.where(~np.isnan(timeseries))[0]

    if len(valid_indices) > 0:
        # Interpolate only between first and last valid points
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        if first_valid < last_valid:
            # Interpolate missing values between first and last valid points
            timeseries[first_valid : last_valid + 1] = np.interp(
                np.arange(first_valid, last_valid + 1),
                valid_indices,
                timeseries[valid_indices],
            )

            # Round interpolated values to integers (ALSFRS scores are whole numbers)
            timeseries[first_valid : last_valid + 1] = np.round(
                timeseries[first_valid : last_valid + 1]
            )

    return timeseries, max_timestep


# %% [markdown]
# Create interpolated time series for all patients

# %%
# Get unique patient IDs
patient_ids = df_ALSFRS["subject_id"].unique()
print(f"Total number of patients: {len(patient_ids)}")

# List to store all interpolated records
interpolated_records = []

# Dictionary to store processed patient data for visualization
processed_patient_data = {}

# Track statistics
max_timesteps_overall = 0
timestep_counts = {}

for patient_id in patient_ids:
    # Get data for this patient
    patient_data = df_ALSFRS[df_ALSFRS["subject_id"] == patient_id].copy()

    # Process patient data (add timestep, filter, keep latest per timestep)
    patient_data["timestep"] = patient_data["ALSFRS_Delta"].apply(delta_to_timestep)
    patient_data = patient_data[patient_data["timestep"].notna()]
    patient_data = patient_data[patient_data["timestep"] >= 0]

    if len(patient_data) == 0:
        continue

    patient_data = patient_data.sort_values("ALSFRS_Delta")
    patient_data = patient_data.groupby("timestep").last().reset_index()

    # Store processed data for visualization
    processed_patient_data[patient_id] = patient_data

    # Determine the maximum timestep for this patient
    max_timestep = int(patient_data["timestep"].max())
    if max_timestep > max_timesteps_overall:
        max_timesteps_overall = max_timestep

    # Create timeseries array and fill with available data
    timeseries = np.full(max_timestep + 1, np.nan)
    for _, row in patient_data.iterrows():
        timestep = int(row["timestep"])
        timeseries[timestep] = row["ALSFRS_Total"]

    # Perform linear interpolation for missing values
    valid_indices = np.where(~np.isnan(timeseries))[0]

    if len(valid_indices) > 0:
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        if first_valid < last_valid:
            timeseries[first_valid : last_valid + 1] = np.interp(
                np.arange(first_valid, last_valid + 1),
                valid_indices,
                timeseries[valid_indices],
            )
            timeseries[first_valid : last_valid + 1] = np.round(
                timeseries[first_valid : last_valid + 1]
            )

    # Get a template row from the first record for this patient (to preserve other columns)
    if len(patient_data) > 0:
        template_row = patient_data.iloc[0].to_dict()

        # Create a record for each timestep
        for timestep in range(max_timestep + 1):
            if not np.isnan(timeseries[timestep]):
                record = template_row.copy()
                record["subject_id"] = patient_id
                record["timestep"] = timestep
                record["ALSFRS_Total"] = timeseries[timestep]
                # Update normalized score as well
                record["ALSFRS_Total_norm"] = timeseries[timestep] / 40.0
                # Update ALSFRS_Delta to reflect the timestep (in days)
                record["ALSFRS_Delta"] = timestep * 30.0
                interpolated_records.append(record)

                # Track timestep statistics
                timestep_counts[timestep] = timestep_counts.get(timestep, 0) + 1

print(f"Created {len(interpolated_records)} interpolated records")
print(f"Maximum timestep across all patients: {max_timesteps_overall} months")


# %%
# Convert to DataFrame format
# Each row represents one timestep for a patient, maintaining the original column structure
df_ALSFRS_timeseries = pd.DataFrame(interpolated_records)

# Select relevant columns in a logical order
important_cols = [
    "subject_id",
    "timestep",
    "ALSFRS_Delta",
    "ALSFRS_Total",
    "ALSFRS_Total_norm",
]
# Get any remaining columns
other_cols = [col for col in df_ALSFRS_timeseries.columns if col not in important_cols]
# Reorder columns
df_ALSFRS_timeseries = df_ALSFRS_timeseries[important_cols + other_cols]

print("Interpolated ALSFRS Time Series DataFrame:")
print(df_ALSFRS_timeseries.head(20))
print(f"\nShape: {df_ALSFRS_timeseries.shape}")
print(f"\nData types:\n{df_ALSFRS_timeseries.dtypes}")
print(f"\nSample patient data (first patient, all timesteps):")
first_patient = df_ALSFRS_timeseries["subject_id"].iloc[0]
print(df_ALSFRS_timeseries[df_ALSFRS_timeseries["subject_id"] == first_patient])


# %% [markdown]
# ### Visualize interpolation for sample patients

# %%
# Visualize interpolation for a few sample patients
num_samples = 3
sample_patient_ids = patient_ids[:num_samples]

fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))
if num_samples == 1:
    axes = [axes]

for idx, patient_id in enumerate(sample_patient_ids):
    # Use the SAME processed data that was used to create the timeseries
    patient_data = processed_patient_data[patient_id]

    # Extract original (kept) data points
    original_timesteps = patient_data["timestep"].astype(int).tolist()
    original_scores = patient_data["ALSFRS_Total"].tolist()

    # Get interpolated data from the dataframe
    patient_timeseries = df_ALSFRS_timeseries[
        df_ALSFRS_timeseries["subject_id"] == patient_id
    ]
    interpolated_timesteps = patient_timeseries["timestep"].values
    interpolated_scores = patient_timeseries["ALSFRS_Total"].values

    # Separate interpolated points (those not in original data) from original points
    interpolated_only_timesteps = []
    interpolated_only_scores = []
    for i in range(len(interpolated_timesteps)):
        t = interpolated_timesteps[i]
        if t not in original_timesteps:
            interpolated_only_timesteps.append(t)
            interpolated_only_scores.append(interpolated_scores[i])

    # Determine the max timestep for this patient
    max_timestep_patient = int(interpolated_timesteps.max())

    # Plot
    ax = axes[idx]
    # Plot original data points in red
    ax.scatter(
        original_timesteps,
        original_scores,
        c="red",
        s=150,
        zorder=5,
        label="Original Data",
        marker="o",
    )
    # Plot interpolated points in blue
    if len(interpolated_only_timesteps) > 0:
        ax.scatter(
            interpolated_only_timesteps,
            interpolated_only_scores,
            c="blue",
            s=100,
            zorder=4,
            label="Interpolated",
            marker="s",
        )
    # Connect all points with a line for visualization
    ax.plot(
        interpolated_timesteps,
        interpolated_scores,
        "gray",
        linewidth=1,
        alpha=0.5,
        zorder=1,
    )

    ax.set_xlabel("Time Step (Months)", fontsize=12)
    ax.set_ylabel("ALSFRS Total Score", fontsize=12)
    ax.set_title(
        f"Patient {patient_id}: ALSFRS Score Over Time ({max_timestep_patient + 1} months)",
        fontsize=14,
    )
    ax.set_xlim(-0.5, max_timestep_patient + 0.5)
    ax.set_ylim(0, 40)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Statistics about the interpolated data

# %%
# Statistics about the interpolated data
print("Statistics about interpolated data:")
print(f"Total number of records: {len(df_ALSFRS_timeseries)}")
print(
    f"Total number of unique patients: {df_ALSFRS_timeseries['subject_id'].nunique()}"
)

# Find the maximum timestep across all patients
max_timestep_all = int(df_ALSFRS_timeseries["timestep"].max())
print(f"Maximum timestep across all patients: {max_timestep_all} months")

# Count records per timestep
print("\nRecords per timestep (showing first 20 timesteps):")
timestep_counts_df = df_ALSFRS_timeseries.groupby("timestep").size()
total_patients = df_ALSFRS_timeseries["subject_id"].nunique()

# Use conditional to avoid calling built-in min() which is shadowed
display_limit = 21 if max_timestep_all >= 21 else max_timestep_all + 1
for timestep in range(display_limit):
    count = timestep_counts_df.get(timestep, 0)
    pct = (count / total_patients) * 100
    print(f"  t{timestep}: {count}/{total_patients} patients ({pct:.2f}%)")

if max_timestep_all >= 21:
    print(f"  ... (showing only first 20 timesteps)")
    print(f"  Full range: t0 to t{max_timestep_all}")

# Show distribution of patients by number of timesteps
patient_timestep_counts = df_ALSFRS_timeseries.groupby("subject_id").size()
print("\nDistribution of patients by number of timesteps:")
print(patient_timestep_counts.describe())

# Show how many patients have different ranges of timesteps
print("\nPatients by timestep range:")
print(f"  1-6 months: {(patient_timestep_counts <= 6).sum()} patients")
print(
    f"  7-12 months: {((patient_timestep_counts > 6) & (patient_timestep_counts <= 12)).sum()} patients"
)
print(
    f"  13-24 months: {((patient_timestep_counts > 12) & (patient_timestep_counts <= 24)).sum()} patients"
)
print(f"  25+ months: {(patient_timestep_counts > 24).sum()} patients")

# Display summary statistics for ALSFRS_Total across all timesteps
print("\nSummary statistics for ALSFRS_Total across all records:")
print(df_ALSFRS_timeseries["ALSFRS_Total"].describe())

# Example: Show how to create a feature matrix for a single patient
print("\n--- Example: Creating features × timesteps matrix for a patient ---")
example_patient = df_ALSFRS_timeseries["subject_id"].iloc[0]
patient_data = df_ALSFRS_timeseries[
    df_ALSFRS_timeseries["subject_id"] == example_patient
]
print(f"Patient {example_patient}:")
print(f"Number of timesteps: {len(patient_data)}")
print("\nFeature values across timesteps:")
print(
    patient_data[["timestep", "ALSFRS_Total", "ALSFRS_Total_norm"]].to_string(
        index=False
    )
)


# %% [markdown]
# ### Save interpolated time series data

# %%
# Save the interpolated time series data
output_path = f"{DATASET_DIR}/processed/ALSFRS_timeseries.csv"
df_ALSFRS_timeseries.to_csv(output_path, index=False)
print(f"Saved interpolated time series data to: {output_path}")
print(f"Shape: {df_ALSFRS_timeseries.shape}")
