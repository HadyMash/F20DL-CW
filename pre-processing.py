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
from pathlib import Path

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
print(df_DEMO["Age"].isnull().sum())

# %%
df_DEMO.describe()

# %% [markdown]
# # Save dataframes

# %%
# make a processed directory if not exists
os.makedirs(f"{DATASET_DIR}/processed", exist_ok=True)

# save the csvs
df_ALSFRS.to_csv(f"{DATASET_DIR}/processed/ALSFRS.csv", index=False)
df_DEMO.to_csv(f"{DATASET_DIR}/processed/DEMOGRAPHICS.csv", index=False)
# %% [markdown]
#
