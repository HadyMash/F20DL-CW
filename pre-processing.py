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
# Load the proact ALSFRS, Demographics and death files
df_ALSFRS = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_ALSFRS.csv")
df_DEMO = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_DEMOGRAPHICS.csv")
df_DEATH = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_DEATHDATA.csv")
df_DEMO = df_DEMO.merge(df_DEATH, on="subject_id", how="left")

# %% [markdown]
# # Pre-processing
#
# This notebook pre-processes the data such that it is in a usable format for the model training. This notebook:
#
# 1. Converts ALSFRS Revised scores to ALSFRS scores
# 2. Interpolate missing data linearly (as done in [this paper](https://www.sciencedirect.com/science/article/pii/S2666521225000511))
# 3. Normalizes ALSFRS scores (including converted)
# 4. Convert DOB to Age
# 5. Remove patients missing both age and DOB
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
# ## ALSFRS Interpolation
#
# ### Delta
# Firstly, how spaced out are entries? What's the range of the delta, and how many entries per patient?

# %%
print(
    f"{df_ALSFRS['ALSFRS_Delta'].min()} to {df_ALSFRS['ALSFRS_Delta'].max()} (range of {df_ALSFRS['ALSFRS_Delta'].max() - df_ALSFRS['ALSFRS_Delta'].min()})"
)

# %%
print(df_ALSFRS.groupby("subject_id").size().mean())

# %% [markdown]
# What's the average first entry delta?

# %%
print(df_ALSFRS.groupby("subject_id")["ALSFRS_Delta"].min().mean())

# %% [markdown]
# What's the average last entry delta?

# %%
print(df_ALSFRS.groupby("subject_id")["ALSFRS_Delta"].max().mean())

# %% [markdown]
# Check if any patients have less than 2 entries

# %%
print(df_ALSFRS.groupby("subject_id")["ALSFRS_Delta"].size().min())

# %% [markdown]
# Remove any patients with less than 2 entries

# %%
mask = df_ALSFRS.groupby("subject_id")["ALSFRS_Delta"].transform("size") >= 4
df_ALSFRS = df_ALSFRS[mask]

# %% [markdown]
# Remove any patients with negative ALSFRS scores

# %%
mask = df_ALSFRS.groupby("subject_id")["ALSFRS_Total"].transform("min") >= 0
df_ALSFRS = df_ALSFRS[mask]

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
#
