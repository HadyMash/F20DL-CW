# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis on the datasets

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# check the dataset path exists
if not os.path.exists("2025_05_27_PROACT_ALL_FORMS") or not os.path.isdir(
    "2025_05_27_PROACT_ALL_FORMS"
):
    raise ValueError("Please make sure the dataset exists and is unzipped")

# %%
# Load the proact ALSFRS file
df = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_ALSFRS.csv")
df.head()

# %%
df.describe()

# %% [markdown]
# Check how many rows don't have neither total or revise total

# %%
df[df["ALSFRS_Total"].isna() & df["ALSFRS_R_Total"].isna()].shape[0]

# %% [markdown]
# How many unique subject ids?

# %%
len(df["subject_id"].unique())

# %% [markdown]
# Remove the rows without any total

# %%
df.drop(df[df["ALSFRS_Total"].isna() & df["ALSFRS_R_Total"].isna()].index, inplace=True)

# %%
df.describe()

# %% [markdown]
# How many ALSFRS vs ALSFRS Revised?

# %%
df[df["ALSFRS_Total"].notna()].count()

# %%
df[df["ALSFRS_R_Total"].notna()].count()

# %% [markdown]
# # Converting ALSFRS (R) Totals to ALSFRS Totals
#
# Since we also have the questions, we can average out the 3 questions for the
# revised and treat it as the score for the normal Q10. Or we could go the other way around.
#
# ## Feasability test
# For patients with both scores, see how accurate it would be to convert
#
# copy df
# %%
df_test = df.copy()

# %% [markdown]
# Filter down to patients with both ALSFRS_Total and ALSFRS_R_Total

# %%
df_test = df_test[df_test["ALSFRS_Total"].notna() & df_test["ALSFRS_R_Total"].notna()]

# %%
df_test.describe()

# %%
df_test.info()

# %% [markdown]
# Convert by averaging the revised questions to normal Q10 score

# %%
for idx, row in df_test.iterrows():
    r1 = row["R_1_Dyspnea"]
    r2 = row["R_2_Orthopnea"]
    r3 = row["R_3_Respiratory_Insufficiency"]
    avg = (r1 + r2 + r3) / 3  # normal avg

    # subtract sum of r1-3 from R_Total
    total = row["ALSFRS_R_Total"]
    total -= r1 + r2 + r3

    # add the average
    total += avg
    df_test.loc[idx, "ALSFRS_Converted"] = total

# %%
df_test.describe()

# %%
# difference between converted and original
df_test["error"] = df_test["ALSFRS_Converted"] - df_test["ALSFRS_Total"]

# metrics
mae = df_test["error"].abs().mean()  # mean absolute error
mse = (df_test["error"] ** 2).mean()  # mean squared error
rmse = mse**0.5  # root mean squared error

# optional: correlation
corr = df_test[["ALSFRS_Converted", "ALSFRS_Total"]].corr().iloc[0, 1]

# %%
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Corr: {corr}")
# %% [markdown]
#
#
