# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis on the datasets

# %%
# %matplotlib widget
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

plt.ion()

# %%
# check the dataset path exists
if not os.path.exists("2025_05_27_PROACT_ALL_FORMS") or not os.path.isdir(
    "2025_05_27_PROACT_ALL_FORMS"
):
    raise ValueError("Please make sure the dataset exists and is unzipped")

# %%
# Load the proact ALSFRS, Demographics and death files
df_ALS = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_ALSFRS.csv")
df_DEMO = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_DEMOGRAPHICS.csv")
df_DEATH = pd.read_csv("2025_05_27_PROACT_ALL_FORMS/PROACT_DEATHDATA.csv")
df_DEMO = df_DEMO.merge(df_DEATH, on="subject_id", how="left")
df = df_DEMO.merge(df_ALS, on="subject_id", how="left")

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
# Apply changes to main dataframe
# %%
for idx, row in df.iterrows():
    if pd.isna(row["ALSFRS_Total"]) and pd.notna(row["ALSFRS_R_Total"]):
        r1 = row["R_1_Dyspnea"]
        r2 = row["R_2_Orthopnea"]
        r3 = row["R_3_Respiratory_Insufficiency"]
        avg = (r1 + r2 + r3) / 3  # normal avg

        # subtract sum of r1-3 from R_Total
        total = row["ALSFRS_R_Total"]
        total -= r1 + r2 + r3

        # add the average
        total += avg
        df.loc[idx, "ALSFRS_Total"] = total
# %% [markdown]
# Convert DOB to Age and remove missing both
# %%
print(df["Age"].isnull().sum())
df["Age"] = df["Age"].fillna(-df["Date_of_Birth"] / 365.25)
print(df["Age"].isnull().sum())

# %% [markdown]
# Correlations to sex
# %%
model = smf.ols("Death_Days ~ C(Sex) + Age", data=df).fit()
print(model.summary())
# Map female to 0, male to 1
df["Sex_numeric"] = df["Sex"].map({"Female": 0, "Male": 1})
# Check for missing values
df = df[df["Sex_numeric"].notna()]
model = smf.logit("Sex_numeric ~ Age", data=df).fit()
print(model.summary())
# %% [markdown]
# Flatten Age distribution
# %%
df_sample_ALS0 = df.copy()
df_sample_ALS0 = df_sample_ALS0[df_sample_ALS0["ALSFRS_Delta"] < 1]
df_sample = df_sample_ALS0
df_sample["Age"].plot(kind="hist", bins=20, title="Age Distribution")

bins = list(range(30, 85, 5))
labels = [f"{b}-{b+5}" for b in bins[:-1]]
df["Age_bin"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
df_sample = (
    df.groupby("Age_bin")
    .apply(lambda x: x.sample(n=200, replace=True))
    .reset_index(drop=True)
)
df_sample["Age"].plot(kind="hist", bins=10, title="Age Distribution")
# %% [markdown]
# Correlations to time to death ;ie Age of onset ,male vs female ,

# %% markdown
# Plot of Age at study entry vs Days to Death if dead
# %%
print(df_DEATH["Death_Days"].mean())
print(df_DEATH["Death_Days"].std())

df_sample["Death_Days"] = np.where(
    df_sample["Death_Days"] >= 800, np.nan, df_sample["Death_Days"]
)

df_sample = df_sample[
    np.isfinite(df_sample["Death_Days"]) & np.isfinite(df_sample["Age"])
]

xy = np.vstack([df_sample["Age"], df_sample["Death_Days"]])
z = stats.gaussian_kde(xy)(xy)
plt.scatter(df_sample["Age"], df_sample["Death_Days"], c=z, s=20, cmap="Blues")
plt.colorbar(label="Density")
plt.ylabel("Days to Death")
plt.xlabel("Age at study entry")
plt.title("Age vs Days to Death with Density Coloring")

# %% [markdown]
# Histogram of Age for survived patients or over 800 days
# %%

df_sample["Death_Days"] = np.where(
    df_sample["Death_Days"] >= 800, np.nan, df_sample["Death_Days"]
)
df_sample["Death_Days"].isnull().sum()
df_survived = df_sample[df_sample["Death_Days"].isnull()]
df_survived.info()
plt.figure()
sns.histplot(df_survived["Age"], bins=10, kde=False)
plt.show()
# %% [markdown]
# Inital ALSFRS score vs Age at study entry
# %%
df_sample = df_sample[
    np.isfinite(df_sample["ALSFRS_Total"]) & np.isfinite(df_sample["Age"])
]
xy = np.vstack([df_sample["Age"], df_sample["ALSFRS_Total"]])
z = stats.gaussian_kde(xy)(xy)
plt.scatter(df_sample["Age"], df_sample["ALSFRS_Total"], c=z, s=20, cmap="Blues")
plt.colorbar(label="Density")
plt.ylabel("Initial ALSFRS Score")
plt.xlabel("Age at study entry")
plt.title("Age vs Initial ALSFRS Score with Density Coloring")

# %% [markdown]
# 3d graph of Age ,Initial ALSFRS score vs Days to Death -RUN IN NOTEBOOK
# %%
df_sample = df.copy()
df_sample["Death_Days"] = np.where(
    df_sample["Death_Days"] >= 800, np.nan, df_sample["Death_Days"]
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(
    df_sample["Age"],
    df_sample["ALSFRS_Total"],
    df_sample["Death_Days"],
    c=df_sample["Death_Days"],
    cmap="plasma",
)
ax.set_xlabel("Age at study entry")
ax.set_ylabel("Initial ALSFRS Score")
ax.set_zlabel("Days to Death")
ax.set_title("3D Scatter Plot of Age, Initial ALSFRS Score vs Days to Death")
plt.show()
# %%
