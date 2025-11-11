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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# Load data

# %%
DATASET_DIR = "2025_05_27_PROACT_ALL_FORMS"
# Load the proact ALSFRS, Demographics and death files
df_alsfrs = pd.read_csv(f"{DATASET_DIR}/processed/ALSFRS.csv")
df_demo = pd.read_csv(f"{DATASET_DIR}/processed/DEMOGRAPHICS.csv")
df_timeseries = pd.read_csv(f"{DATASET_DIR}/processed/ALSFRS_timeseries.csv")

# %%
df_alsfrs.head()

# %%
df_demo.head()

# %%
df_timeseries.head()

# %% [markdown]
# ## Linear Regressor
