# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis on the datasets

# %%
import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import matplotlib.pyplot as plt
from kagglehub import KaggleDatasetAdapter

# %% [markdown]
# # USA Real Estate Dataset

# %%
# Download latest version
path1 = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")

print("Path to dataset files:", path1)

# %%
# List all files in the directory
files = os.listdir(path1)
print("Files in dataset:", files)

# %%
csv_file = [f for f in files if f.endswith('.csv')][0]

df1 = pd.read_csv(os.path.join(path1, csv_file))
df1.head()

# %%
df1.info()

# %%
df1.describe()


# %% [markdown]
# Show some historgrams

# %%
# Remove outliers using IQR method
def remove_outliers(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        if df_cleaned[col].dtype in ['int64', 'float64']:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned


# %%
# Get numeric columns
numeric_cols = df1.select_dtypes(include=['int64', 'float64']).columns
df1_no_outliers = remove_outliers(df1, numeric_cols)

df1_no_outliers.hist(bins=50, figsize=(20,15))
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df1.isnull().sum())

print("\nPercentage of null values:")
print((df1.isnull().sum() / len(df1)) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df1.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()


# %% [markdown]
# # HouseTS Multimodal Dataset
#
# Download the dataset

# %%
# Download latest version
path2 = kagglehub.dataset_download("shengkunwang/housets-dataset")

print("Path to dataset files:", path2)


# %%
# List all files in the directory
files = os.listdir(path2)
print("Files in dataset:", files)

# %%
csv_file = [f for f in files if f.endswith('.csv')][0]

df2 = pd.read_csv(os.path.join(path2, csv_file))
df2.head()

# %%
df2.info()

# %%
df2.describe()

# %%
print("Total number of distinct zipcodes:", df2['zipcode'].nunique())
print("\nCount of properties per zipcode:")
print(df2['zipcode'].value_counts())

# %% [markdown]
# # Merging datasets
#
# Reduce df1 to only listings in df2's zipcodes
# %%
# df1 filtered
df1f = df1[df1['zip_code'].isin(df2['zipcode'])]
df1f.head()

# %%
df1f.info()

# %%
df1f.describe()

# %% [markdown]
# Show some historgrams

# %%
# Get numeric columns
numeric_cols = df1f.select_dtypes(include=['int64', 'float64']).columns
df1f_no_outliers = remove_outliers(df1f, numeric_cols)

df1f_no_outliers.hist(bins=50, figsize=(20,15))
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df1f.isnull().sum())

print("\nPercentage of null values:")
print((df1f.isnull().sum() / len(df1f)) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df1f.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()
# %% [markdown]
#
#
