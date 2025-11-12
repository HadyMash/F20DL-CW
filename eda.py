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
# # Exploratory Data Analysis on the datasets

# %%
import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
csv_file = [f for f in files if f.endswith(".csv")][0]

df1 = pd.read_csv(os.path.join(path1, csv_file))
df1.head()

# %%
df1.info()

# %%
df1.describe()


# %% [markdown]
# Show some historgrams
#


# %%
# Remove outliers using IQR method
def remove_outliers(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        if df_cleaned[col].dtype in ["int64", "float64"]:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df_cleaned[
                (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
            ]
    return df_cleaned


# %%
# Get numeric columns
numeric_cols = df1.select_dtypes(include=["int64", "float64"]).columns
df1_no_outliers = remove_outliers(df1, numeric_cols)

df1_no_outliers.hist(bins=50, figsize=(20, 15))
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df1.isnull().sum())

print("\nPercentage of null values:")
print((df1.isnull().sum() / len(df1)) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df1.isnull(), cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()

# %% [markdown]
# Correlation

# %%
# Select only numeric columns and calculate correlation with price
numeric_cols = df1.select_dtypes(include=["int64", "float64"]).columns
correlation_with_price = (
    remove_outliers(df1, numeric_cols)[numeric_cols]
    .corr()["price"]
    .sort_values(ascending=False)
)

print("Correlation with price:")
print(correlation_with_price)

# Visualize correlation with price
plt.figure(figsize=(10, 8))
correlation_with_price.drop("price").plot(kind="barh")
plt.title("Correlation of Numeric Features with Price")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
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
csv_file = [f for f in files if f.endswith(".csv")][0]

df2 = pd.read_csv(os.path.join(path2, csv_file))
df2.head()

# %%
df2.info()

# %%
df2.describe()

# %%
print("Total number of distinct zipcodes:", df2["zipcode"].nunique())
print("\nCount of properties per zipcode:")
print(df2["zipcode"].value_counts())

# %% [markdown]
# Correlation

# %%
# Select only numeric columns and calculate correlation with price
numeric_cols = df2.select_dtypes(include=["int64", "float64"]).columns
correlation_with_price = (
    remove_outliers(df2, numeric_cols)[numeric_cols]
    .corr()["median_sale_price"]
    .sort_values(ascending=False)
)

print("Correlation with price:")
print(correlation_with_price)

# Visualize correlation with price
plt.figure(figsize=(10, 8))
correlation_with_price.drop("price").plot(kind="barh")
plt.title("Correlation of Numeric Features with Price")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df2.isnull().sum())

print("\nPercentage of null values:")
print((df2.isnull().sum() / len(df2)) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df2.isnull(), cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()

# %%
# scatter plot of numeric variables vs price
# Get numeric columns excluding median_sale_price
numeric_cols = df2.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [col for col in numeric_cols if col != "median_sale_price"]

print(numeric_cols)

# remove outliers
df2fno = remove_outliers(df2, numeric_cols)

# Calculate number of rows and columns needed
n_cols = len(numeric_cols)
n_rows = (n_cols + 4) // 5  # Calculate rows needed for 5 columns

fig, axes = plt.subplots(n_rows, 5, figsize=(25, 5 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

# Create scatter plots for all numeric variables
for i, col in enumerate(numeric_cols):
    axes[i].scatter(df2fno[col], df2fno["median_sale_price"], alpha=0.1)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("median_sale_price")
    axes[i].set_title(f"median_sale_price vs {col}")
    axes[i].grid(True, alpha=0.1)

# Hide empty subplots
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Merging datasets
#
# Reduce df1 to only listings in df2's zipcodes
# %%
# df1 filtered
df1f = df1[(df1["zip_code"].isin(df2["zipcode"]))]
# remove outliers
numeric_cols = df1f.select_dtypes(include=["int64", "float64"]).columns
df1f = remove_outliers(df1f, numeric_cols)

# get rid of zip codes with less than 5 houses from both datasets

df1f = df1f.groupby("zip_code").filter(lambda x: len(x) >= 5)
df2 = df2[df2["zipcode"].isin(df1f["zip_code"].unique())]

df1f.head()

# %%
df1f.info()

# %%
df1f.describe()

# %% [markdown]
# Show some historgrams

# %%
# Get numeric columns
numeric_cols = df1f.select_dtypes(include=["int64", "float64"]).columns
df1f_no_outliers = remove_outliers(df1f, numeric_cols)

df1f_no_outliers.hist(bins=50, figsize=(15, 15))
plt.show()

# %%
# Check for null values in all columns
print("Null values in each column:")
print(df1f.isnull().sum())

print("\nPercentage of null values:")
print((df1f.isnull().sum() / len(df1f)) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df1f.isnull(), cbar=True, yticklabels=False, cmap="viridis")
plt.title("Missing Data Heatmap")
plt.show()

# %% [markdown]
# Remove null data

# %%
df1f = df1f.dropna()
df1f.info()

# %% [markdown]
# Correlation

# %%
# Select only numeric columns and calculate correlation with price
numeric_cols = df1f.select_dtypes(include=["int64", "float64"]).columns
correlation_with_price = (
    remove_outliers(df1f, numeric_cols)[numeric_cols]
    .corr()["price"]
    .sort_values(ascending=False)
)

print("Correlation with price:")
print(correlation_with_price)

# Visualize correlation with price
plt.figure(figsize=(7, 8))
correlation_with_price.drop("price").plot(kind="barh")
plt.title("Correlation of Numeric Features with Price")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# %% [markdown]
# Plot scatter plots of price vs other factors
# %%
# scatter plot of numeric variables vs price
# Get numeric columns excluding price
numeric_cols = df1f.select_dtypes(include=["int64", "float64"]).columns
numeric_cols = [col for col in numeric_cols if col != "price"]

# remove outliers
df1fno = remove_outliers(df1f, numeric_cols)

# Calculate number of rows and columns needed
n_cols = len(numeric_cols)
n_rows = (n_cols + 2) // 3  # Calculate rows needed for 3 columns

fig, axes = plt.subplots(n_rows, 3, figsize=(16.5, 6 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []

# Create scatter plots for all numeric variables
for i, col in enumerate(numeric_cols):
    axes[i].scatter(df1fno[col], df1fno["price"], alpha=0.1)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Price")
    axes[i].set_title(f"Price vs {col}")
    axes[i].grid(True, alpha=0.1)

# Hide empty subplots
for i in range(len(numeric_cols), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## Zip codes
#
# How much do house prices differ per zip code?

# %%
# Group by zip code and calculate statistics for price
zip_stats = (
    df1f.groupby("zip_code")["price"]
    .agg(
        [
            ("count", "count"),
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max"),
            ("range", lambda x: x.max() - x.min()),
        ]
    )
    .round(2)
)

print("Statistics for house prices by zip code:")
print("=" * 80)
print(f"\nNumber of zip codes: {len(zip_stats)}")
print(f"\nMean of zip code averages: ${zip_stats['mean'].mean():,.2f}")
print(f"Median of zip code averages: ${zip_stats['mean'].median():,.2f}")
print(f"Std of zip code averages: ${zip_stats['mean'].std():,.2f}")
print(f"\nMean of zip code std deviations: ${zip_stats['std'].mean():,.2f}")
print(f"Median of zip code std deviations: ${zip_stats['std'].median():,.2f}")
print(f"\nMean of zip code price ranges: ${zip_stats['range'].mean():,.2f}")
print(f"Median of zip code price ranges: ${zip_stats['range'].median():,.2f}")

print("\n" + "=" * 80)
print("\nTop 10 zip codes by average price:")
print(zip_stats.sort_values("mean", ascending=False).head(10))

print("\n" + "=" * 80)
print("\nTop 10 zip codes by price variability (std):")
print(zip_stats.sort_values("std", ascending=False).head(10))

print("\n" + "=" * 80)
print("\nTop 10 zip codes by number of listings:")
print(zip_stats.sort_values("count", ascending=False).head(10))


# %%
# Visualize zip code statistics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Distribution of mean prices by zip code
axes[0, 0].hist(zip_stats["mean"], bins=30, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Mean Price ($)")
axes[0, 0].set_ylabel("Number of Zip Codes")
axes[0, 0].set_title("Distribution of Mean Prices Across Zip Codes")
axes[0, 0].grid(True, alpha=0.3)

# Distribution of std by zip code
axes[0, 1].hist(zip_stats["std"], bins=30, edgecolor="black", alpha=0.7, color="orange")
axes[0, 1].set_xlabel("Standard Deviation ($)")
axes[0, 1].set_ylabel("Number of Zip Codes")
axes[0, 1].set_title("Distribution of Price Std Dev Across Zip Codes")
axes[0, 1].grid(True, alpha=0.3)

# Distribution of range by zip code
axes[0, 2].hist(
    zip_stats["range"], bins=30, edgecolor="black", alpha=0.7, color="green"
)
axes[0, 2].set_xlabel("Price Range ($)")
axes[0, 2].set_ylabel("Number of Zip Codes")
axes[0, 2].set_title("Distribution of Price Ranges Across Zip Codes")
axes[0, 2].grid(True, alpha=0.3)

# Distribution of listing counts by zip code
axes[1, 0].hist(zip_stats["count"], bins=30, edgecolor="black", alpha=0.7, color="red")
axes[1, 0].set_xlabel("Number of Listings")
axes[1, 0].set_ylabel("Number of Zip Codes")
axes[1, 0].set_title("Distribution of Listing Counts Across Zip Codes")
axes[1, 0].grid(True, alpha=0.3)

# Box plot of mean prices
axes[1, 1].boxplot(zip_stats["mean"], vert=True)
axes[1, 1].set_ylabel("Mean Price ($)")
axes[1, 1].set_title("Box Plot of Mean Prices Across Zip Codes")
axes[1, 1].grid(True, alpha=0.3)

# Scatter: mean vs std
axes[1, 2].scatter(zip_stats["mean"], zip_stats["std"], alpha=0.6, color="purple")
axes[1, 2].set_xlabel("Mean Price ($)")
axes[1, 2].set_ylabel("Standard Deviation ($)")
axes[1, 2].set_title("Mean Price vs Price Variability by Zip Code")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Save the cleaned datasets

# %%
os.makedirs("processed", exist_ok=True)

df1f.to_csv("processed/usa_real_estate.csv", index=False)
df2.to_csv("processed/zipcodes.csv", index=False)
