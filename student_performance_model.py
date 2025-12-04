# ---------------------------------------------
# TASK 1: LOAD AND EXPLORE THE DATASET
# Week 7: Assignment
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable inline plotting if running in Jupyter Notebook
# %matplotlib inline

# Error handling for loading dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add species column
    df["species"] = iris.target
    df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    print("Dataset loaded successfully!")

except Exception as e:
    print("Error loading dataset:", e)


# ---- Display the first few rows ----
print("\nFIRST 5 ROWS:\n")
print(df.head())


# ---- Check data structure ----
print("\nDATA TYPES:\n")
print(df.dtypes)

print("\nMISSING VALUES PER COLUMN:\n")
print(df.isnull().sum())


# ---- Clean dataset (fill missing values if any) ----
df = df.fillna(df.mean(numeric_only=True))
print("\nMissing values handled (if any).\n")


# ---------------------------------------------
# TASK 2: BASIC DATA ANALYSIS
# ---------------------------------------------

# ---- Basic statistics ----
print("\nSTATISTICS SUMMARY:\n")
print(df.describe())


# ---- Group by species and compute mean ----
print("\nMEAN VALUES GROUPED BY SPECIES:\n")
group_means = df.groupby("species").mean()
print(group_means)


# ---------------------------------------------
# TASK 3: DATA VISUALIZATION
# ---------------------------------------------

sns.set(style="whitegrid")  # Better chart style

# ---- 1. Line Chart: Sepal Length over index (pseudo time-series) ----
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["sepal length (cm)"], linewidth=2)
plt.title("Simulated Time-Series: Sepal Length Over Index", fontsize=16)
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ---- 2. Bar Chart: Average Petal Length per Species ----
plt.figure(figsize=(10, 6))
plt.bar(group_means.index, group_means["petal length (cm)"])
plt.title("Average Petal Length per Species", fontsize=16)
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()


# ---- 3. Histogram: Sepal Width ----
plt.figure(figsize=(10, 6))
plt.hist(df["sepal width (cm)"], bins=15, edgecolor="black")
plt.title("Histogram of Sepal Width", fontsize=16)
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ---- 4. Scatter Plot: Sepal Length vs Petal Length ----
plt.figure(figsize=(10, 6))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], s=50)
plt.title("Scatter Plot: Sepal Length vs Petal Length", fontsize=16)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

