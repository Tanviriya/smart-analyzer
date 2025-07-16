# smartanalyzer.py project

#  Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Welcome message........
print(" Welcome to Smart Data Analyzer!")
print("Let's analyze your SAT vs GPA dataset like a pro.\n")

# Step 1: Asking for the CSV file path
file_path = input("Enter the path to your CSV file (e.g. 1.01. Simple linear regression.csv): ")

#  let's Check if file exists
if not os.path.exists(file_path):
    print(" File not found. Please check the path and try again.")
    exit()

# Loading the dataset......
data = pd.read_csv(file_path)
print("\n File loaded successfully!")

# Step 2: Display basic informations............
print("\n Dataset Info:")
print(data.info())

#  Step 3: Show first 5 rows of data
print("\n First 5 Rows:")
print(data.head())

# Step 4: Descriptive statistics......
print("\n Descriptive Statistics:")
print(data.describe())

# Step 5: Check for missing values
print("\n Missing Values:")
missing = data.isnull().sum()
missing_cols = missing[missing > 0]

if not missing_cols.empty:
    print(missing_cols)
    print("\n Please handle missing data using fillna() or dropna().")
else:
    print(" No missing values found!")

# Step 6: Histogram plots for each numeric column
print("\n Generating histograms...")

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True, color='skyblue', bins=10)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{col}_histogram.png")
    plt.close()
    print(f" Saved: {col}_histogram.png")

#  Step 7: Simple Linear Regression algo.......
choice = input("\nWould you like to perform linear regression on SAT vs GPA? (yes/no): ").lower()

if choice == 'yes':
    try:
        # Extractin X and y..
        X = data[['SAT']].dropna()
        y = data['GPA'].dropna()

        # Ensure matching length.......
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Training the model
        model = LinearRegression()
        model.fit(X, y)

        coef = model.coef_[0]
        intercept = model.intercept_

        print(f"\n Model trained successfully!")
        print(f" Equation: GPA = {coef:.4f} * SAT + {intercept:.4f}")

        # Plot regression
        plt.figure(figsize=(7, 4))
        plt.scatter(X, y, color='blue', label='Actual GPA')
        plt.plot(X, model.predict(X), color='red', label='Predicted GPA')
        plt.title("Linear Regression: SAT vs GPA")
        plt.xlabel("SAT Score")
        plt.ylabel("GPA")
        plt.legend()
        plt.tight_layout()
        plt.savefig("regression_plot.png")
        plt.close()
        print(" Regression plot saved as 'regression_plot.png'.")

    except Exception as e:
        print(f"Error: {e}")
else:
    print(" Skipping regression.")

# Final message
print("\n Analysis complete. Thank you for using Smart Data Analyzer!")
