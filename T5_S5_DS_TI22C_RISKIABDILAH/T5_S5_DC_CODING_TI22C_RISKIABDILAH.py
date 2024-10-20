# Step 1: Import the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Step 2: Load the Iris dataset
iris_dataset = load_iris()

# Creating a DataFrame
iris_df = pd.DataFrame(iris_dataset['data'], columns=iris_dataset['feature_names'])

# Adding species as a column and converting labels to species names
iris_df['species'] = iris_dataset['target']
iris_df['species'] = iris_df['species'].apply(lambda x: iris_dataset['target_names'][x])

# Step 3: Understanding the Data
# Dataset size
print("Dataset size:", iris_df.shape)

# Column names and data types
print("\nDataset columns:", iris_df.columns)
print("\nData types:", iris_df.dtypes)

# Display first 5 rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Descriptive statistics
print("\nDescriptive statistics:")
print(iris_df.describe())

# Step 4: Data Visualization
# Pairplot to visualize feature distribution and relationships between features
sns.pairplot(iris_df, hue='species')
plt.show()

# Boxplot to visualize distribution and outliers in features
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df.drop(columns='species'))
plt.title('Feature Distribution and Outliers')
plt.show()

# Correlation heatmap between features
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Features')
plt.show()
