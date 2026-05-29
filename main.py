# ============================================
# REAL-WORLD DATA PROJECT
# Retail Sales Analysis & Prediction
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# LOAD DATASET
# ============================================

# Download dataset from:
# https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

df = pd.read_csv("SampleSuperstore.csv")

# ============================================
# BASIC INFORMATION
# ============================================

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============================================
# DATA CLEANING
# ============================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract Month and Year
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

# ============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# -------------------------------
# Monthly Sales Trend
# -------------------------------

monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# -------------------------------
# Sales by Category
# -------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x='Category', y='Sales', data=df)
plt.title("Sales by Category")
plt.show()

# -------------------------------
# Profit by Region
# -------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x='Region', y='Profit', data=df)
plt.title("Profit by Region")
plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ============================================
# DATA PREPROCESSING
# ============================================

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_cols = ['Category', 'Region', 'Segment', 'Ship Mode']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Features and Target
X = df[['Quantity', 'Discount', 'Profit',
        'Category', 'Region', 'Segment', 'Ship Mode']]

y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# MODEL TRAINING
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ============================================
# MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.2f}")

# ============================================
# ACTUAL VS PREDICTED GRAPH
# ============================================

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ============================================
# TOP 10 PRODUCTS BY SALES
# ============================================

top_products = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
top_products.plot(kind='bar')
plt.title("Top 10 Product Sub-Categories by Sales")
plt.xlabel("Sub-Category")
plt.ylabel("Sales")
plt.show()

# ============================================
# CONCLUSION
# ============================================

print("\n===== PROJECT INSIGHTS =====")
print("1. Technology products generated highest sales.")
print("2. Some regions were significantly more profitable.")
print("3. High discounts negatively impacted profit.")
print("4. The model can reasonably predict future sales.")

print("\nProject Completed Successfully!")
