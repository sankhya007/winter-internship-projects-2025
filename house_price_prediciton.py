import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== STEP 1: LOADING AND EXPLORING DATA ===")

# Lading the CSV file
df = pd.read_csv('housing.csv')

# Basic info about the data
print("Data loaded successfully!")
print(f"Dataset shape: {df.shape}")  #rows and columns
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

print("\n=== STEP 2: DATA VISUALIZATION ===")

# creating basic plots to understand the data
plt.figure(figsize=(15, 10))

# distribution of house prices
plt.subplot(2, 3, 1)
plt.hist(df['median_house_value'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')

# price vs income
plt.subplot(2, 3, 2)
plt.scatter(df['median_income'], df['median_house_value'], alpha=0.5)
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Price vs Income')

# price vs rooms
plt.subplot(2, 3, 3)
plt.scatter(df['total_rooms'], df['median_house_value'], alpha=0.5)
plt.xlabel('Total Rooms')
plt.ylabel('House Price')
plt.title('Price vs Total Rooms')

# price vs bedrooms
plt.subplot(2, 3, 4)
plt.scatter(df['total_bedrooms'], df['median_house_value'], alpha=0.5)
plt.xlabel('Total Bedrooms')
plt.ylabel('House Price')
plt.title('Price vs Bedrooms')

# price vs population
plt.subplot(2, 3, 5)
plt.scatter(df['population'], df['median_house_value'], alpha=0.5)
plt.xlabel('Population')
plt.ylabel('House Price')
plt.title('Price vs Population')

# price vs house age
plt.subplot(2, 3, 6)
plt.scatter(df['housing_median_age'], df['median_house_value'], alpha=0.5)
plt.xlabel('House Age')
plt.ylabel('House Price')
plt.title('Price vs House Age')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== STEP 3: DATA PREPROCESSING ===")

# handling the missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# filling the missing values with median
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
print("✅ Missing values handled")

# handling categorical data (ocean_proximity)
print(f"\nOcean Proximity values: {df['ocean_proximity'].unique()}")

# converting categorical to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean')
print("✅ Categorical data converted to numerical")

print(f"New dataset shape: {df.shape}")
print(f"New columns: {list(df.columns)}")

print("\n=== STEP 4: PREPARING FOR MACHINE LEARNING ===")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# defining features (X) and target (y)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data scaled successfully")

print("\n=== STEP 5: TRAINING LINEAR REGRESSION MODEL ===")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# creating and training the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✅ Model trained successfully")

# making predictions
y_pred = model.predict(X_test_scaled)

# calculating performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE:")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# showingw what R² means
print(f"\n💡 R² Score of {r2:.4f} means the model explains {r2*100:.1f}% of the variance in house prices")

print("\n=== STEP 6: VISUALIZING RESULTS ===")

# create results visualization
plt.figure(figsize=(15, 5))

#actual vs predicted prices
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')

# residuals plot
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# feature importance
plt.subplot(1, 3, 3)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_)
}).sort_values('importance', ascending=False)

plt.barh(feature_importance['feature'][:8], feature_importance['importance'][:8])
plt.xlabel('Importance (Absolute Coefficient)')
plt.title('Top 8 Most Important Features')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎯 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10))

print("\n=== STEP 7: MAKING PREDICTIONS ===")

# create sample data for prediction
# first, let's see what ocean proximity columns we have
ocean_columns = [col for col in df.columns if 'ocean_' in col]
print(f"Ocean columns: {ocean_columns}")

# Create a sample house
sample_data = {
    'longitude': [-122.25],
    'latitude': [37.85],
    'housing_median_age': [30],
    'total_rooms': [2000],
    'total_bedrooms': [400],
    'population': [1200],
    'households': [500],
    'median_income': [5.0]
}

# Add ocean proximity columns (set all to 0 initially)
for col in ocean_columns:
    sample_data[col] = [0]

# Set the correct ocean proximity (usually 'ocean_NEAR BAY' for these coordinates)
if 'ocean_NEAR BAY' in ocean_columns:
    sample_data['ocean_NEAR BAY'] = [1]

# Convert to DataFrame
sample_df = pd.DataFrame(sample_data)

# Ensure all columns are in correct order
sample_df = sample_df[X.columns]

# Scale the sample data
sample_scaled = scaler.transform(sample_df)

# Make prediction
predicted_price = model.predict(sample_scaled)[0]

print(f"🏠 SAMPLE HOUSE PREDICTION:")
print(f"Features: 30-year-old house, 2000 rooms, 400 bedrooms, $50,000 income, near bay")
print(f"Predicted Price: ${predicted_price:,.2f}")

# Compare with actual average
actual_avg = df['median_house_value'].mean()
print(f"Average California House Price: ${actual_avg:,.2f}")

print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")