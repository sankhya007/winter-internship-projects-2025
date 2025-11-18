# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# import warnings
# warnings.filterwarnings('ignore')

# print("=== HOUSE PRICE PREDICTION USING LINEAR REGRESSION ===")
# print("=" * 60)

# # 1. Debug file location
# print("\n1. CHECKING FILE LOCATION...")
# print(f"Current directory: {os.getcwd()}")
# print("Files in current directory:")
# for file in os.listdir('.'):
#     print(f"  - {file}")

# # 2. Try multiple ways to load the file
# print("\n2. ATTEMPTING TO LOAD DATASET...")

# df = None
# load_methods = [
#     # Method 1: Direct read
#     lambda: pd.read_csv('housing.csv'),
#     # Method 2: With different encoding
#     lambda: pd.read_csv('housing.csv', encoding='utf-8'),
#     # Method 3: With latin-1 encoding
#     lambda: pd.read_csv('housing.csv', encoding='latin-1'),
#     # Method 4: Check if file exists first
#     lambda: pd.read_csv('housing.csv') if os.path.exists('housing.csv') else None
# ]

# for i, method in enumerate(load_methods, 1):
#     try:
#         print(f"  Trying method {i}...")
#         df = method()
#         if df is not None:
#             print(f"SUCCESS: Dataset loaded using method {i}")
#             break
#     except Exception as e:
#         print(f"  Method {i} failed: {e}")
#         continue

# if df is None:
#     print("\nERROR: All loading methods failed!")
#     print("Let's create sample data instead...")
    
#     # Create sample data
#     np.random.seed(42)
#     sample_data = {
#         'longitude': np.random.uniform(-124, -114, 1000),
#         'latitude': np.random.uniform(32, 42, 1000),
#         'housing_median_age': np.random.randint(1, 50, 1000),
#         'total_rooms': np.random.randint(100, 10000, 1000),
#         'total_bedrooms': np.random.randint(50, 5000, 1000),
#         'population': np.random.randint(200, 5000, 1000),
#         'households': np.random.randint(100, 3000, 1000),
#         'median_income': np.random.uniform(1, 10, 1000),
#         'median_house_value': np.random.randint(50000, 500000, 1000),
#         'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN'], 1000)
#     }
#     df = pd.DataFrame(sample_data)
#     print("SUCCESS: Sample dataset created")
    
#     # Save it for future use
#     try:
#         df.to_csv('housing_sample.csv', index=False)
#         print("Sample data saved as housing_sample.csv")
#     except:
#         print("Could not save sample data file")

# print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# # 3. Data Preprocessing
# print("\n3. DATA PREPROCESSING...")

# # Handle missing values
# if df.isnull().sum().sum() > 0:
#     print("Handling missing values...")
#     imputer = SimpleImputer(strategy='median')
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# # Handle categorical data
# if 'ocean_proximity' in df.columns:
#     print(f"Ocean Proximity categories: {df['ocean_proximity'].unique()}")
#     df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='location')
#     print("Categorical data encoded")

# # Create new features
# df['rooms_per_household'] = df['total_rooms'] / df['households']
# df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
# df['population_per_household'] = df['population'] / df['households']

# print(f"Final dataset shape: {df.shape}")

# # 4. Feature Analysis
# print("\n4. FEATURE ANALYSIS...")

# X = df.drop('median_house_value', axis=1)
# y = df['median_house_value']

# # Calculate correlations
# correlations = df.corr()['median_house_value'].sort_values(ascending=False)
# print("\nFEATURE CORRELATIONS WITH HOUSE PRICE:")
# for feature, corr in correlations.items():
#     if feature != 'median_house_value' and abs(corr) > 0.1:
#         print(f"   {feature:25}: {corr:+.3f}")

# # 5. Prepare Data for Modeling
# print("\n5. PREPARING DATA FOR MODELING...")

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(f"Training set: {X_train.shape}")
# print(f"Testing set: {X_test.shape}")

# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# print("Features scaled successfully")

# # 6. Train Linear Regression Model
# print("\n6. TRAINING LINEAR REGRESSION MODEL...")

# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)

# print("Model trained successfully")

# # Make predictions
# y_pred = lr_model.predict(X_test_scaled)

# # Calculate metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print("\nMODEL PERFORMANCE METRICS:")
# print(f"   Mean Absolute Error (MAE): ${mae:,.2f}")
# print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
# print(f"   R-squared (R2) Score: {r2:.4f}")
# print(f"   Model Accuracy: {r2*100:.1f}% of variance explained")

# # 7. Feature Importance
# print("\n7. FEATURE IMPORTANCE ANALYSIS...")

# feature_importance = pd.DataFrame({
#     'feature': X.columns,
#     'coefficient': lr_model.coef_,
#     'abs_importance': abs(lr_model.coef_)
# }).sort_values('abs_importance', ascending=False)

# print("\nTOP 10 MOST IMPORTANT FEATURES:")
# print(feature_importance.head(10).to_string(index=False))

# # 8. Predictions
# print("\n8. MAKING PREDICTIONS...")

# # Simple prediction example
# print("Sample prediction for average house:")
# avg_features = X.mean().to_dict()
# sample_house = {col: avg_features.get(col, 0) for col in X.columns}

# house_df = pd.DataFrame([sample_house])
# house_scaled = scaler.transform(house_df)
# predicted_price = lr_model.predict(house_scaled)[0]

# print(f"Predicted Price for Average House: ${predicted_price:,.2f}")
# print(f"Actual Average Price: ${y.mean():,.2f}")

# # 9. Results Summary
# print("\n9. RESULTS SUMMARY")
# print("=" * 50)
# print(f"Model Performance: {r2*100:.1f}% accuracy")
# print(f"Average Error: ${mae:,.0f}")
# print(f"Top Feature: {feature_importance.iloc[0]['feature']}")
# print("=" * 50)
# print("LINEAR REGRESSION MODEL COMPLETED SUCCESSFULLY!")












import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Safe visualization - only import if needed and close properly
def create_safe_visualizations(df, y_test, y_pred, feature_importance):
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        print("\nCREATING SAFE VISUALIZATIONS...")
        
        # Figure 1: Basic plots
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Price distribution
        ax1.hist(df['median_house_value'], bins=30, edgecolor='black', alpha=0.7, color='lightblue')
        ax1.axvline(df['median_house_value'].mean(), color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('House Price ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of House Prices')
        ax1.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted
        ax2.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', linewidth=2)
        ax2.set_xlabel('Actual Prices ($)')
        ax2.set_ylabel('Predicted Prices ($)')
        ax2.set_title('Actual vs Predicted Prices')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = y_test - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.5, color='green')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Prices ($)')
        ax3.set_ylabel('Residuals ($)')
        ax3.set_title('Residual Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (top 8)
        top_features = feature_importance.head(8)
        ax4.barh(top_features['feature'], top_features['abs_importance'], color='orange')
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('Top 8 Most Important Features')
        ax4.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('model_analysis.png', dpi=100, bbox_inches='tight')
        plt.close(fig1)  # Important: close the figure to free memory
        
        # Figure 2: Correlation and relationships
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top correlated feature vs price
        correlations = df.corr()['median_house_value'].sort_values(ascending=False)
        top_feature = correlations.index[1]  # Skip the target itself
        ax1.scatter(df[top_feature], df['median_house_value'], alpha=0.3, color='purple')
        ax1.set_xlabel(top_feature)
        ax1.set_ylabel('House Price ($)')
        ax1.set_title(f'Price vs {top_feature}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Second top correlated feature
        if len(correlations) > 2:
            second_feature = correlations.index[2]
            ax2.scatter(df[second_feature], df['median_house_value'], alpha=0.3, color='brown')
            ax2.set_xlabel(second_feature)
            ax2.set_ylabel('House Price ($)')
            ax2.set_title(f'Price vs {second_feature}')
            ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax3.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Prediction Error ($)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Prediction Errors')
        ax3.grid(True, alpha=0.3)
        
        # 4. Price by ocean proximity (if available)
        if 'ocean_proximity' in df.columns:
            ocean_avg = df.groupby('ocean_proximity')['median_house_value'].mean()
            ocean_avg.plot(kind='bar', ax=ax4, color=['skyblue', 'lightcoral', 'lightgreen'])
            ax4.set_xlabel('Ocean Proximity')
            ax4.set_ylabel('Average Price ($)')
            ax4.set_title('Average Price by Ocean Proximity')
            ax4.tick_params(axis='x', rotation=45)
        else:
            # Alternative: Price by income quartile
            df['income_quartile'] = pd.qcut(df['median_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            income_avg = df.groupby('income_quartile')['median_house_value'].mean()
            income_avg.plot(kind='bar', ax=ax4, color='teal')
            ax4.set_xlabel('Income Quartile')
            ax4.set_ylabel('Average Price ($)')
            ax4.set_title('Average Price by Income Quartile')
        
        plt.tight_layout()
        plt.savefig('data_relationships.png', dpi=100, bbox_inches='tight')
        plt.close(fig2)  # Important: close the figure
        
        print("Visualizations saved as:")
        print("   - model_analysis.png")
        print("   - data_relationships.png")
        
    except Exception as e:
        print(f"Visualization skipped due to: {e}")
        print("Continuing with text output only...")

print("=== HOUSE PRICE PREDICTION WITH VISUALIZATIONS ===")
print("=" * 60)

# 1. Load Dataset
print("\n1. LOADING DATASET...")

df = None
try:
    df = pd.read_csv('housing.csv')
    print("SUCCESS: Dataset loaded from housing.csv")
except:
    try:
        df = pd.read_csv('housing_sample.csv')
        print("SUCCESS: Dataset loaded from housing_sample.csv")
    except:
        print("Creating sample data...")
        np.random.seed(42)
        sample_data = {
            'longitude': np.random.uniform(-124, -114, 1000),
            'latitude': np.random.uniform(32, 42, 1000),
            'housing_median_age': np.random.randint(1, 50, 1000),
            'total_rooms': np.random.randint(100, 10000, 1000),
            'total_bedrooms': np.random.randint(50, 5000, 1000),
            'population': np.random.randint(200, 5000, 1000),
            'households': np.random.randint(100, 3000, 1000),
            'median_income': np.random.uniform(1, 10, 1000),
            'median_house_value': np.random.randint(50000, 500000, 1000),
            'ocean_proximity': np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN'], 1000)
        }
        df = pd.DataFrame(sample_data)

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Data Preprocessing
print("\n2. DATA PREPROCESSING...")

# Handle missing values
if df.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Handle categorical data
if 'ocean_proximity' in df.columns:
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='location')

# Create new features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

print(f"Final dataset shape: {df.shape}")

# 3. Prepare Data
print("\n3. PREPARING DATA FOR MODELING...")
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

# 4. Train Model
print("\n4. TRAINING LINEAR REGRESSION MODEL...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print(f"   MAE: ${mae:,.2f}")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f} ({r2*100:.1f}% variance explained)")

# 5. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr_model.coef_,
    'abs_importance': abs(lr_model.coef_)
}).sort_values('abs_importance', ascending=False)

print(f"\nTOP 5 FEATURES:")
print(feature_importance.head(5).to_string(index=False))

# 6. Create Safe Visualizations
create_safe_visualizations(df, y_test, y_pred, feature_importance)

# 7. Final Results
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"Model Accuracy: {r2*100:.1f}%")
print(f"Average Error: ${mae:,.0f}")
print(f"Most Important Feature: {feature_importance.iloc[0]['feature']}")
print(f"Top 3 Price Drivers:")
for i in range(3):
    feat = feature_importance.iloc[i]
    direction = "increases" if feat['coefficient'] > 0 else "decreases"
    print(f"   {i+1}. {feat['feature']} ({direction} price)")

if os.path.exists('model_analysis.png'):
    print(f"\nVisualizations created:")
    print(f"   - model_analysis.png (Model performance)")
    print(f"   - data_relationships.png (Data insights)")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)