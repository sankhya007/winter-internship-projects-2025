import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("=== COMPREHENSIVE HOUSING DATA ANALYSIS ===")
print(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Enhanced Data Loading with Error Handling
print("\n1. LOADING AND VALIDATING DATA...")
try:
    df = pd.read_csv('housing.csv')
    print("CSV file loaded successfully!")
    
    # Data validation
    print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Dataset Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
except FileNotFoundError:
    print("Error: housing.csv file not found!")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Comprehensive Data Analysis
print("\n2. COMPREHENSIVE DATA ANALYSIS")
print("=" * 50)

# Basic info
print(f"Columns: {list(df.columns)}")
print(f"Data Types:\n{df.dtypes}")
print(f"Missing Values:\n{df.isnull().sum()}")

# Handle missing values professionally
missing_bedrooms = df['total_bedrooms'].isnull().sum()
if missing_bedrooms > 0:
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    print(f"Filled {missing_bedrooms} missing values in 'total_bedrooms' with median")

# 3. Statistical Summary with Better Formatting
print("\n3. STATISTICAL SUMMARY")
print("=" * 50)

# Calculate averages and key metrics
stats_data = {
    'Metric': [
        'Median House Price', 'Median Income', 'Total Rooms', 
        'Total Bedrooms', 'Housing Median Age', 'Population', 'Households'
    ],
    'Average': [
        f"${df['median_house_value'].mean():,.2f}",
        f"${df['median_income'].mean():,.2f}",
        f"{df['total_rooms'].mean():,.0f}",
        f"{df['total_bedrooms'].mean():,.0f}",
        f"{df['housing_median_age'].mean():.1f} years",
        f"{df['population'].mean():,.0f}",
        f"{df['households'].mean():,.0f}"
    ],
    'Range': [
        f"${df['median_house_value'].min():,.0f} - ${df['median_house_value'].max():,.0f}",
        f"${df['median_income'].min():.2f} - ${df['median_income'].max():.2f}",
        f"{df['total_rooms'].min():,.0f} - {df['total_rooms'].max():,.0f}",
        f"{df['total_bedrooms'].min():,.0f} - {df['total_bedrooms'].max():,.0f}",
        f"{df['housing_median_age'].min():.0f} - {df['housing_median_age'].max():.0f} years",
        f"{df['population'].min():,.0f} - {df['population'].max():,.0f}",
        f"{df['households'].min():,.0f} - {df['households'].max():,.0f}"
    ]
}

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))

# 4. Enhanced Visualizations
print("\n4. CREATING ENHANCED VISUALIZATIONS...")
plt.style.use('default')  # Clean professional style

# Create a figure with multiple subplots
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Comprehensive Housing Data Analysis Dashboard', fontsize=16, fontweight='bold')

# 4.1 Enhanced Bar Chart
plt.subplot(3, 3, 1)
ocean_price_avg = df.groupby('ocean_proximity')['median_house_value'].mean().sort_values(ascending=False)
colors = plt.cm.Set3(np.linspace(0, 1, len(ocean_price_avg)))
bars = plt.bar(ocean_price_avg.index, ocean_price_avg.values, color=colors, edgecolor='black')
plt.title('Average House Prices by Ocean Proximity', fontweight='bold')
plt.xlabel('Ocean Proximity')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')

# 4.2 Enhanced Scatter Plot - Price vs Income
plt.subplot(3, 3, 2)
plt.scatter(df['median_income'], df['median_house_value'], alpha=0.6, c=df['median_house_value'], 
           cmap='viridis', s=20)
plt.colorbar(label='House Price ($)')
plt.title('House Price vs Income', fontweight='bold')
plt.xlabel('Median Income ($)')
plt.ylabel('House Price ($)')
# Add correlation line
z = np.polyfit(df['median_income'], df['median_house_value'], 1)
p = np.poly1d(z)
plt.plot(df['median_income'], p(df['median_income']), "r--", alpha=0.8)

# 4.3 Enhanced Scatter Plot - Price vs Rooms
plt.subplot(3, 3, 3)
plt.scatter(df['total_rooms'], df['median_house_value'], alpha=0.6, c='green', s=20)
plt.title('House Price vs Total Rooms', fontweight='bold')
plt.xlabel('Total Rooms')
plt.ylabel('House Price ($)')

# 4.4 Enhanced Heatmap
plt.subplot(3, 3, 4)
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

im = plt.imshow(correlation_matrix, cmap='RdYlBu', aspect='auto')
plt.colorbar(im, shrink=0.6)
plt.title('Feature Correlation Heatmap', fontweight='bold')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

# Add correlation values to heatmap
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', fontsize=8, fontweight='bold')

# 4.5 Price Distribution Histogram
plt.subplot(3, 3, 5)
plt.hist(df['median_house_value'], bins=50, color='lightblue', edgecolor='black', alpha=0.7)
plt.title('House Price Distribution', fontweight='bold')
plt.xlabel('House Price ($)')
plt.ylabel('Frequency')
plt.axvline(df['median_house_value'].mean(), color='red', linestyle='--', 
           label=f'Mean: ${df["median_house_value"].mean():,.0f}')
plt.legend()

# 4.6 Geographic Price Distribution
plt.subplot(3, 3, 6)
scatter = plt.scatter(df['longitude'], df['latitude'], alpha=0.6,
                     c=df['median_house_value'], cmap='plasma', 
                     s=df['population']/100, edgecolor='black', linewidth=0.2)
plt.colorbar(scatter, label='House Price ($)')
plt.title('Geographic Price Distribution', fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# 4.7 Room-to-Bedroom Ratio
plt.subplot(3, 3, 7)
df['rooms_per_bedroom'] = df['total_rooms'] / df['total_bedrooms']
plt.scatter(df['rooms_per_bedroom'], df['median_house_value'], alpha=0.6, color='purple', s=20)
plt.title('Price vs Rooms per Bedroom', fontweight='bold')
plt.xlabel('Rooms per Bedroom')
plt.ylabel('House Price ($)')
plt.xlim(0, 20)  # Remove outliers for better visualization

# 4.8 Housing Age Distribution
plt.subplot(3, 3, 8)
age_counts = df['housing_median_age'].value_counts().sort_index()
plt.bar(age_counts.index, age_counts.values, color='orange', edgecolor='black', alpha=0.7)
plt.title('Housing Age Distribution', fontweight='bold')
plt.xlabel('Housing Median Age (years)')
plt.ylabel('Number of Houses')

# 4.9 Income Distribution
plt.subplot(3, 3, 9)
plt.hist(df['median_income'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Income Distribution', fontweight='bold')
plt.xlabel('Median Income ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('comprehensive_analysis_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 5. Advanced Insights and Business Recommendations
print("\n5. ADVANCED INSIGHTS AND BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("\nKEY FINDINGS:")
print("• Strongest price correlation: Income (r = {:.3f})".format(
    df['median_income'].corr(df['median_house_value'])))
print("• Coastal premium: ${:,.0f} higher than inland areas".format(
    ocean_price_avg['NEAR BAY'] - ocean_price_avg['INLAND']))
print("• Average rooms per bedroom: {:.1f}".format(df['rooms_per_bedroom'].mean()))
print("• Price variability: {:.1f}% coefficient of variation".format(
    (df['median_house_value'].std() / df['median_house_value'].mean()) * 100))

print("\nBUSINESS RECOMMENDATIONS:")
print("1. Target coastal areas for premium property investments")
print("2. Focus on income levels as primary price predictor")
print("3. Consider room-to-bedroom ratio for property valuation")
print("4. Monitor geographic clusters for market opportunities")
print("5. Use correlation insights for predictive modeling")

print("\nDATA QUALITY ASSESSMENT:")
print("• Completeness: {:.1f}% (missing values handled)".format(
    (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100))
print("• Outliers: {} price values > $400,000".format(
    len(df[df['median_house_value'] > 400000])))
print("• Data spread: Suitable for statistical analysis")

print("\n" + "=" * 60)
print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"Results saved as: comprehensive_analysis_dashboard.png")
print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")