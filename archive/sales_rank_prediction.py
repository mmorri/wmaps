import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
with open('categories.json', 'r') as f:
    categories = json.load(f)

# Load products - handle NDJSON format (one JSON object per line)
products = []
with open('products.json', 'r') as f:
    for line in f:
        try:
            product = json.loads(line.strip())
            products.append(product)
        except json.JSONDecodeError:
            continue

print(f"Total products loaded: {len(products)}")

# Create DataFrame
df = pd.DataFrame(products)

# Display basic info
print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few products sample:")
available_cols = ['title', 'category', 'salesrank']
display_cols = [col for col in available_cols if col in df.columns]
if display_cols:
    print(df[display_cols].head())

# Filter for products with salesrank (lowercase in the data)
df_with_rank = df[df['salesrank'].notna()].copy()
print(f"\nProducts with sales rank: {len(df_with_rank)}")

# Determine product type based on category
def get_product_type(category):
    if pd.isna(category) or not isinstance(category, str):
        return 'unknown'
    category_lower = category.lower()
    if 'books' in category_lower:
        return 'book'
    elif 'games' in category_lower or 'video games' in category_lower:
        return 'videogame'
    else:
        return 'other'

df_with_rank['product_type'] = df_with_rank['category'].apply(get_product_type)

# Check distribution
print(f"\nProduct type distribution:")
print(df_with_rank['product_type'].value_counts())

# Filter for books and videogames only
df_filtered = df_with_rank[df_with_rank['product_type'].isin(['book', 'videogame'])].copy()
print(f"\nFiltered dataset (books and videogames only): {len(df_filtered)}")
print(f"Books: {len(df_filtered[df_filtered['product_type'] == 'book'])}")
print(f"Videogames: {len(df_filtered[df_filtered['product_type'] == 'videogame'])}")

if len(df_filtered) == 0:
    print("\nNo books or videogames found with sales rank. Checking category field structure...")
    print("\nSample categories from products with sales rank:")
    sample_categories = df_with_rank['category'].dropna().head(10)
    for i, cat in enumerate(sample_categories):
        print(f"  {i+1}: {cat}")

    # Try alternative category detection
    def get_product_type_v2(row):
        # Check multiple fields for category info
        category = str(row.get('category', '')).lower()
        title = str(row.get('title', '')).lower()

        # Check for books
        if 'book' in category or 'book' in title:
            return 'book'
        # Check for games
        elif 'game' in category or 'game' in title or 'xbox' in category or 'playstation' in category or 'nintendo' in category:
            return 'videogame'
        else:
            return 'other'

    df_with_rank['product_type'] = df_with_rank.apply(get_product_type_v2, axis=1)
    print(f"\nUpdated product type distribution:")
    print(df_with_rank['product_type'].value_counts())

    df_filtered = df_with_rank[df_with_rank['product_type'].isin(['book', 'videogame'])].copy()
    print(f"\nFiltered dataset after alternative detection: {len(df_filtered)}")

if len(df_filtered) < 10:
    print("\nInsufficient data for modeling. Need at least 10 samples.")
    exit()

# Feature engineering
print("\nEngineering features...")

# Extract subcategory if available
def get_subcategory(category):
    if pd.isna(category) or not isinstance(category, str):
        return 'unknown'
    # Try to extract subcategory
    if '/' in category:
        parts = category.split('/')
        if len(parts) > 1:
            return parts[-1]
    return category if category else 'unknown'

df_filtered['subcategory'] = df_filtered['category'].apply(get_subcategory)

# Extract price (handle missing values)
df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce')
df_filtered['has_price'] = df_filtered['price'].notna().astype(int)
price_median = df_filtered['price'].median()
if pd.isna(price_median):
    price_median = 0
df_filtered['price'] = df_filtered['price'].fillna(price_median)

# Count related products (handle missing or non-list fields)
def safe_list_len(x):
    if isinstance(x, list):
        return len(x)
    return 0

df_filtered['num_also_bought'] = df_filtered.get('also_bought', pd.Series()).apply(safe_list_len)
df_filtered['num_also_viewed'] = df_filtered.get('also_viewed', pd.Series()).apply(safe_list_len)
df_filtered['num_bought_together'] = df_filtered.get('bought_together', pd.Series()).apply(safe_list_len)

# Title length
df_filtered['title_length'] = df_filtered['title'].fillna('').apply(len)

# Brand feature (1 if has brand, 0 otherwise)
df_filtered['has_brand'] = df_filtered.get('brand', pd.Series()).notna().astype(int)

# Encode categorical variables
le_product = LabelEncoder()
le_subcategory = LabelEncoder()

df_filtered['product_type_encoded'] = le_product.fit_transform(df_filtered['product_type'])
df_filtered['subcategory_encoded'] = le_subcategory.fit_transform(df_filtered['subcategory'])

# Select features for modeling
feature_columns = [
    'product_type_encoded',
    'subcategory_encoded',
    'price',
    'has_price',
    'num_also_bought',
    'num_also_viewed',
    'num_bought_together',
    'title_length',
    'has_brand'
]

# Ensure all feature columns exist
for col in feature_columns:
    if col not in df_filtered.columns:
        df_filtered[col] = 0

X = df_filtered[feature_columns].fillna(0)
y = df_filtered['salesrank']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Sales rank range: {y.min():.0f} - {y.max():.0f}")
print(f"Sales rank mean: {y.mean():.2f}")
print(f"Sales rank median: {y.median():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train RandomForest model
print("\n" + "="*50)
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# RandomForest predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# RandomForest metrics
print("\nRandom Forest Results:")
print("-" * 30)
print("Train Set:")
print(f"  MAE: {mean_absolute_error(y_train, rf_train_pred):,.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, rf_train_pred)):,.2f}")
print(f"  R2 Score: {r2_score(y_train, rf_train_pred):.4f}")

print("\nTest Set:")
print(f"  MAE: {mean_absolute_error(y_test, rf_test_pred):,.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, rf_test_pred)):,.2f}")
print(f"  R2 Score: {r2_score(y_test, rf_test_pred):.4f}")

# Feature importance for RandomForest
rf_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest Feature Importance:")
for idx, row in rf_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Train XGBoost model
print("\n" + "="*50)
print("Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# XGBoost predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# XGBoost metrics
print("\nXGBoost Results:")
print("-" * 30)
print("Train Set:")
print(f"  MAE: {mean_absolute_error(y_train, xgb_train_pred):,.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, xgb_train_pred)):,.2f}")
print(f"  R2 Score: {r2_score(y_train, xgb_train_pred):.4f}")

print("\nTest Set:")
print(f"  MAE: {mean_absolute_error(y_test, xgb_test_pred):,.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, xgb_test_pred)):,.2f}")
print(f"  R2 Score: {r2_score(y_test, xgb_test_pred):.4f}")

# Feature importance for XGBoost
xgb_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nXGBoost Feature Importance:")
for idx, row in xgb_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Model comparison
print("\n" + "="*50)
print("MODEL COMPARISON (Test Set Performance)")
print("-" * 50)
print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15} {'R2 Score':<10}")
print("-" * 50)
print(f"{'Random Forest':<15} {mean_absolute_error(y_test, rf_test_pred):>14,.2f} {np.sqrt(mean_squared_error(y_test, rf_test_pred)):>14,.2f} {r2_score(y_test, rf_test_pred):>9.4f}")
print(f"{'XGBoost':<15} {mean_absolute_error(y_test, xgb_test_pred):>14,.2f} {np.sqrt(mean_squared_error(y_test, xgb_test_pred)):>14,.2f} {r2_score(y_test, xgb_test_pred):>9.4f}")

# Analyze predictions by product type
print("\n" + "="*50)
print("PERFORMANCE BY PRODUCT TYPE (Test Set)")
print("-" * 50)

X_test_with_type = X_test.copy()
X_test_with_type['actual'] = y_test.values
X_test_with_type['rf_pred'] = rf_test_pred
X_test_with_type['xgb_pred'] = xgb_test_pred
X_test_with_type['product_type'] = le_product.inverse_transform(X_test_with_type['product_type_encoded'])

for product_type in ['book', 'videogame']:
    subset = X_test_with_type[X_test_with_type['product_type'] == product_type]
    if len(subset) > 0:
        print(f"\n{product_type.upper()}S (n={len(subset)}):")
        print(f"  Random Forest MAE: {mean_absolute_error(subset['actual'], subset['rf_pred']):,.2f}")
        print(f"  XGBoost MAE: {mean_absolute_error(subset['actual'], subset['xgb_pred']):,.2f}")
        print(f"  Average Actual Sales Rank: {subset['actual'].mean():,.2f}")
        print(f"  Sales Rank Std Dev: {subset['actual'].std():,.2f}")

print("\n" + "="*50)
print("Analysis complete!")