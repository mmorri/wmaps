import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Training and saving models...")

# Load products - handle NDJSON format
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

# Filter for products with salesrank
df_with_rank = df[df['salesrank'].notna()].copy()

# Determine product type
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

# Filter for books and videogames only
df_filtered = df_with_rank[df_with_rank['product_type'].isin(['book', 'videogame'])].copy()

# Calculate average sales rank by subcategory for books
def get_subcategory(category):
    if pd.isna(category) or not isinstance(category, str):
        return 'unknown'
    if '/' in category:
        parts = category.split('/')
        if len(parts) > 1:
            return parts[-1]
    return category if category else 'unknown'

df_filtered['subcategory'] = df_filtered['category'].apply(get_subcategory)

# Calculate average sales rank for books by subcategory
books_df = df_filtered[df_filtered['product_type'] == 'book'].copy()
avg_salesrank_by_category = books_df.groupby('subcategory')['salesrank'].agg(['mean', 'median', 'count']).to_dict()

# Feature engineering
df_filtered['price'] = pd.to_numeric(df_filtered['price'], errors='coerce')
df_filtered['has_price'] = df_filtered['price'].notna().astype(int)
price_median = df_filtered['price'].median()
if pd.isna(price_median):
    price_median = 0
df_filtered['price'] = df_filtered['price'].fillna(price_median)

def safe_list_len(x):
    if isinstance(x, list):
        return len(x)
    return 0

df_filtered['num_also_bought'] = df_filtered.get('also_bought', pd.Series()).apply(safe_list_len)
df_filtered['num_also_viewed'] = df_filtered.get('also_viewed', pd.Series()).apply(safe_list_len)
df_filtered['num_bought_together'] = df_filtered.get('bought_together', pd.Series()).apply(safe_list_len)
df_filtered['title_length'] = df_filtered['title'].fillna('').apply(len)
df_filtered['has_brand'] = df_filtered.get('brand', pd.Series()).notna().astype(int)

# Encode categorical variables
le_product = LabelEncoder()
le_subcategory = LabelEncoder()

df_filtered['product_type_encoded'] = le_product.fit_transform(df_filtered['product_type'])
df_filtered['subcategory_encoded'] = le_subcategory.fit_transform(df_filtered['subcategory'])

# Features for modeling
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

for col in feature_columns:
    if col not in df_filtered.columns:
        df_filtered[col] = 0

X = df_filtered[feature_columns].fillna(0)
y = df_filtered['salesrank']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Train XGBoost model
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Calculate median values for features (for default predictions)
books_only = df_filtered[df_filtered['product_type'] == 'book']
feature_medians = {}
for col in feature_columns:
    if col not in ['product_type_encoded', 'subcategory_encoded']:
        feature_medians[col] = books_only[col].median() if len(books_only) > 0 else 0

# Save models and encoders
print("Saving models and metadata...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_product': le_product,
        'le_subcategory': le_subcategory,
        'feature_columns': feature_columns,
        'avg_salesrank_by_category': avg_salesrank_by_category,
        'feature_medians': feature_medians,
        'price_median': price_median
    }, f)

print("Models and metadata saved successfully!")
print(f"Available book categories: {list(le_subcategory.classes_)}")