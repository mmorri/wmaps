import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')


def load_products_data():
    """Load products from JSON file."""
    products = []
    with open('products.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    product = json.loads(line)
                    products.append(product)
                except json.JSONDecodeError:
                    continue
    return products


def prepare_features(products):
    """Prepare features for XGBoost model."""

    # Extract relevant data
    data = []
    book_categories = ["business", "history", "programming", "cooking"]

    for product in products:
        if 'salesrank' in product and 'category' in product:
            # Only include products with salesrank
            salesrank = product.get('salesrank', 0)
            if salesrank > 0:  # Filter out products with 0 or missing salesrank
                category = product.get('category', '')
                price = product.get('price', 0) / 100 if 'price' in product else 0

                # Create binary features for each book category
                features = {
                    'salesrank': salesrank,
                    'price': price,
                    'is_book': 1 if '/books' in category else 0,
                    'is_game': 1 if '/games' in category else 0,
                }

                # Add specific book category features
                for cat in book_categories:
                    features[f'is_{cat}'] = 1 if f'/books/{cat}' in category else 0

                # Additional features
                features['has_images'] = 1 if 'images' in product else 0
                features['description_length'] = len(str(product.get('description', '')))
                features['num_features'] = len(product.get('feature', []))
                features['title_length'] = len(product.get('title', ''))

                # Platform features for games
                if '/games' in category:
                    features['is_pc'] = 1 if '/pc' in category else 0
                    features['is_ps3'] = 1 if '/ps3' in category else 0
                    features['is_ps4'] = 1 if '/ps4' in category else 0
                    features['is_xbox'] = 1 if '/xbox' in category else 0
                    features['is_wiiu'] = 1 if '/wiiu' in category else 0
                else:
                    features['is_pc'] = 0
                    features['is_ps3'] = 0
                    features['is_ps4'] = 0
                    features['is_xbox'] = 0
                    features['is_wiiu'] = 0

                data.append(features)

    return pd.DataFrame(data)


def train_xgboost_model(df):
    """Train XGBoost model to predict sales rank."""

    # Separate features and target
    X = df.drop('salesrank', axis=1)
    y = df['salesrank']

    # Log transform the target (salesrank) to handle skewness
    y_log = np.log1p(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # Create and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Make predictions
    y_pred_log = model.predict(X_test)

    # Convert back from log scale
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y_log, cv=5,
                               scoring='neg_mean_absolute_error')

    return model, X_train, X_test, y_test_original, y_pred, mae, rmse, r2, cv_scores


def analyze_book_categories(df, model):
    """Analyze the impact of book categories on sales rank."""

    book_categories = ["business", "history", "programming", "cooking"]

    # Calculate average sales rank for each category
    category_stats = []
    for cat in book_categories:
        cat_products = df[df[f'is_{cat}'] == 1]
        if len(cat_products) > 0:
            avg_rank = cat_products['salesrank'].mean()
            median_rank = cat_products['salesrank'].median()
            count = len(cat_products)
            category_stats.append({
                'Category': cat.capitalize(),
                'Count': count,
                'Avg Sales Rank': avg_rank,
                'Median Sales Rank': median_rank,
                'Lower is Better': '✓' if avg_rank < df['salesrank'].mean() else '✗'
            })

    stats_df = pd.DataFrame(category_stats)

    # Feature importance for book categories
    feature_importance = model.feature_importances_
    feature_names = model.feature_names_in_

    book_features = [f'is_{cat}' for cat in book_categories]
    book_importance = {}
    for feat in book_features:
        if feat in feature_names:
            idx = list(feature_names).index(feat)
            book_importance[feat.replace('is_', '')] = feature_importance[idx]

    return stats_df, book_importance


def create_visualizations(df, model, X_test, y_test, y_pred, mae, rmse, r2):
    """Create comprehensive visualizations."""

    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 12))

    # 1. Feature Importance
    ax1 = plt.subplot(2, 3, 1)
    plot_importance(model, ax=ax1, max_num_features=10, importance_type='weight')
    ax1.set_title('Feature Importance (Top 10)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('F Score')

    # 2. Actual vs Predicted
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Sales Rank')
    ax2.set_ylabel('Predicted Sales Rank')
    ax2.set_title('Actual vs Predicted Sales Rank', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Sales Rank by Book Category
    ax3 = plt.subplot(2, 3, 3)
    book_categories = ["business", "history", "programming", "cooking"]
    book_data = []
    for cat in book_categories:
        cat_ranks = df[df[f'is_{cat}'] == 1]['salesrank'].values
        if len(cat_ranks) > 0:
            book_data.append(cat_ranks)

    if book_data:
        bp = ax3.boxplot(book_data, tick_labels=[c.capitalize() for c in book_categories[:len(book_data)]],
                         patch_artist=True)
        colors = sns.color_palette('husl', len(book_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_ylabel('Sales Rank (lower is better)')
        ax3.set_title('Sales Rank Distribution by Book Category', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticklabels([c.capitalize() for c in book_categories[:len(book_data)]], rotation=45)

    # 4. Residual Plot
    ax4 = plt.subplot(2, 3, 4)
    residuals = y_test - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Predicted Sales Rank')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Category Count and Average Sales Rank
    ax5 = plt.subplot(2, 3, 5)
    category_counts = []
    category_avg_ranks = []
    category_names = []

    for cat in book_categories:
        cat_df = df[df[f'is_{cat}'] == 1]
        if len(cat_df) > 0:
            category_counts.append(len(cat_df))
            category_avg_ranks.append(cat_df['salesrank'].mean())
            category_names.append(cat.capitalize())

    if category_counts:
        x = np.arange(len(category_names))
        width = 0.35

        ax5_twin = ax5.twinx()

        bars1 = ax5.bar(x - width/2, category_counts, width, label='Count', color='skyblue')
        bars2 = ax5_twin.bar(x + width/2, category_avg_ranks, width, label='Avg Rank', color='coral')

        ax5.set_xlabel('Book Category')
        ax5.set_ylabel('Product Count', color='skyblue')
        ax5_twin.set_ylabel('Average Sales Rank', color='coral')
        ax5.set_title('Product Count vs Average Sales Rank', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(category_names)
        ax5.tick_params(axis='y', labelcolor='skyblue')
        ax5_twin.tick_params(axis='y', labelcolor='coral')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            ax5_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 6. Feature Correlation Heatmap
    ax6 = plt.subplot(2, 3, 6)

    # Select book-related features for correlation
    book_features = ['is_business', 'is_history', 'is_programming', 'is_cooking',
                    'price', 'salesrank']
    corr_df = df[book_features].corr()

    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, ax=ax6, cbar_kws={"shrink": 0.8})
    ax6.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('XGBoost Sales Rank Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('salesrank_prediction_analysis.png', dpi=100, bbox_inches='tight')
    print("\n✅ Visualization saved as 'salesrank_prediction_analysis.png'")

    return fig


def main():
    print("=" * 60)
    print("XGBoost Sales Rank Prediction")
    print("=" * 60)

    # Load data
    print("\n1. Loading products data...")
    products = load_products_data()
    print(f"   ✅ Loaded {len(products)} products")

    # Prepare features
    print("\n2. Preparing features...")
    df = prepare_features(products)
    print(f"   ✅ Created feature matrix: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"   📊 Products with sales rank: {len(df)}")

    # Show feature columns
    feature_cols = [col for col in df.columns if col != 'salesrank']
    print(f"\n   Features used for prediction:")
    for i, col in enumerate(feature_cols, 1):
        if i <= 5:
            print(f"   • {col}")
    print(f"   ... and {len(feature_cols) - 5} more features")

    # Train model
    print("\n3. Training XGBoost model...")
    model, X_train, X_test, y_test, y_pred, mae, rmse, r2, cv_scores = train_xgboost_model(df)

    print(f"   ✅ Model trained successfully")
    print(f"\n   📈 Model Performance:")
    print(f"   • MAE:  {mae:.2f}")
    print(f"   • RMSE: {rmse:.2f}")
    print(f"   • R²:   {r2:.3f}")
    print(f"   • Cross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    # Analyze book categories
    print("\n4. Analyzing book categories impact...")
    stats_df, book_importance = analyze_book_categories(df, model)

    print("\n   📚 Book Category Statistics:")
    print("   " + "-" * 70)
    for _, row in stats_df.iterrows():
        print(f"   {row['Category']:12} | Count: {row['Count']:3} | "
              f"Avg Rank: {row['Avg Sales Rank']:8.1f} | "
              f"Median: {row['Median Sales Rank']:8.1f} | "
              f"Better than avg: {row['Lower is Better']}")

    print("\n   🎯 Feature Importance for Book Categories:")
    for cat, importance in sorted(book_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cat:12}: {importance:.4f}")

    # Predictions for specific book categories
    print("\n5. Sample predictions for book categories...")

    # Create sample data for each category
    sample_predictions = []
    for cat in ["business", "history", "programming", "cooking"]:
        sample = pd.DataFrame({
            'price': [25.0],  # $25 book
            'is_book': [1],
            'is_game': [0],
            'is_business': [1 if cat == 'business' else 0],
            'is_history': [1 if cat == 'history' else 0],
            'is_programming': [1 if cat == 'programming' else 0],
            'is_cooking': [1 if cat == 'cooking' else 0],
            'has_images': [1],
            'description_length': [500],
            'num_features': [3],
            'title_length': [30],
            'is_pc': [0],
            'is_ps3': [0],
            'is_ps4': [0],
            'is_xbox': [0],
            'is_wiiu': [0]
        })

        pred_log = model.predict(sample)[0]
        pred_rank = np.expm1(pred_log)
        sample_predictions.append((cat.capitalize(), pred_rank))

    print("\n   🔮 Predicted sales ranks for $25 books:")
    for cat, rank in sorted(sample_predictions, key=lambda x: x[1]):
        print(f"   • {cat:12}: {rank:8.0f} {'⭐ (Best)' if rank == min(p[1] for p in sample_predictions) else ''}")

    # Create visualizations
    print("\n6. Creating visualizations...")
    create_visualizations(df, model, X_test, y_test, y_pred, mae, rmse, r2)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    # Final insights
    print("\n📊 Key Insights:")
    best_category = min(sample_predictions, key=lambda x: x[1])
    worst_category = max(sample_predictions, key=lambda x: x[1])

    print(f"   • Best performing category: {best_category[0]} (predicted rank: {best_category[1]:.0f})")
    print(f"   • Worst performing category: {worst_category[0]} (predicted rank: {worst_category[1]:.0f})")
    print(f"   • Model R² score: {r2:.3f} - {'Good' if r2 > 0.7 else 'Moderate' if r2 > 0.5 else 'Needs improvement'} predictive power")

    return model, df


if __name__ == "__main__":
    model, df = main()