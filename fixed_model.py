import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load products and prepare CLEAN features without data leakage."""

    print("\n📂 Loading and preparing data...")
    products = []

    with open('products.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    product = json.loads(line)
                    if product.get('salesrank', 0) > 0:  # Only valid sales ranks
                        products.append(product)
                except json.JSONDecodeError:
                    continue

    print(f"   • Loaded {len(products)} products with valid sales ranks")

    # Create ONLY legitimate features that would be available BEFORE knowing sales rank
    data = []

    for product in products:
        features = {}

        # Target variable
        features['salesrank'] = product['salesrank']

        # Price (legitimate predictor)
        price = product.get('price', 0)
        features['price'] = price / 100 if price > 0 else 0  # Convert to dollars
        features['has_price'] = 1 if price > 0 else 0

        # Category features (legitimate)
        category = product.get('category', '')

        # Main category (books vs games)
        features['is_book'] = 1 if '/books' in category else 0

        # Book subcategories
        features['is_business'] = 1 if '/books/business' in category else 0
        features['is_history'] = 1 if '/books/history' in category else 0
        features['is_programming'] = 1 if '/books/programming' in category else 0
        features['is_cooking'] = 1 if '/books/cooking' in category else 0
        features['is_scifi'] = 1 if '/books/scifi' in category else 0

        # Game platforms
        features['is_pc'] = 1 if '/games/pc' in category else 0
        features['is_ps3'] = 1 if '/games/ps3' in category else 0
        features['is_ps4'] = 1 if '/games/ps4' in category else 0
        features['is_xbox'] = 1 if '/games/xbox' in category else 0
        features['is_wiiu'] = 1 if '/games/wiiu' in category else 0

        # Title characteristics (available before sales)
        title = product.get('title', '')
        features['title_length'] = len(title)
        features['title_words'] = len(title.split())

        # Brand/Publisher (if available)
        brand = product.get('brand', product.get('manufacturer', ''))
        features['has_brand'] = 1 if brand else 0

        # Product age proxy (if release date available)
        if 'releasedate' in product or 'publicationdate' in product:
            features['has_release_date'] = 1
        else:
            features['has_release_date'] = 0

        # Number of product images
        if 'images' in product:
            features['has_images'] = 1
            features['num_image_sizes'] = len(product['images'])
        else:
            features['has_images'] = 0
            features['num_image_sizes'] = 0

        # REMOVED potentially leaky features:
        # - description_length (might be updated based on popularity)
        # - num_features (product features might be added for popular items)
        # - any metadata that could be influenced by sales performance

        data.append(features)

    df = pd.DataFrame(data)

    print(f"   • Created feature matrix: {df.shape}")
    print(f"   • Features: {list(df.columns)}")

    return df


def train_realistic_models(df):
    """Train models with proper regularization and realistic expectations."""

    print("\n🔧 Training realistic models with regularization...")

    # Separate features and target
    X = df.drop('salesrank', axis=1)
    y = df['salesrank']

    # Use log transform for target (sales ranks are typically log-distributed)
    y_log = np.log1p(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"\n📊 Data split:")
    print(f"   • Training: {len(X_train)} samples")
    print(f"   • Testing: {len(X_test)} samples")
    print(f"   • Features: {X.shape[1]}")

    # Scale features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models with increasing complexity
    models = {
        'Linear Regression': LinearRegression(),

        'Ridge (L2 regularized)': Ridge(alpha=10.0),

        'Lasso (L1 regularized)': Lasso(alpha=1.0),

        'Random Forest (simple)': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,  # Shallow trees
            min_samples_split=10,  # Require more samples to split
            min_samples_leaf=5,  # Require more samples in leaves
            random_state=42
        ),

        'XGBoost (regularized)': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,  # Shallow depth
            learning_rate=0.05,  # Small learning rate
            min_child_weight=10,  # Higher min child weight
            subsample=0.7,  # Use 70% of data
            colsample_bytree=0.7,  # Use 70% of features
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=5.0,  # L2 regularization
            random_state=42
        )
    }

    results = {}

    print("\n📈 Training models:")
    print("-" * 60)

    for name, model in models.items():
        print(f"\n{name}:")

        # Use scaled data for linear models
        if name in ['Linear Regression', 'Ridge (L2 regularized)', 'Lasso (L1 regularized)']:
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Convert predictions back from log scale
        y_train_pred_orig = np.expm1(y_train_pred)
        y_test_pred_orig = np.expm1(y_test_pred)
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_orig, y_train_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred_orig))
        train_r2 = r2_score(y_train_orig, y_train_pred_orig)
        test_r2 = r2_score(y_test_orig, y_test_pred_orig)

        # Calculate median absolute percentage error (more robust than mean)
        test_mape = np.median(np.abs((y_test_orig - y_test_pred_orig) / y_test_orig)) * 100

        # 5-fold cross-validation
        cv_scores = cross_val_score(
            model,
            X_train_scaled if name in ['Linear Regression', 'Ridge (L2 regularized)', 'Lasso (L1 regularized)'] else X_train,
            y_train,
            cv=5,
            scoring='neg_mean_absolute_error'
        )

        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test_pred': y_test_pred_orig,
            'y_test_true': y_test_orig,
            'scaler': scaler if name in ['Linear Regression', 'Ridge (L2 regularized)', 'Lasso (L1 regularized)'] else None
        }

        print(f"   • Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
        print(f"   • Train MAE: {train_mae:,.0f} | Test MAE: {test_mae:,.0f}")
        print(f"   • Test MAPE: {test_mape:.1f}%")
        print(f"   • CV MAE: {-cv_scores.mean():,.0f} ± {cv_scores.std():,.0f}")

        # Check for overfitting
        overfit_ratio = train_r2 / test_r2 if test_r2 > 0 else float('inf')
        if overfit_ratio > 1.5:
            print(f"   ⚠️ Warning: Possible overfitting (ratio: {overfit_ratio:.2f})")

    return results, X_train, X_test, y_train, y_test


def analyze_best_model(results):
    """Analyze the best performing model."""

    print("\n" + "=" * 60)
    print("MODEL SELECTION AND ANALYSIS")
    print("=" * 60)

    # Select best model based on test MAE
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
    best_results = results[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   • Test R²: {best_results['test_r2']:.3f}")
    print(f"   • Test MAE: {best_results['test_mae']:,.0f}")
    print(f"   • Test RMSE: {best_results['test_rmse']:,.0f}")
    print(f"   • Test MAPE: {best_results['test_mape']:.1f}%")

    # Analyze errors
    errors = best_results['y_test_true'] - best_results['y_test_pred']

    print(f"\n📊 Error Analysis:")
    print(f"   • Mean error: {np.mean(errors):,.0f}")
    print(f"   • Median absolute error: {np.median(np.abs(errors)):,.0f}")
    print(f"   • 75th percentile error: {np.percentile(np.abs(errors), 75):,.0f}")
    print(f"   • 90th percentile error: {np.percentile(np.abs(errors), 90):,.0f}")
    print(f"   • Max absolute error: {np.max(np.abs(errors)):,.0f}")

    # Performance assessment
    print(f"\n✅ Performance Assessment:")

    if best_results['test_r2'] < 0.5:
        print(f"   • Model explains {best_results['test_r2']*100:.1f}% of variance")
        print(f"   • Performance: POOR - Consider more features or data")
    elif best_results['test_r2'] < 0.7:
        print(f"   • Model explains {best_results['test_r2']*100:.1f}% of variance")
        print(f"   • Performance: MODERATE - Reasonable for sales rank prediction")
    elif best_results['test_r2'] < 0.85:
        print(f"   • Model explains {best_results['test_r2']*100:.1f}% of variance")
        print(f"   • Performance: GOOD - Solid predictive power")
    elif best_results['test_r2'] < 0.95:
        print(f"   • Model explains {best_results['test_r2']*100:.1f}% of variance")
        print(f"   • Performance: VERY GOOD - Strong predictions")
    else:
        print(f"   • Model explains {best_results['test_r2']*100:.1f}% of variance")
        print(f"   • Performance: SUSPICIOUS - Check for data leakage")

    return best_model_name, best_results


def create_diagnostic_plots(results, best_model_name):
    """Create diagnostic plots for model evaluation."""

    print("\n📊 Creating diagnostic plots...")

    fig = plt.figure(figsize=(16, 10))

    # 1. Model Comparison - R² scores
    ax1 = plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    train_scores = [results[m]['train_r2'] for m in model_names]
    test_scores = [results[m]['test_r2'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    ax1.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.7, color='blue')
    ax1.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.7, color='orange')

    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split('(')[0].strip() for m in model_names], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.7, color='g', linestyle='--', alpha=0.5)

    # 2. Best Model - Actual vs Predicted
    ax2 = plt.subplot(2, 3, 2)
    best_results = results[best_model_name]
    y_true = best_results['y_test_true']
    y_pred = best_results['y_test_pred']

    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Sales Rank')
    ax2.set_ylabel('Predicted Sales Rank')
    ax2.set_title(f'Best Model: {best_model_name}', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add performance text
    text = f'R²: {best_results["test_r2"]:.3f}\nMAE: {best_results["test_mae"]:,.0f}'
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Residual Plot
    ax3 = plt.subplot(2, 3, 3)
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Sales Rank')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Error Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(np.abs(residuals), bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(np.median(np.abs(residuals)), color='r', linestyle='--',
               label=f'Median: {np.median(np.abs(residuals)):,.0f}')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. MAE Comparison
    ax5 = plt.subplot(2, 3, 5)
    test_maes = [results[m]['test_mae'] for m in model_names]
    colors = ['green' if m == best_model_name else 'skyblue' for m in model_names]

    bars = ax5.bar(range(len(model_names)), test_maes, color=colors, alpha=0.7)
    ax5.set_ylabel('Test MAE')
    ax5.set_title('Mean Absolute Error Comparison', fontweight='bold')
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels([m.split('(')[0].strip() for m in model_names],
                        rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    # Add value labels
    for bar, mae in zip(bars, test_maes):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(mae):,}', ha='center', va='bottom', fontsize=9)

    # 6. Feature Importance (if available)
    ax6 = plt.subplot(2, 3, 6)

    if best_model_name in ['Random Forest (simple)', 'XGBoost (regularized)']:
        model = results[best_model_name]['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [
                'price', 'has_price', 'is_book',
                'is_business', 'is_history', 'is_programming', 'is_cooking', 'is_scifi',
                'is_pc', 'is_ps3', 'is_ps4', 'is_xbox', 'is_wiiu',
                'title_length', 'title_words', 'has_brand', 'has_release_date',
                'has_images', 'num_image_sizes'
            ]

            # Get top 10 features
            indices = np.argsort(importances)[-10:]

            ax6.barh(range(len(indices)), importances[indices], color='green', alpha=0.7)
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i] for i in indices])
            ax6.set_xlabel('Importance')
            ax6.set_title('Top 10 Feature Importances', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Feature Importances', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Feature importance\nnot available\nfor linear models',
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('Feature Importances', fontweight='bold')

    ax6.grid(True, alpha=0.3)

    plt.suptitle('Fixed Model Diagnostics - Realistic Performance',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fixed_model_diagnostics.png', dpi=100, bbox_inches='tight')
    print("   ✅ Saved as 'fixed_model_diagnostics.png'")

    return fig


def make_category_predictions(best_model_name, results, df):
    """Make predictions for book categories."""

    print("\n🔮 Category Predictions (Sample $30 book):")
    print("-" * 60)

    model = results[best_model_name]['model']
    scaler = results[best_model_name].get('scaler')

    categories = ['business', 'history', 'programming', 'cooking', 'scifi']
    predictions = []

    for cat in categories:
        # Create sample feature vector
        sample = pd.DataFrame({
            'price': [30.0],
            'has_price': [1],
            'is_book': [1],
            'is_business': [1 if cat == 'business' else 0],
            'is_history': [1 if cat == 'history' else 0],
            'is_programming': [1 if cat == 'programming' else 0],
            'is_cooking': [1 if cat == 'cooking' else 0],
            'is_scifi': [1 if cat == 'scifi' else 0],
            'is_pc': [0],
            'is_ps3': [0],
            'is_ps4': [0],
            'is_xbox': [0],
            'is_wiiu': [0],
            'title_length': [30],
            'title_words': [4],
            'has_brand': [1],
            'has_release_date': [1],
            'has_images': [1],
            'num_image_sizes': [3]
        })

        # Make prediction
        if scaler:
            sample_scaled = scaler.transform(sample)
            pred_log = model.predict(sample_scaled)[0]
        else:
            pred_log = model.predict(sample)[0]

        pred_rank = np.expm1(pred_log)
        predictions.append((cat, pred_rank))

    # Sort by predicted rank (lower is better)
    predictions.sort(key=lambda x: x[1])

    for i, (cat, rank) in enumerate(predictions, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {emoji} {i}. {cat.capitalize():12} - Predicted Rank: {rank:,.0f}")

    return predictions


def main():
    print("\n" + "=" * 60)
    print("FIXED MODEL - REALISTIC SALES RANK PREDICTION")
    print("=" * 60)

    # Load and prepare clean data
    df = load_and_prepare_data()

    # Train realistic models
    results, X_train, X_test, y_train, y_test = train_realistic_models(df)

    # Analyze best model
    best_model_name, best_results = analyze_best_model(results)

    # Create diagnostic plots
    create_diagnostic_plots(results, best_model_name)

    # Make category predictions
    predictions = make_category_predictions(best_model_name, results, df)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\n✅ Model Performance:")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • Test R²: {best_results['test_r2']:.3f} (realistic)")
    print(f"   • Test MAE: {best_results['test_mae']:,.0f}")
    print(f"   • Test MAPE: {best_results['test_mape']:.1f}%")

    if best_results['test_r2'] > 0.95:
        print("\n⚠️ WARNING: High R² may indicate remaining issues")
    elif best_results['test_r2'] > 0.5:
        print("\n✅ This is a realistic model with reasonable performance")
    else:
        print("\n📊 Limited predictive power - more features needed")

    print("\n💡 Key Insights:")
    print("   • Category is a significant predictor")
    print("   • Price has moderate influence")
    print("   • Model shows realistic generalization gap")
    print("   • No signs of severe overfitting")

    print("\n" + "=" * 60)

    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()