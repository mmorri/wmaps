import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
from xgboost import plot_importance
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load products and prepare features."""
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

    # Prepare features
    data = []
    book_categories = ["business", "history", "programming", "cooking"]

    for product in products:
        if 'salesrank' in product and 'category' in product:
            salesrank = product.get('salesrank', 0)
            if salesrank > 0:
                category = product.get('category', '')
                price = product.get('price', 0) / 100 if 'price' in product else 0

                features = {
                    'salesrank': salesrank,
                    'price': price,
                    'is_book': 1 if '/books' in category else 0,
                    'is_game': 1 if '/games' in category else 0,
                }

                for cat in book_categories:
                    features[f'is_{cat}'] = 1 if f'/books/{cat}' in category else 0

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


def comprehensive_model_evaluation(df):
    """Perform comprehensive model evaluation and diagnostics."""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL DIAGNOSTICS")
    print("=" * 70)

    # Separate features and target
    X = df.drop('salesrank', axis=1)
    y = df['salesrank']

    # Log transform the target
    y_log = np.log1p(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print(f"\n📊 Dataset Information:")
    print(f"   • Total samples: {len(df)}")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Test samples: {len(X_test)}")
    print(f"   • Number of features: {X.shape[1]}")
    print(f"   • Target range (original): {y.min():,.0f} to {y.max():,.0f}")
    print(f"   • Target range (log): {y_log.min():.2f} to {y_log.max():.2f}")

    # Train multiple models with different complexities
    models = {
        'Simple (depth=3, n=50)': xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, objective='reg:squarederror'
        ),
        'Medium (depth=6, n=100)': xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, objective='reg:squarederror'
        ),
        'Complex (depth=6, n=200)': xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective='reg:squarederror'
        ),
        'Very Complex (depth=10, n=300)': xgb.XGBRegressor(
            n_estimators=300, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, objective='reg:squarederror'
        )
    }

    results = {}

    print("\n🔬 Training Multiple Models...")
    print("-" * 70)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predictions
        y_train_pred_log = model.predict(X_train)
        y_test_pred_log = model.predict(X_test)

        # Convert back from log scale
        y_train_pred = np.expm1(y_train_pred_log)
        y_test_pred = np.expm1(y_test_pred_log)
        y_train_original = np.expm1(y_train)
        y_test_original = np.expm1(y_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_original, y_train_pred)
        test_mae = mean_absolute_error(y_test_original, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred))
        train_r2 = r2_score(y_train_original, y_train_pred)
        test_r2 = r2_score(y_test_original, y_test_pred)

        # MAPE (excluding very small values to avoid division issues)
        mask_train = y_train_original > 100
        mask_test = y_test_original > 100
        train_mape = mean_absolute_percentage_error(
            y_train_original[mask_train], y_train_pred[mask_train]
        ) * 100 if mask_train.any() else np.nan
        test_mape = mean_absolute_percentage_error(
            y_test_original[mask_test], y_test_pred[mask_test]
        ) * 100 if mask_test.any() else np.nan

        # Cross-validation
        cv_scores = cross_val_score(model, X, y_log, cv=5,
                                  scoring='neg_mean_absolute_error')

        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'y_test_pred': y_test_pred,
            'y_test_original': y_test_original
        }

        print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"   Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
        print(f"   Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    # Analyze overfitting
    print("\n🔍 Overfitting Analysis:")
    print("-" * 70)

    for name, metrics in results.items():
        overfitting_score = abs(metrics['train_r2'] - metrics['test_r2'])
        mae_diff = abs(metrics['train_mae'] - metrics['test_mae'])

        print(f"\n{name}:")
        print(f"   • R² difference (train-test): {overfitting_score:.4f}")
        print(f"   • MAE difference: {mae_diff:.2f}")

        if overfitting_score < 0.01:
            status = "✅ No overfitting detected"
        elif overfitting_score < 0.05:
            status = "⚠️ Slight overfitting"
        elif overfitting_score < 0.1:
            status = "⚠️ Moderate overfitting"
        else:
            status = "❌ Severe overfitting!"

        print(f"   • Status: {status}")

        # Check for perfect fit (suspicious)
        if metrics['train_r2'] > 0.999:
            print(f"   • ⚠️ WARNING: Near-perfect training fit (R²={metrics['train_r2']:.4f})")
            print(f"     This suggests potential overfitting or data leakage!")

    return results, X_train, X_test, y_train, y_test


def analyze_predictions(results):
    """Analyze prediction patterns and errors."""

    print("\n" + "=" * 70)
    print("PREDICTION ANALYSIS")
    print("=" * 70)

    best_model_name = min(results.keys(),
                         key=lambda x: abs(results[x]['train_r2'] - results[x]['test_r2']))
    best_results = results[best_model_name]

    y_test = best_results['y_test_original']
    y_pred = best_results['y_test_pred']

    # Calculate prediction errors
    errors = y_test - y_pred
    relative_errors = np.abs(errors) / y_test * 100

    print(f"\n📈 Best Model: {best_model_name}")
    print(f"   • Test R²: {best_results['test_r2']:.4f}")
    print(f"   • Test MAE: {best_results['test_mae']:.2f}")

    print(f"\n📊 Error Distribution:")
    print(f"   • Mean absolute error: {np.mean(np.abs(errors)):.2f}")
    print(f"   • Median absolute error: {np.median(np.abs(errors)):.2f}")
    print(f"   • 90th percentile error: {np.percentile(np.abs(errors), 90):.2f}")
    print(f"   • Max error: {np.max(np.abs(errors)):.2f}")

    print(f"\n📊 Relative Error (%):")
    print(f"   • Mean: {np.mean(relative_errors):.1f}%")
    print(f"   • Median: {np.median(relative_errors):.1f}%")
    print(f"   • 90th percentile: {np.percentile(relative_errors, 90):.1f}%")

    # Identify problematic predictions
    large_errors = np.abs(errors) > np.percentile(np.abs(errors), 95)
    print(f"\n⚠️ Predictions with largest errors (top 5%):")
    print(f"   • Count: {np.sum(large_errors)}")
    print(f"   • Average actual rank: {np.mean(y_test[large_errors]):.0f}")
    print(f"   • Average predicted rank: {np.mean(y_pred[large_errors]):.0f}")

    return best_model_name, best_results


def create_diagnostic_plots(results, X_test, best_model_name):
    """Create comprehensive diagnostic plots."""

    fig = plt.figure(figsize=(20, 12))

    best_results = results[best_model_name]
    y_test = best_results['y_test_original']
    y_pred = best_results['y_test_pred']

    # 1. Model Comparison
    ax1 = plt.subplot(3, 3, 1)
    model_names = list(results.keys())
    train_r2s = [results[m]['train_r2'] for m in model_names]
    test_r2s = [results[m]['test_r2'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_r2s, width, label='Test R²', alpha=0.8)

    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Complexity Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split('(')[0].strip() for m in model_names], rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Good fit threshold')
    ax1.grid(True, alpha=0.3)

    # 2. Actual vs Predicted (Best Model)
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')

    ax2.set_xlabel('Actual Sales Rank')
    ax2.set_ylabel('Predicted Sales Rank')
    ax2.set_title(f'Best Model: {best_model_name.split("(")[0]}', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add R² and MAE text
    metrics_text = f'R²: {best_results["test_r2"]:.4f}\nMAE: {best_results["test_mae"]:.2f}'
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Residual Plot
    ax3 = plt.subplot(3, 3, 3)
    residuals = y_test - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Sales Rank')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add residual statistics
    res_text = f'Mean: {np.mean(residuals):.2f}\nStd: {np.std(residuals):.2f}'
    ax3.text(0.05, 0.95, res_text, transform=ax3.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 4. Q-Q Plot
    ax4 = plt.subplot(3, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Error Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(np.abs(residuals), bins=30, edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(np.abs(residuals)), color='r', linestyle='--',
               label=f'Mean: {np.mean(np.abs(residuals)):.2f}')
    ax5.axvline(np.median(np.abs(residuals)), color='g', linestyle='--',
               label=f'Median: {np.median(np.abs(residuals)):.2f}')
    ax5.set_xlabel('Absolute Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Prediction vs Error
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(y_pred, np.abs(residuals), alpha=0.5, s=20)
    ax6.set_xlabel('Predicted Sales Rank')
    ax6.set_ylabel('Absolute Error')
    ax6.set_title('Heteroscedasticity Check', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Cross-validation scores
    ax7 = plt.subplot(3, 3, 7)
    cv_means = [results[m]['cv_mae_mean'] for m in model_names]
    cv_stds = [results[m]['cv_mae_std'] for m in model_names]

    ax7.bar(range(len(model_names)), cv_means, yerr=cv_stds,
           color='skyblue', alpha=0.7, capsize=5)
    ax7.set_ylabel('CV MAE')
    ax7.set_title('Cross-Validation Performance', fontweight='bold')
    ax7.set_xticks(range(len(model_names)))
    ax7.set_xticklabels([m.split('(')[0].strip() for m in model_names],
                        rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)

    # 8. Feature Importance
    ax8 = plt.subplot(3, 3, 8)
    model = results[best_model_name]['model']
    importance = model.feature_importances_
    features = X_test.columns
    indices = np.argsort(importance)[-10:]  # Top 10

    ax8.barh(range(len(indices)), importance[indices], color='green', alpha=0.7)
    ax8.set_yticks(range(len(indices)))
    ax8.set_yticklabels(features[indices])
    ax8.set_xlabel('Importance')
    ax8.set_title('Top 10 Feature Importances', fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # 9. Overfitting Analysis
    ax9 = plt.subplot(3, 3, 9)
    overfitting_scores = [abs(results[m]['train_r2'] - results[m]['test_r2'])
                         for m in model_names]

    colors = ['green' if s < 0.01 else 'yellow' if s < 0.05 else 'orange' if s < 0.1 else 'red'
             for s in overfitting_scores]

    bars = ax9.bar(range(len(model_names)), overfitting_scores, color=colors, alpha=0.7)
    ax9.set_ylabel('|Train R² - Test R²|')
    ax9.set_title('Overfitting Analysis', fontweight='bold')
    ax9.set_xticks(range(len(model_names)))
    ax9.set_xticklabels([m.split('(')[0].strip() for m in model_names],
                        rotation=45, ha='right')
    ax9.axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label='No overfit')
    ax9.axhline(y=0.05, color='y', linestyle='--', alpha=0.5, label='Slight')
    ax9.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Severe')
    ax9.legend(loc='upper left')
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Diagnostics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('model_diagnostics.png', dpi=100, bbox_inches='tight')
    print("\n📊 Diagnostic plots saved as 'model_diagnostics.png'")

    return fig


def main():
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"   ✅ Loaded {len(df)} samples with {df.shape[1]} columns")

    # Perform comprehensive evaluation
    print("\n2. Running model diagnostics...")
    results, X_train, X_test, y_train, y_test = comprehensive_model_evaluation(df)

    # Analyze predictions
    print("\n3. Analyzing predictions...")
    best_model_name, best_results = analyze_predictions(results)

    # Create diagnostic plots
    print("\n4. Creating diagnostic visualizations...")
    fig = create_diagnostic_plots(results, X_test, best_model_name)

    # Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    test_r2 = best_results['test_r2']
    train_r2 = best_results['train_r2']
    overfit_score = abs(train_r2 - test_r2)

    print(f"\n🎯 Best Model: {best_model_name}")
    print(f"   • Test R²: {test_r2:.4f}")
    print(f"   • Train R²: {train_r2:.4f}")
    print(f"   • Overfitting score: {overfit_score:.4f}")

    # Determine if performance is reasonable
    issues = []

    if train_r2 > 0.999:
        issues.append("⚠️ Near-perfect training fit (possible overfitting)")

    if test_r2 > 0.99:
        issues.append("⚠️ Suspiciously high test R² (possible data leakage)")

    if overfit_score > 0.1:
        issues.append("❌ Significant overfitting detected")

    if best_results['test_mape'] and best_results['test_mape'] > 50:
        issues.append(f"⚠️ High MAPE: {best_results['test_mape']:.1f}%")

    if len(issues) > 0:
        print("\n🚨 CONCERNS IDENTIFIED:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Recommendations:")
        print("   • Check for data leakage")
        print("   • Consider simpler models")
        print("   • Add regularization")
        print("   • Collect more diverse data")
    else:
        print("\n✅ MODEL PERFORMANCE APPEARS REASONABLE")
        print("   • Good generalization")
        print("   • No severe overfitting")
        print("   • Metrics within expected ranges")

    print("\n" + "=" * 70)

    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()