# Sales Rank Prediction: Books vs Video Games

## Dataset Overview

- **Total Products**: 1,000
- **Products with Sales Rank**: 1,000
  - Books: 500
  - Video Games: 500
- **Sales Rank Range**: 0 - 365,378
- **Mean Sales Rank**: 10,614.75
- **Median Sales Rank**: 894.50

## Feature Engineering

### Features Used
1. **Product Type** (encoded: book vs videogame)
2. **Subcategory** (encoded from category field)
3. **Price** (median-filled for missing values)
4. **Has Price** (binary indicator)
5. **Number of Also Bought** (related products count)
6. **Number of Also Viewed** (related products count)
7. **Number of Bought Together** (related products count)
8. **Title Length** (character count)
9. **Has Brand** (binary indicator)

## Model Performance

### Overall Results (Test Set - 200 samples)

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Random Forest** | 176.04 | 361.45 | 0.9999 |
| **XGBoost** | 208.59 | 309.63 | 0.9999 |

### Performance by Product Type

#### Books (n=104)
| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| MAE | **164.28** | 226.26 |
| Average Actual Sales Rank | 18,123.05 | 18,123.05 |
| Sales Rank Std Dev | 56,770.92 | 56,770.92 |

#### Video Games (n=96)
| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| MAE | 188.77 | **189.45** |
| Average Actual Sales Rank | 1,093.41 | 1,093.41 |
| Sales Rank Std Dev | 1,452.53 | 1,452.53 |

## Feature Importance

### Random Forest
1. **Price**: 79.46%
2. **Title Length**: 13.25%
3. **Subcategory**: 4.06%
4. **Has Brand**: 2.31%
5. **Has Price**: 0.86%
6. **Product Type**: 0.05%
7. Related Products: ~0%

### XGBoost
1. **Price**: 42.11%
2. **Has Price**: 30.46%
3. **Title Length**: 10.31%
4. **Subcategory**: 9.48%
5. **Has Brand**: 6.09%
6. **Product Type**: 1.56%
7. Related Products: ~0%

## Key Insights

### Model Comparison
- **Random Forest** performs better overall with lower MAE (176.04 vs 208.59)
- **Random Forest** is particularly effective for books (MAE: 164.28)
- **XGBoost** shows comparable performance for video games
- Both models achieve exceptional R² scores (>0.999), indicating excellent fit

### Product Type Differences
- **Books** have significantly higher average sales rank (18,123) compared to video games (1,093)
- **Books** show much higher variability in sales rank (std dev: 56,771 vs 1,453)
- This suggests video games have more consistent sales patterns while books vary widely

### Feature Analysis
- **Price is the dominant predictor** for both models (79% and 42% importance)
- **Title length** is the second most important feature
- **Product type itself** has minimal direct importance (<2%), suggesting other features capture the differences
- Related product counts (also bought, also viewed, bought together) show no predictive power

## Recommendations

1. **Use Random Forest for production** - Lower error rates, especially for books
2. **Price data quality is critical** - Ensure accurate and complete pricing information
3. **Consider separate models** for books and video games given their different sales patterns
4. **Investigate additional features** like:
   - Review counts and ratings
   - Publication/release date
   - Author/publisher reputation
   - Seasonal trends

## Technical Details

- **Train/Test Split**: 80/20 (800/200 samples)
- **Random Forest Configuration**:
  - 100 estimators
  - Max depth: 10
  - Min samples split: 5
  - Min samples leaf: 2
- **XGBoost Configuration**:
  - 100 estimators
  - Max depth: 6
  - Learning rate: 0.1
  - Subsample: 0.8