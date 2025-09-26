# Book Sales Rank Prediction API - Setup and Usage Guide

## Prerequisites

Ensure you have Python 3.7+ installed with the following packages:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost
```

## Files Required

1. **products.json** - Product data in NDJSON format
2. **categories.json** - Category hierarchy data
3. **book_api.py** - FastAPI application
4. **train_and_save_models.py** - Model training script

## Setup Instructions

### Step 1: Train the Models

First, train and save the machine learning models:

```bash
python train_and_save_models.py
```

This will create three pickle files:
- `rf_model.pkl` - Random Forest model
- `xgb_model.pkl` - XGBoost model
- `encoders.pkl` - Label encoders and metadata

### Step 2: Start the API Server

Run the FastAPI server:

```bash
python book_api.py
```

The API will start on `http://localhost:8001`

You should see:
```
Loading models...
Models loaded successfully!
Starting server on http://localhost:8001
API documentation available at http://localhost:8001/docs
```

## API Endpoints

### 1. Root Endpoint
```bash
curl http://localhost:8001/
```

Returns API information and available endpoints.

### 2. Health Check
```bash
curl http://localhost:8001/health
```

Returns server status and confirms models are loaded.

### 3. List Categories
```bash
curl http://localhost:8001/categories
```

Returns available book categories with statistics:
- Average sales rank
- Median sales rank
- Sample count

Available categories:
- business
- cooking
- history
- programming
- scifi

### 4. Predict Sales Rank

**Basic prediction (uses default values):**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"category": "programming"}'
```

**With custom parameters:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "cooking",
    "price": 24.99,
    "title_length": 75,
    "has_brand": true
  }'
```

#### Request Parameters:
- `category` (required): Book category (e.g., "cooking", "programming")
- `price` (optional): Book price in dollars
- `title_length` (optional): Length of book title in characters
- `has_brand` (optional): Whether the book has a brand association

#### Response Format:
```json
{
  "category": "cooking",
  "average_sales_rank": 9752.72,
  "samples_in_category": 100,
  "predictions": {
    "random_forest": 688.48,
    "xgboost": 7553.90,
    "average_prediction": 4121.19
  },
  "input_features": {
    "price": 24.99,
    "title_length": 75,
    "has_brand": true
  },
  "available_categories": ["business", "cooking", ...]
}
```

## Example Usage Scenarios

### Scenario 1: Budget Programming Book
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "programming",
    "price": 19.99,
    "title_length": 50
  }'
```

### Scenario 2: Premium Cookbook
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "cooking",
    "price": 45.00,
    "title_length": 100,
    "has_brand": true
  }'
```

### Scenario 3: Science Fiction Novel
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "scifi",
    "price": 14.99,
    "title_length": 120
  }'
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

1. **Swagger UI**: http://localhost:8001/docs
   - Interactive API testing interface
   - Try out endpoints directly from browser
   - View request/response schemas

2. **ReDoc**: http://localhost:8001/redoc
   - Alternative documentation format
   - Clean, readable API reference

## Testing the API

Run the included test script:

```bash
python test_api.py
```

This will:
- Test all endpoints
- Validate responses
- Show example predictions for each category
- Test error handling

## Understanding Predictions

### Sales Rank Interpretation
- **Lower rank = Better sales** (rank 1 is bestseller)
- **Higher rank = Slower sales**

### Model Outputs
- **Random Forest**: Generally more conservative predictions
- **XGBoost**: May show more variation
- **Average Prediction**: Combined estimate from both models

### Key Factors Affecting Predictions
1. **Price**: Lower prices typically predict better ranks
2. **Category**: Each category has different baseline performance
3. **Title Length**: Optimal length varies by category
4. **Brand**: Brand presence can impact sales

## Troubleshooting

### Port Already in Use
If port 8001 is occupied, edit `book_api.py` line 174:
```python
uvicorn.run(app, host="0.0.0.0", port=8002)  # Change to available port
```

### Model Files Not Found
Ensure you run `train_and_save_models.py` first to generate:
- rf_model.pkl
- xgb_model.pkl
- encoders.pkl

### Invalid Category Error
Check available categories with:
```bash
curl http://localhost:8001/categories
```

### Module Not Found Errors
Install missing dependencies:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost pickle
```

## Stopping the Server

Press `Ctrl+C` in the terminal running the API to stop the server.

## Performance Notes

- API response time: typically < 100ms
- Models trained on 1,000 products (500 books, 500 games)
- R² score > 0.999 on test data
- Best performance for books with complete feature data