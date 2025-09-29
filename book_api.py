from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import status
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import uvicorn
import os
from typing import Optional, Dict, Any

# Create FastAPI app
app = FastAPI(
    title="Book Sales Rank Prediction API",
    description="API for predicting book sales rank based on category",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals set during startup
rf_model = None
xgb_model = None
le_product = None
le_subcategory = None
feature_columns = []
avg_salesrank_by_category = {}
feature_medians = {}
price_median: float | None = None


@app.on_event("startup")
def load_models() -> None:
    global rf_model, xgb_model, le_product, le_subcategory
    global feature_columns, avg_salesrank_by_category, feature_medians, price_median

    try:
        print("Loading models...")
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('xgb_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            le_product = encoders['le_product']
            le_subcategory = encoders['le_subcategory']
            feature_columns = encoders['feature_columns']
            avg_salesrank_by_category = encoders['avg_salesrank_by_category']
            feature_medians = encoders['feature_medians']
            price_median = encoders['price_median']
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        # Defer failure to endpoints; keep app running with health reporting
        print(f"Warning: model file missing: {e}")
    except Exception as e:
        print(f"Warning: failed to load models: {e}")

# Request model
class BookPredictionRequest(BaseModel):
    category: str = Field(..., description="Book subcategory: business, cooking, history, programming, scifi")
    price: Optional[float] = Field(None, ge=0, description="Price in USD")
    title_length: Optional[int] = Field(None, ge=0, description="Title length in characters")
    has_brand: Optional[bool] = Field(None, description="Whether the item has a brand/publisher")

# Response model
class BookPredictionResponse(BaseModel):
    category: str
    average_sales_rank: Optional[float]
    samples_in_category: Optional[int]
    predictions: Dict[str, float]
    input_features: Dict[str, Any]
    available_categories: list

@app.get("/")
async def root():
    return {
        "message": "Book Sales Rank Prediction API",
        "endpoints": {
            "/predict": "POST - Predict sales rank for a book category",
            "/categories": "GET - List available book categories",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": all([
        rf_model is not None,
        xgb_model is not None,
        le_subcategory is not None,
    ])}

@app.get("/categories")
async def get_categories():
    # Filter only book categories
    book_categories = ['business', 'cooking', 'history', 'programming', 'scifi']
    if le_subcategory is None:
        return {"available_categories": [], "category_statistics": {}}
    available = [cat for cat in book_categories if cat in le_subcategory.classes_]

    categories_with_stats = {}
    for cat in available:
        if cat in avg_salesrank_by_category['mean']:
            categories_with_stats[cat] = {
                "average_sales_rank": avg_salesrank_by_category['mean'][cat],
                "median_sales_rank": avg_salesrank_by_category['median'][cat],
                "sample_count": int(avg_salesrank_by_category['count'][cat])
            }

    return {
        "available_categories": available,
        "category_statistics": categories_with_stats
    }

@app.post("/predict", response_model=BookPredictionResponse)
async def predict_sales_rank(request: BookPredictionRequest):
    # Ensure models are loaded
    if rf_model is None or xgb_model is None or le_subcategory is None or le_product is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Models not loaded. Train/export models and restart the API.")
    # Validate category
    if request.category not in le_subcategory.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Category '{request.category}' not found. Available categories: {list(le_subcategory.classes_)}"
        )

    # Prepare features
    features = pd.DataFrame(columns=feature_columns)

    # Encode book type (always book for this API)
    features.loc[0, 'product_type_encoded'] = le_product.transform(['book'])[0]

    # Encode subcategory
    features.loc[0, 'subcategory_encoded'] = le_subcategory.transform([request.category])[0]

    # Set price (use provided or median)
    if request.price is not None:
        features.loc[0, 'price'] = request.price
        features.loc[0, 'has_price'] = 1
    else:
        features.loc[0, 'price'] = feature_medians.get('price', price_median)
        features.loc[0, 'has_price'] = feature_medians.get('has_price', 0)

    # Set title length (use provided or median)
    if request.title_length is not None:
        features.loc[0, 'title_length'] = request.title_length
    else:
        features.loc[0, 'title_length'] = feature_medians.get('title_length', 50)

    # Set brand (use provided or median)
    if request.has_brand is not None:
        features.loc[0, 'has_brand'] = int(request.has_brand)
    else:
        features.loc[0, 'has_brand'] = feature_medians.get('has_brand', 0)

    # Set other features to median/default values
    features.loc[0, 'num_also_bought'] = feature_medians.get('num_also_bought', 0)
    features.loc[0, 'num_also_viewed'] = feature_medians.get('num_also_viewed', 0)
    features.loc[0, 'num_bought_together'] = feature_medians.get('num_bought_together', 0)

    # Convert to numeric
    features = features.astype(float)

    # Make predictions
    rf_prediction = rf_model.predict(features)[0]
    xgb_prediction = xgb_model.predict(features)[0]

    # Get average sales rank for this category
    avg_rank = None
    sample_count = None
    if request.category in avg_salesrank_by_category['mean']:
        avg_rank = avg_salesrank_by_category['mean'][request.category]
        sample_count = int(avg_salesrank_by_category['count'][request.category])

    # Prepare response
    return BookPredictionResponse(
        category=request.category,
        average_sales_rank=avg_rank,
        samples_in_category=sample_count,
        predictions={
            "random_forest": round(rf_prediction, 2),
            "xgboost": round(xgb_prediction, 2),
            "average_prediction": round((rf_prediction + xgb_prediction) / 2, 2)
        },
        input_features={
            "price": float(features.loc[0, 'price']),
            "title_length": int(features.loc[0, 'title_length']),
            "has_brand": bool(features.loc[0, 'has_brand'])
        },
        available_categories=list(le_subcategory.classes_)
    )

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", "8001"))
    print(f"Starting server on http://localhost:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
