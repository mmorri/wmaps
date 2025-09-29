"""
Train models and export artifacts for the Book API.

Outputs:
- rf_model.pkl
- xgb_model.pkl
- encoders.pkl (label encoders, feature columns, category stats, medians)

Data source: products.json (NDJSON) with optional categories.json
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_products(path: Path) -> list[dict]:
    products: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                products.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return products


def get_product_type(category: str | None) -> str:
    if not category or not isinstance(category, str):
        return "unknown"
    cl = category.lower()
    if "books" in cl:
        return "book"
    if "games" in cl or "video games" in cl:
        return "videogame"
    return "other"


def get_subcategory(category: str | None) -> str:
    if not category or not isinstance(category, str):
        return "unknown"
    parts = category.split("/")
    return parts[-1] if len(parts) > 1 else category


def train_and_export(root: Path) -> None:
    print("Training and exporting models for API...")
    products_path = root / "products.json"
    if not products_path.exists():
        raise FileNotFoundError("products.json not found")

    products = load_products(products_path)
    print(f"Loaded {len(products)} products")

    df = pd.DataFrame(products)
    df = df[df["salesrank"].notna()].copy()
    df["product_type"] = df["category"].apply(get_product_type)
    df = df[df["product_type"].isin(["book", "videogame"])].copy()
    df["subcategory"] = df["category"].apply(get_subcategory)

    # Category stats for books
    books_df = df[df["product_type"] == "book"].copy()
    avg_salesrank_by_category = books_df.groupby("subcategory")["salesrank"].agg(["mean", "median", "count"]).to_dict()

    # Features
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["has_price"] = df["price"].notna().astype(int)
    price_median = float(df["price"].median()) if not np.isnan(df["price"].median()) else 0.0
    df["price"] = df["price"].fillna(price_median)

    def safe_len(x):
        return len(x) if isinstance(x, list) else 0

    df["num_also_bought"] = df.get("also_bought", pd.Series(dtype=object)).apply(safe_len)
    df["num_also_viewed"] = df.get("also_viewed", pd.Series(dtype=object)).apply(safe_len)
    df["num_bought_together"] = df.get("bought_together", pd.Series(dtype=object)).apply(safe_len)
    df["title_length"] = df.get("title", "").fillna("").apply(len)
    df["has_brand"] = df.get("brand").notna().astype(int) if "brand" in df.columns else 0

    # Encoders
    le_product = LabelEncoder()
    le_subcategory = LabelEncoder()
    df["product_type_encoded"] = le_product.fit_transform(df["product_type"])  # book/videogame
    df["subcategory_encoded"] = le_subcategory.fit_transform(df["subcategory"])  # books subcats

    feature_columns = [
        "product_type_encoded",
        "subcategory_encoded",
        "price",
        "has_price",
        "num_also_bought",
        "num_also_viewed",
        "num_bought_together",
        "title_length",
        "has_brand",
    ]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_columns].fillna(0)
    y = df["salesrank"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ).fit(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    ).fit(X_train, y_train)

    # Feature medians restricted to books for better defaults
    feature_medians = {}
    books_only = df[df["product_type"] == "book"]
    for col in feature_columns:
        if col not in ["product_type_encoded", "subcategory_encoded"]:
            feature_medians[col] = float(books_only[col].median()) if len(books_only) > 0 else 0.0

    print("Saving artifacts...")
    with (root / "rf_model.pkl").open("wb") as f:
        pickle.dump(rf_model, f)
    with (root / "xgb_model.pkl").open("wb") as f:
        pickle.dump(xgb_model, f)
    with (root / "encoders.pkl").open("wb") as f:
        pickle.dump(
            {
                "le_product": le_product,
                "le_subcategory": le_subcategory,
                "feature_columns": feature_columns,
                "avg_salesrank_by_category": avg_salesrank_by_category,
                "feature_medians": feature_medians,
                "price_median": price_median,
            },
            f,
        )

    print("Done. Artifacts saved: rf_model.pkl, xgb_model.pkl, encoders.pkl")


if __name__ == "__main__":
    train_and_export(Path("."))

