# E-Commerce Sales Rank Analysis

A comprehensive machine learning system for analyzing and predicting Amazon product sales rankings, featuring data visualization, REST API, and containerized deployment.

## 📊 Project Overview

This project provides end-to-end analysis of Amazon product data (810 items across books and games categories) to predict sales performance using advanced machine learning techniques. The system includes data cleaning pipelines, visualization tools, trained ML models, and a production-ready API for real-time predictions.

### Key Capabilities
- **Data Processing**: Automated validation and cleaning of e-commerce JSON data
- **Machine Learning**: Random Forest and XGBoost models with 80.4% R² accuracy
- **Visualization**: Comprehensive charts and histograms for data insights
- **API Service**: FastAPI endpoint for real-time sales rank predictions
- **Containerization**: Docker support for easy deployment and scaling

## 🚀 Quick Start with Docker

### Using Docker Compose

```bash
# Build and run the main analysis
docker-compose up sales-analyzer

# Run data validation
docker-compose up data-validator

# Generate histograms
docker-compose up histogram-generator

# Start Jupyter notebook for interactive analysis
docker-compose up jupyter
# Then visit http://localhost:8888
```

### Using Docker directly

```bash
# Build the image
docker build -t ecommerce-analyzer .

# Run the analysis
docker run -v $(pwd)/visualizations:/app/visualizations ecommerce-analyzer

# Run specific script
docker run -v $(pwd)/visualizations:/app/visualizations ecommerce-analyzer python analyze_json.py
```

## 🐍 Local Installation

### Requirements
- Python 3.9+
- pip or uv package manager

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt
```

## 📁 Project Structure

```
wmaps/
│
├── 📊 DATA FILES
│   ├── products.json              # Main dataset: 810 Amazon products
│   ├── categories.json            # Category hierarchy mapping
│   ├── rf_model.pkl              # Trained Random Forest model
│   ├── xgb_model.pkl             # Trained XGBoost model
│   └── encoders.pkl              # Label encoders and feature metadata
│
├── 🔧 CORE ANALYSIS SCRIPTS
│   ├── analyze_json.py           # Validates JSON data and generates category distribution plots
│   ├── fixed_model.py            # Main ML training script with regularization
│   ├── filter_products.py        # Removes invalid data (0 or missing sales ranks)
│   ├── verify_filtering.py       # Verifies data integrity post-cleaning
│   └── model_diagnostics.py      # Evaluates model performance and generates metrics
│
├── 📈 VISUALIZATION SCRIPTS
│   ├── plot_book_salesrank.py    # Creates sales rank plots by book category
│   ├── plot_salesrank_histograms.py # Generates detailed histogram visualizations
│   └── predict_salesrank.py      # Runs predictions and creates comparison charts
│
├── 🌐 API & SERVICES
│   ├── book_api.py               # FastAPI service for real-time predictions
│   ├── test_api.py               # API endpoint testing suite
│   └── test_docker.py            # Docker container health checks
│
├── 🐳 DEPLOYMENT
│   ├── Dockerfile                # Container configuration
│   ├── docker-compose.yml        # Multi-service orchestration
│   ├── .dockerignore            # Docker build exclusions
│   └── requirements.txt          # Python dependencies
│
├── 📊 OUTPUT DIRECTORIES
│   ├── visualizations/           # All generated charts and plots
│   │   └── book_histograms/     # Category-specific histogram images
│   ├── backups/                 # Automatic data backups
│   └── archive/                 # Legacy/experimental scripts
│
└── 📝 DOCUMENTATION
    ├── README.md                 # This file
    ├── api_test_results.md      # API testing documentation
    ├── run_api.md               # API usage guide
    └── book_vs_games.md         # Category comparison analysis
```

## 🔧 File Descriptions

### Core ML Pipeline
- **`fixed_model.py`**: Main training script that builds Random Forest and XGBoost models with L1/L2 regularization, cross-validation, and performance evaluation
- **`model_diagnostics.py`**: Comprehensive model analysis including residual plots, feature importance, and prediction vs actual comparisons
- **`predict_salesrank.py`**: Generates predictions for all categories and creates comparative visualizations

### Data Processing
- **`analyze_json.py`**: Validates JSON integrity, checks for missing fields, and generates category distribution charts
- **`filter_products.py`**: Cleans dataset by removing entries with invalid/zero sales ranks, creates automatic backups
- **`verify_filtering.py`**: Post-processing verification to ensure data quality and consistency
- **`check_data_source.py`**: Analyzes data provenance and structure
- **`download_json.py`**: Utility for fetching product data from remote sources

### Visualization Suite
- **`plot_book_salesrank.py`**: Creates detailed sales rank analysis for book categories with statistical overlays
- **`plot_salesrank_histograms.py`**: Generates comprehensive histogram visualizations with distribution statistics

### API Service
- **`book_api.py`**: FastAPI application providing REST endpoints for sales rank predictions
  - `GET /` - API documentation
  - `GET /health` - Service health check
  - `GET /categories` - List available categories with statistics
  - `POST /predict` - Get sales rank predictions for a book
- **`test_api.py`**: Automated testing suite for API endpoints
- **`test_docker.py`**: Container health and integration tests

## 📈 Model Performance

### Best Model: Random Forest
- **R² Score**: 0.804 (80.4% variance explained)
- **MAE**: 4,362 (realistic error for 365K range)
- **MAPE**: 45.7% (reasonable uncertainty)

### Category Rankings (Lower rank = Better sales)
1. 🥇 **Business**: 13,424
2. 🥈 **SciFi**: 16,001
3. 🥉 **Cooking**: 16,646
4. **History**: 20,071
5. **Programming**: 22,133

## 📊 Data Details

- **Source**: Real Amazon product data (2009-2014)
- **Size**: 810 products after cleaning
- **Categories**: Books (5 subcategories) and Games (5 platforms)
- **Features**: Price, category, title length, brand, images, etc.

## 🚀 Running the API

### Local Development
```bash
# Start the API server
python book_api.py

# API will be available at:
# http://localhost:8001
# Documentation: http://localhost:8001/docs
```

### Example API Request
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"category": "programming", "price": 39.99}'
```

### Docker Deployment
```bash
docker-compose up book-api
```

## 🛠️ Development Workflow

### 1. Data Pipeline
```bash
# Step 1: Validate raw data
python analyze_json.py

# Step 2: Clean data
python filter_products.py

# Step 3: Verify cleaning
python verify_filtering.py

# Step 4: Train models
python fixed_model.py

# Step 5: Generate visualizations
python plot_salesrank_histograms.py
```

### 2. Model Testing
```bash
# Run diagnostics
python model_diagnostics.py

# Test predictions
python predict_salesrank.py
```

### 3. API Testing
```bash
# Start API
python book_api.py

# Run tests
python test_api.py
```

## 📝 Notes

- All models use log-transformed sales ranks for better performance
- Regularization prevents overfitting (L1/L2 penalties, shallow trees)
- Cross-validation ensures robust performance estimates

## 🤝 Contributing

1. Keep unused scripts in `archive/` directory
2. Save visualizations in `visualizations/`
3. Document new features in this README
4. Maintain realistic model expectations (R² < 0.95)

## 📄 License

This project uses publicly available e-commerce sample data for educational purposes.