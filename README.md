# E-Commerce Sales Rank Analysis

Analysis and prediction system for Amazon product sales rankings using machine learning.

## 📊 Project Overview

This project analyzes real Amazon product data (books and games) to:
- Validate and clean e-commerce data
- Visualize sales rank distributions by category
- Predict sales rankings using XGBoost and Random Forest models
- Identify top-performing product categories

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
.
├── products.json                 # Main product data (810 items)
├── categories.json              # Category hierarchy
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Multi-service setup
│
├── Core Scripts/
│   ├── analyze_json.py         # Data validation & category analysis
│   ├── fixed_model.py          # Main ML model (Random Forest/XGBoost)
│   ├── filter_products.py      # Data cleaning utility
│   ├── verify_filtering.py     # Data verification
│   └── plot_salesrank_histograms.py  # Visualization generator
│
├── visualizations/             # Generated plots
├── book_histograms/           # Category-specific histograms
├── archive/                   # Unused/experimental scripts
└── backups/                   # Data backups
```

## 🔧 Key Scripts

### 1. **fixed_model.py** - Main Analysis
Trains and evaluates machine learning models with proper regularization.
```bash
python fixed_model.py
```
**Output**: Model performance metrics, predictions by category

### 2. **analyze_json.py** - Data Validation
Validates JSON structure and creates category distribution plots.
```bash
python analyze_json.py
```
**Output**: `category_analysis.png`

### 3. **plot_salesrank_histograms.py** - Histogram Generator
Creates detailed histograms for each book category.
```bash
python plot_salesrank_histograms.py
```
**Output**: Individual histograms in `book_histograms/`

### 4. **filter_products.py** - Data Cleaning
Removes products with invalid sales ranks (0 or missing).
```bash
python filter_products.py
```
**Output**: Cleaned `products.json`, backup created

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

## 🛠️ Development

### Adding New Analysis

1. Create your script in the project root
2. Update `requirements.txt` if needed
3. Rebuild Docker image: `docker-compose build`

### Running Tests

```bash
# Verify data integrity
python verify_filtering.py

# Check model performance
python model_diagnostics.py
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