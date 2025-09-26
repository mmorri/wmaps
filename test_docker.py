#!/usr/bin/env python3
"""Quick test script to verify Docker environment."""

print("=" * 60)
print("DOCKER ENVIRONMENT TEST")
print("=" * 60)

# Test imports
try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✅ seaborn imported successfully")
except ImportError as e:
    print(f"❌ seaborn import failed: {e}")

try:
    import sklearn
    print("✅ scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")

try:
    import xgboost as xgb
    print("✅ xgboost imported successfully")
except ImportError as e:
    print(f"❌ xgboost import failed: {e}")

try:
    import requests
    print("✅ requests imported successfully")
except ImportError as e:
    print(f"❌ requests import failed: {e}")

# Test data files
import os
print("\n📁 Data Files:")
for file in ['products.json', 'categories.json']:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✅ {file}: {size:,} bytes")
    else:
        print(f"  ❌ {file}: NOT FOUND")

# Test Python scripts
print("\n📄 Python Scripts:")
scripts = ['analyze_json.py', 'fixed_model.py', 'filter_products.py']
for script in scripts:
    if os.path.exists(script):
        print(f"  ✅ {script}: Found")
    else:
        print(f"  ❌ {script}: NOT FOUND")

print("\n" + "=" * 60)
print("✅ Docker environment test complete!")
print("=" * 60)