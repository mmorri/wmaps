"""Analyze codebase to identify which files are actively used."""

import os
import ast

def get_imports(file_path):
    """Extract imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass
    return imports

# Core analysis scripts (final versions we want to keep)
core_scripts = {
    'analyze_json.py': 'Validates JSON and plots major categories',
    'fixed_model.py': 'Fixed XGBoost model with realistic performance',
    'plot_salesrank_histograms.py': 'Creates histogram visualizations',
    'filter_products.py': 'Removes products with invalid sales ranks',
    'verify_filtering.py': 'Verifies data cleaning results'
}

# Utility scripts
utility_scripts = {
    'download_json.py': 'Downloads initial data files',
    'check_data_source.py': 'Verifies data authenticity'
}

# Earlier versions / experimental (potentially removable)
experimental_scripts = {
    'predict_salesrank.py': 'Original model (had overfitting issues)',
    'model_diagnostics.py': 'Diagnostics that revealed overfitting',
    'plot_book_salesrank.py': 'Alternative plotting script'
}

# Unused or test files (likely removable)
unused_scripts = {
    'analyze_prices.py': 'Not referenced in main workflow',
    'book_api.py': 'API implementation not used',
    'check_book_prices.py': 'Price checking not used',
    'sales_rank_prediction.py': 'Appears to be early version',
    'test_api.py': 'Test file for unused API',
    'train_and_save_models.py': 'Model saving not implemented'
}

print("=" * 60)
print("CODEBASE ANALYSIS")
print("=" * 60)

print("\n✅ CORE SCRIPTS (KEEP):")
for file, desc in core_scripts.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  • {file:30} ({size:,} bytes) - {desc}")

print("\n📁 UTILITY SCRIPTS (KEEP):")
for file, desc in utility_scripts.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  • {file:30} ({size:,} bytes) - {desc}")

print("\n⚠️  EXPERIMENTAL (CONSIDER REMOVING):")
for file, desc in experimental_scripts.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  • {file:30} ({size:,} bytes) - {desc}")

print("\n❌ UNUSED (RECOMMEND REMOVING):")
for file, desc in unused_scripts.items():
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  • {file:30} ({size:,} bytes) - {desc}")

# Check data files
print("\n📊 DATA FILES:")
data_files = [
    'products.json',
    'categories.json',
    'products_backup_20250926_095547.json'
]

for file in data_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  • {file:30} ({size:,} bytes)")

# Check generated visualizations
print("\n🎨 VISUALIZATIONS:")
viz_files = [f for f in os.listdir('.') if f.endswith('.png')]
for file in sorted(viz_files):
    size = os.path.getsize(file)
    print(f"  • {file:40} ({size:,} bytes)")

print("\n📁 DIRECTORIES:")
if os.path.exists('book_histograms'):
    files = len(os.listdir('book_histograms'))
    print(f"  • book_histograms/ ({files} files)")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("  Keep: Core + Utility scripts + data files")
print("  Remove: Unused scripts + old model versions")
print("  Archive: Backup file + experimental scripts")
print("=" * 60)