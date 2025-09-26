import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def check_json_formatting(filepath):
    """Check JSON file for formatting errors and validate structure."""
    errors = []
    warnings = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, (list, dict)):
            errors.append(f"Root element should be list or dict, got {type(data).__name__}")
            return errors, warnings, None

        if isinstance(data, list) and len(data) == 0:
            warnings.append("JSON file contains empty list")

        return errors, warnings, data

    except json.JSONDecodeError as e:
        errors.append(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
        return errors, warnings, None
    except FileNotFoundError:
        errors.append(f"File not found: {filepath}")
        return errors, warnings, None
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return errors, warnings, None


def validate_products_structure(products):
    """Validate the structure of products data."""
    errors = []
    warnings = []
    required_fields = {'title', 'category'}
    optional_fields = {'price', 'currency', '_id', 'images', 'description', 'metadata'}

    for i, product in enumerate(products):
        if not isinstance(product, dict):
            errors.append(f"Product {i}: Expected dict, got {type(product).__name__}")
            continue

        # Check required fields
        missing = required_fields - set(product.keys())
        if missing:
            warnings.append(f"Product {i} ({product.get('title', 'Unknown')}): Missing fields {missing}")

        # Check for unexpected fields (informational)
        all_expected = required_fields | optional_fields
        extra_fields = set(product.keys()) - all_expected

        # Validate specific field types
        if 'price' in product and not isinstance(product['price'], (int, float)):
            warnings.append(f"Product {i} ({product.get('title', 'Unknown')}): Price should be numeric")

        if 'category' in product and not isinstance(product['category'], str):
            errors.append(f"Product {i} ({product.get('title', 'Unknown')}): Category should be string")

    return errors, warnings


def validate_categories_structure(categories):
    """Validate the structure of categories data."""
    errors = []
    warnings = []
    required_fields = {'name', 'category', 'parent'}

    for i, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append(f"Category {i}: Expected dict, got {type(category).__name__}")
            continue

        # Check required fields
        missing = required_fields - set(category.keys())
        if missing:
            errors.append(f"Category {i} ({category.get('name', 'Unknown')}): Missing fields {missing}")

        # Validate parent-child relationships
        if 'children' in category and not isinstance(category['children'], list):
            errors.append(f"Category {i} ({category.get('name', 'Unknown')}): Children should be a list")

    return errors, warnings


def analyze_categories(products):
    """Analyze product distribution across categories."""
    category_counts = Counter()
    category_prices = defaultdict(list)

    for product in products:
        if 'category' in product:
            category = product['category']
            category_counts[category] += 1

            if 'price' in product and isinstance(product['price'], (int, float)):
                category_prices[category].append(product['price'] / 100)  # Convert cents to dollars

    # Extract major categories (top level)
    major_categories = defaultdict(int)
    for cat, count in category_counts.items():
        if '/' in cat:
            parts = cat.split('/')
            if len(parts) >= 2:
                major_cat = '/' + parts[1] if parts[1] else 'root'
                major_categories[major_cat] += count

    return category_counts, category_prices, major_categories


def plot_category_distribution(major_categories, category_counts, category_prices):
    """Create visualizations for category distribution."""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Major categories pie chart
    if major_categories:
        ax1 = axes[0, 0]
        sorted_major = sorted(major_categories.items(), key=lambda x: x[1], reverse=True)
        labels = [cat.strip('/') or 'root' for cat, _ in sorted_major]
        sizes = [count for _, count in sorted_major]
        colors = sns.color_palette('husl', len(labels))

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90)
        ax1.set_title('Distribution of Products by Major Category', fontsize=12, fontweight='bold')

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

    # 2. Top 10 subcategories bar chart
    ax2 = axes[0, 1]
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_categories:
        categories = [cat for cat, _ in top_categories]
        counts = [count for _, count in top_categories]

        bars = ax2.bar(range(len(categories)), counts, color=sns.color_palette('viridis', len(categories)))
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat.split('/')[-1] if '/' in cat else cat for cat in categories],
                            rotation=45, ha='right')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Number of Products')
        ax2.set_title('Top 10 Product Categories', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)

    # 3. Price distribution by major category (box plot)
    ax3 = axes[1, 0]
    price_data = []
    price_labels = []

    for cat in sorted(major_categories.keys()):
        cat_prices = []
        for subcat, prices in category_prices.items():
            if subcat.startswith(cat):
                cat_prices.extend(prices)
        if cat_prices:
            price_data.append(cat_prices)
            price_labels.append(cat.strip('/') or 'root')

    if price_data:
        bp = ax3.boxplot(price_data, tick_labels=price_labels, patch_artist=True)

        # Color the boxes
        colors = sns.color_palette('husl', len(price_labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_xlabel('Major Category')
        ax3.set_ylabel('Price ($)')
        ax3.set_title('Price Distribution by Major Category', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    # Calculate summary statistics
    summary_data = []
    total_products = sum(major_categories.values())

    for cat in sorted(major_categories.keys()):
        cat_prices = []
        for subcat, prices in category_prices.items():
            if subcat.startswith(cat):
                cat_prices.extend(prices)

        count = major_categories[cat]
        percentage = (count / total_products) * 100 if total_products > 0 else 0

        if cat_prices:
            avg_price = sum(cat_prices) / len(cat_prices)
            min_price = min(cat_prices)
            max_price = max(cat_prices)
        else:
            avg_price = min_price = max_price = 0

        summary_data.append([
            cat.strip('/') or 'root',
            count,
            f'{percentage:.1f}%',
            f'${avg_price:.2f}',
            f'${min_price:.2f}',
            f'${max_price:.2f}'
        ])

    # Add total row
    all_prices = [p for prices in category_prices.values() for p in prices]
    if all_prices:
        summary_data.append([
            'TOTAL',
            total_products,
            '100.0%',
            f'${sum(all_prices)/len(all_prices):.2f}',
            f'${min(all_prices):.2f}',
            f'${max(all_prices):.2f}'
        ])

    table = ax4.table(cellText=summary_data,
                     colLabels=['Category', 'Count', '%', 'Avg Price', 'Min Price', 'Max Price'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style total row
    if len(summary_data) > 1:
        for i in range(6):
            table[(len(summary_data), i)].set_facecolor('#FFC107')
            table[(len(summary_data), i)].set_text_props(weight='bold')

    ax4.set_title('Category Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('E-Commerce Product Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    plt.savefig('category_analysis.png', dpi=100, bbox_inches='tight')
    print("\nVisualization saved as 'category_analysis.png'")


def main():
    print("=" * 60)
    print("JSON File Analysis and Validation")
    print("=" * 60)

    # Check categories.json
    print("\n1. Checking categories.json...")
    print("-" * 40)
    cat_errors, cat_warnings, categories_data = check_json_formatting('categories.json')

    if cat_errors:
        print("L ERRORS found in categories.json:")
        for error in cat_errors:
            print(f"   • {error}")
    else:
        print(" No formatting errors in categories.json")

        if isinstance(categories_data, list):
            struct_errors, struct_warnings = validate_categories_structure(categories_data)

            if struct_errors:
                print("\nL Structure ERRORS:")
                for error in struct_errors:
                    print(f"   • {error}")

            if struct_warnings:
                print("\n�  Structure WARNINGS:")
                for warning in struct_warnings:
                    print(f"   • {warning}")

            if not struct_errors and not struct_warnings:
                print(" Category structure is valid")

            print(f"\n=� Found {len(categories_data)} categories")

    if cat_warnings:
        print("\n�  WARNINGS:")
        for warning in cat_warnings:
            print(f"   • {warning}")

    # Check products.json
    print("\n2. Checking products.json...")
    print("-" * 40)

    # Read products.json line by line since it's not standard JSON array
    products_data = []
    line_errors = []

    try:
        with open('products.json', 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        product = json.loads(line)
                        products_data.append(product)
                    except json.JSONDecodeError as e:
                        line_errors.append(f"Line {line_num}: {e.msg}")
    except FileNotFoundError:
        print("L products.json file not found")
        sys.exit(1)

    if line_errors:
        print(f"L Found {len(line_errors)} lines with JSON errors:")
        for error in line_errors[:5]:  # Show first 5 errors
            print(f"   • {error}")
        if len(line_errors) > 5:
            print(f"   ... and {len(line_errors) - 5} more errors")
    else:
        print(" No formatting errors in products.json")

    print(f"\n=� Successfully parsed {len(products_data)} products")

    if products_data:
        struct_errors, struct_warnings = validate_products_structure(products_data)

        if struct_errors:
            print("\nL Structure ERRORS:")
            for error in struct_errors[:5]:  # Show first 5
                print(f"   • {error}")
            if len(struct_errors) > 5:
                print(f"   ... and {len(struct_errors) - 5} more errors")

        if struct_warnings:
            print("\n�  Structure WARNINGS:")
            for warning in struct_warnings[:5]:  # Show first 5
                print(f"   • {warning}")
            if len(struct_warnings) > 5:
                print(f"   ... and {len(struct_warnings) - 5} more warnings")

        if not struct_errors and not struct_warnings:
            print(" Product structure is valid")

    # Analyze categories
    if products_data:
        print("\n3. Analyzing Product Categories...")
        print("-" * 40)

        category_counts, category_prices, major_categories = analyze_categories(products_data)

        print(f"=� Found {len(category_counts)} unique categories")
        print(f"=� Major categories: {', '.join(sorted(major_categories.keys()))}")

        print("\n=� Product distribution:")
        for cat in sorted(major_categories.keys()):
            count = major_categories[cat]
            percentage = (count / len(products_data)) * 100
            print(f"   • {cat.strip('/') or 'root'}: {count} products ({percentage:.1f}%)")

        # Create visualizations
        print("\n4. Creating Visualizations...")
        print("-" * 40)
        try:
            plot_category_distribution(major_categories, category_counts, category_prices)
            print(" Visualizations created successfully!")
        except Exception as e:
            print(f"L Error creating visualizations: {e}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()