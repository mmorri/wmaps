import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_products_data():
    """Load products from JSON file."""
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
    return products


def analyze_book_sales_rank(products):
    """Analyze sales rank for all book categories."""

    # Dictionary to store sales ranks by category
    category_ranks = defaultdict(list)

    for product in products:
        category = product.get('category', '')

        # Only process book products
        if '/books' in category:
            salesrank = product.get('salesrank', None)

            # Only include products with valid sales rank
            if salesrank and salesrank > 0:
                # Extract subcategory
                if category == '/books':
                    subcategory = 'general'
                else:
                    parts = category.split('/')
                    if len(parts) >= 3:
                        subcategory = parts[2]
                    else:
                        subcategory = 'general'

                category_ranks[subcategory].append(salesrank)

    # Calculate statistics for each category
    category_stats = []
    for cat, ranks in category_ranks.items():
        if ranks:
            category_stats.append({
                'category': cat,
                'count': len(ranks),
                'mean': np.mean(ranks),
                'median': np.median(ranks),
                'std': np.std(ranks),
                'min': np.min(ranks),
                'max': np.max(ranks),
                'q1': np.percentile(ranks, 25),
                'q3': np.percentile(ranks, 75),
                'ranks': ranks
            })

    # Sort by average rank (lower is better)
    category_stats.sort(key=lambda x: x['mean'])

    return category_stats


def create_comprehensive_visualization(category_stats):
    """Create comprehensive visualization of book sales ranks."""

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Bar plot of average sales rank
    ax1 = plt.subplot(2, 3, 1)
    categories = [stat['category'].capitalize() for stat in category_stats]
    means = [stat['mean'] for stat in category_stats]

    bars = ax1.bar(range(len(categories)), means,
                   color=sns.color_palette('viridis', len(categories)))

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(mean):,}', ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_xlabel('Book Category')
    ax1.set_ylabel('Average Sales Rank (lower is better)')
    ax1.set_title('Average Sales Rank by Book Category', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Color code bars - green for better, red for worse
    best_mean = min(means)
    worst_mean = max(means)
    for i, bar in enumerate(bars):
        if means[i] == best_mean:
            bar.set_color('green')
            bar.set_alpha(0.7)
        elif means[i] == worst_mean:
            bar.set_color('red')
            bar.set_alpha(0.7)

    # 2. Box plot of sales rank distribution
    ax2 = plt.subplot(2, 3, 2)
    box_data = [stat['ranks'] for stat in category_stats]
    bp = ax2.boxplot(box_data, tick_labels=categories, patch_artist=True)

    # Color the boxes
    colors = sns.color_palette('husl', len(categories))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xlabel('Book Category')
    ax2.set_ylabel('Sales Rank Distribution')
    ax2.set_title('Sales Rank Distribution by Category', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.set_ylabel('Sales Rank (log scale)')

    # 3. Median vs Mean comparison
    ax3 = plt.subplot(2, 3, 3)
    x = np.arange(len(categories))
    width = 0.35

    means_plot = [stat['mean'] for stat in category_stats]
    medians = [stat['median'] for stat in category_stats]

    bars1 = ax3.bar(x - width/2, means_plot, width, label='Mean', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x + width/2, medians, width, label='Median', alpha=0.8, color='coral')

    ax3.set_xlabel('Book Category')
    ax3.set_ylabel('Sales Rank')
    ax3.set_title('Mean vs Median Sales Rank', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Product count per category
    ax4 = plt.subplot(2, 3, 4)
    counts = [stat['count'] for stat in category_stats]

    bars = ax4.bar(range(len(categories)), counts,
                   color=sns.color_palette('plasma', len(categories)))

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.set_xlabel('Book Category')
    ax4.set_ylabel('Number of Products')
    ax4.set_title('Number of Products per Category', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Standard deviation (variability)
    ax5 = plt.subplot(2, 3, 5)
    stds = [stat['std'] for stat in category_stats]

    bars = ax5.bar(range(len(categories)), stds,
                   color=sns.color_palette('coolwarm', len(categories)))

    # Add value labels
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(std):,}', ha='center', va='bottom', fontsize=9)

    ax5.set_xticks(range(len(categories)))
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    ax5.set_xlabel('Book Category')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_title('Sales Rank Variability by Category', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    # Create summary table
    summary_data = []
    for i, stat in enumerate(category_stats, 1):
        summary_data.append([
            stat['category'].capitalize(),
            stat['count'],
            f"{stat['mean']:,.0f}",
            f"{stat['median']:,.0f}",
            f"{stat['min']:,}",
            f"{stat['max']:,}",
            f"#{i}"
        ])

    table = ax6.table(cellText=summary_data,
                     colLabels=['Category', 'Count', 'Avg Rank', 'Median', 'Min', 'Max', 'Rank'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best and worst performers
    if summary_data:
        # Best performer (first row after header)
        for i in range(7):
            table[(1, i)].set_facecolor('#90EE90')

        # Worst performer (last row)
        for i in range(7):
            table[(len(summary_data), i)].set_facecolor('#FFB6C1')

    ax6.set_title('Summary Statistics (Ranked by Avg Sales Rank)',
                  fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Book Categories Sales Rank Analysis\n(Lower Rank = Better Sales Performance)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    plt.savefig('book_categories_salesrank.png', dpi=100, bbox_inches='tight')
    print("\n✅ Visualization saved as 'book_categories_salesrank.png'")


def print_detailed_analysis(category_stats):
    """Print detailed analysis of the results."""

    print("\n" + "=" * 70)
    print("📚 DETAILED BOOK CATEGORY SALES RANK ANALYSIS")
    print("=" * 70)

    print("\n📊 Categories Ranked by Performance (Best to Worst):")
    print("-" * 70)

    for i, stat in enumerate(category_stats, 1):
        print(f"\n{i}. {stat['category'].upper()}")
        print(f"   • Products: {stat['count']}")
        print(f"   • Average Rank: {stat['mean']:,.0f}")
        print(f"   • Median Rank: {stat['median']:,.0f}")
        print(f"   • Std Deviation: {stat['std']:,.0f}")
        print(f"   • Range: {stat['min']:,} - {stat['max']:,}")
        print(f"   • IQR: Q1={stat['q1']:,.0f}, Q3={stat['q3']:,.0f}")

        # Performance indicator
        if i == 1:
            print(f"   • Performance: ⭐ BEST PERFORMER")
        elif i == len(category_stats):
            print(f"   • Performance: ⚠️ NEEDS IMPROVEMENT")

    # Overall statistics
    print("\n" + "=" * 70)
    print("📈 OVERALL STATISTICS")
    print("=" * 70)

    all_means = [stat['mean'] for stat in category_stats]
    print(f"\n• Best Performing Category: {category_stats[0]['category'].capitalize()}")
    print(f"  Average Rank: {category_stats[0]['mean']:,.0f}")
    print(f"\n• Worst Performing Category: {category_stats[-1]['category'].capitalize()}")
    print(f"  Average Rank: {category_stats[-1]['mean']:,.0f}")
    print(f"\n• Performance Gap: {category_stats[-1]['mean'] - category_stats[0]['mean']:,.0f} rank positions")
    print(f"• Average Across All Categories: {np.mean(all_means):,.0f}")

    # Insights
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)

    # Find categories with high variability
    high_var_categories = [stat for stat in category_stats if stat['std'] > np.mean([s['std'] for s in category_stats])]

    print("\n• Categories with High Variability (inconsistent performance):")
    for stat in high_var_categories:
        print(f"  - {stat['category'].capitalize()}: std={stat['std']:,.0f}")

    # Categories with skewed distributions
    print("\n• Distribution Skewness (Mean vs Median):")
    for stat in category_stats:
        skew_ratio = stat['mean'] / stat['median'] if stat['median'] > 0 else 0
        if skew_ratio > 2:
            print(f"  - {stat['category'].capitalize()}: Highly right-skewed (few very poor performers)")
        elif skew_ratio > 1.5:
            print(f"  - {stat['category'].capitalize()}: Moderately right-skewed")

    print("\n" + "=" * 70)


def main():
    print("\n" + "=" * 70)
    print("BOOK SALES RANK ANALYSIS BY CATEGORY")
    print("=" * 70)

    # Load data
    print("\n📂 Loading products data...")
    products = load_products_data()
    print(f"✅ Loaded {len(products)} products")

    # Analyze book sales ranks
    print("\n📊 Analyzing book categories...")
    category_stats = analyze_book_sales_rank(products)

    books_with_rank = sum(stat['count'] for stat in category_stats)
    print(f"✅ Found {len(category_stats)} book categories")
    print(f"✅ Total books with sales rank: {books_with_rank}")

    # Print detailed analysis
    print_detailed_analysis(category_stats)

    # Create visualizations
    print("\n🎨 Creating visualizations...")
    create_comprehensive_visualization(category_stats)

    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)

    return category_stats


if __name__ == "__main__":
    category_stats = main()