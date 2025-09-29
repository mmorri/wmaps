import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os


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


def organize_book_data(products):
    """Organize book data by category."""
    category_data = defaultdict(list)

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

                category_data[subcategory].append(salesrank)

    return category_data


def plot_individual_histogram(category, sales_ranks, output_dir='visualizations/book_histograms'):
    """Create and save histogram for a single category."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate statistics
    mean_rank = np.mean(sales_ranks)
    median_rank = np.median(sales_ranks)
    std_rank = np.std(sales_ranks)
    min_rank = np.min(sales_ranks)
    max_rank = np.max(sales_ranks)

    # Color based on performance
    if mean_rank < 20000:
        color = 'green'
        performance = 'Excellent'
    elif mean_rank < 40000:
        color = 'blue'
        performance = 'Good'
    elif mean_rank < 60000:
        color = 'orange'
        performance = 'Average'
    else:
        color = 'red'
        performance = 'Needs Improvement'

    # Plot 1: Regular histogram
    ax1.hist(sales_ranks, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax1.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rank:,.0f}')
    ax1.axvline(median_rank, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_rank:,.0f}')

    ax1.set_xlabel('Sales Rank', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'{category.upper()} Books - Sales Rank Distribution\n({len(sales_ranks)} products)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add statistics box
    stats_text = f'Performance: {performance}\n'
    stats_text += f'Count: {len(sales_ranks)}\n'
    stats_text += f'Std Dev: {std_rank:,.0f}\n'
    stats_text += f'Range: {min_rank:,} - {max_rank:,}'

    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)

    # Plot 2: Log-scale histogram
    ax2.hist(sales_ranks, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax2.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rank:,.0f}')
    ax2.axvline(median_rank, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_rank:,.0f}')

    ax2.set_xlabel('Sales Rank (log scale)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{category.upper()} Books - Log Scale View',
                  fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add percentile information
    percentiles = [25, 50, 75, 90, 95]
    perc_text = 'Percentiles:\n'
    for p in percentiles:
        value = np.percentile(sales_ranks, p)
        perc_text += f'{p}th: {value:,.0f}\n'

    ax2.text(0.98, 0.97, perc_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=9)

    plt.suptitle(f'Sales Rank Analysis: {category.upper()} Category',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    filename = f'{output_dir}/histogram_{category.lower()}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        'category': category,
        'mean': mean_rank,
        'median': median_rank,
        'std': std_rank,
        'count': len(sales_ranks),
        'filename': filename
    }


def plot_combined_histogram(category_data, output_dir='visualizations/book_histograms'):
    """Create combined histogram showing all categories together."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Define color palette
    colors = sns.color_palette('husl', len(category_data))

    # Subplot 1: Overlapping histograms (normalized)
    ax1 = plt.subplot(2, 2, 1)
    for (category, ranks), color in zip(category_data.items(), colors):
        ax1.hist(ranks, bins=30, alpha=0.5, label=category.capitalize(),
                density=True, color=color, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Sales Rank', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Normalized Distribution Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Overlapping histograms (log scale)
    ax2 = plt.subplot(2, 2, 2)
    for (category, ranks), color in zip(category_data.items(), colors):
        ax2.hist(ranks, bins=30, alpha=0.5, label=category.capitalize(),
                density=True, color=color, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Sales Rank (log scale)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Log Scale Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Box plots comparison
    ax3 = plt.subplot(2, 2, 3)
    box_data = [ranks for ranks in category_data.values()]
    box_labels = [cat.capitalize() for cat in category_data.keys()]

    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('Sales Rank', fontsize=11)
    ax3.set_title('Box Plot Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylabel('Sales Rank (log scale)')

    # Rotate x-axis labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Subplot 4: Violin plots
    ax4 = plt.subplot(2, 2, 4)

    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    for category, ranks in category_data.items():
        plot_data.extend(ranks)
        plot_labels.extend([category.capitalize()] * len(ranks))

    df_violin = pd.DataFrame({'Category': plot_labels, 'Sales Rank': plot_data})

    # Create violin plot
    violin_parts = ax4.violinplot(box_data, positions=range(len(box_labels)),
                                  showmeans=True, showmedians=True)

    # Color the violin plots
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax4.set_xticks(range(len(box_labels)))
    ax4.set_xticklabels(box_labels, rotation=45, ha='right')
    ax4.set_ylabel('Sales Rank (log scale)', fontsize=11)
    ax4.set_title('Violin Plot Comparison', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle('Book Categories Sales Rank: Combined Analysis',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    filename = f'{output_dir}/histogram_all_categories.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def create_summary_table(stats_list, output_dir='visualizations/book_histograms'):
    """Create a summary table visualization."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort by mean rank (best to worst)
    stats_list.sort(key=lambda x: x['mean'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for i, stats in enumerate(stats_list, 1):
        performance_emoji = '⭐' if i == 1 else '✅' if i <= 2 else '📊' if i <= 3 else '⚠️'
        table_data.append([
            f"{i}",
            stats['category'].upper(),
            f"{stats['count']}",
            f"{stats['mean']:,.0f}",
            f"{stats['median']:,.0f}",
            f"{stats['std']:,.0f}",
            performance_emoji
        ])

    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'Category', 'Count', 'Mean', 'Median', 'Std Dev', 'Performance'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows based on performance
    for i in range(1, len(table_data) + 1):
        if i == 1:  # Best performer
            color = '#90EE90'  # Light green
        elif i == len(table_data):  # Worst performer
            color = '#FFB6C1'  # Light red
        else:
            color = '#F0F0F0'  # Light gray

        for j in range(7):
            table[(i, j)].set_facecolor(color)

    plt.title('Sales Rank Summary Statistics by Book Category',
             fontsize=14, fontweight='bold', pad=20)

    # Save the figure
    filename = f'{output_dir}/summary_table.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return filename


def main():
    print("\n" + "=" * 70)
    print("BOOK SALES RANK HISTOGRAM ANALYSIS")
    print("=" * 70)

    # Load data
    print("\n📂 Loading products data...")
    products = load_products_data()
    print(f"✅ Loaded {len(products)} products")

    # Organize book data by category
    print("\n📊 Organizing book data by category...")
    category_data = organize_book_data(products)
    print(f"✅ Found {len(category_data)} book categories with sales rank data")

    # Create output directory
    output_dir = 'visualizations/book_histograms'
    os.makedirs(output_dir, exist_ok=True)

    # Plot individual histograms
    print(f"\n📈 Creating individual histograms...")
    stats_list = []

    for category in sorted(category_data.keys()):
        ranks = category_data[category]
        print(f"   • Processing {category}: {len(ranks)} products...", end='')
        stats = plot_individual_histogram(category, ranks, output_dir)
        stats_list.append(stats)
        print(f" ✅ Saved as {stats['filename']}")

    # Plot combined histogram
    print(f"\n📊 Creating combined histogram...")
    combined_file = plot_combined_histogram(category_data, output_dir)
    print(f"   ✅ Saved as {combined_file}")

    # Create summary table
    print(f"\n📋 Creating summary table...")
    summary_file = create_summary_table(stats_list, output_dir)
    print(f"   ✅ Saved as {summary_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\n📁 All files saved in '{output_dir}/' directory:")
    print(f"   • Individual histograms: {len(stats_list)} files")
    print(f"   • Combined analysis: histogram_all_categories.png")
    print(f"   • Summary table: summary_table.png")

    print("\n📊 Category Performance Ranking (by average sales rank):")
    for i, stats in enumerate(sorted(stats_list, key=lambda x: x['mean']), 1):
        emoji = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '
        print(f"   {emoji} {i}. {stats['category'].upper():12} - Avg: {stats['mean']:8,.0f} | Median: {stats['median']:8,.0f}")

    print("\n" + "=" * 70)
    print("✅ All visualizations have been created successfully!")
    print("=" * 70)

    return category_data, stats_list


if __name__ == "__main__":
    category_data, stats_list = main()
