import json
import numpy as np
import matplotlib.pyplot as plt


def verify_no_zeros():
    """Verify that no products have sales_rank of 0 or missing."""

    print("\n" + "=" * 70)
    print("VERIFICATION: CHECKING FOR ZERO OR MISSING SALES RANKS")
    print("=" * 70)

    # Load products
    products = []
    zero_count = 0
    missing_count = 0
    min_rank = float('inf')
    max_rank = 0

    with open('products.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    product = json.loads(line)
                    products.append(product)

                    salesrank = product.get('salesrank', None)

                    if salesrank is None:
                        missing_count += 1
                        print(f"   ❌ Found product with missing sales_rank: {product.get('title', 'Unknown')[:50]}")
                    elif salesrank == 0:
                        zero_count += 1
                        print(f"   ❌ Found product with zero sales_rank: {product.get('title', 'Unknown')[:50]}")
                    else:
                        min_rank = min(min_rank, salesrank)
                        max_rank = max(max_rank, salesrank)

                except json.JSONDecodeError:
                    continue

    print(f"\n📊 Verification Results:")
    print(f"   • Total products: {len(products)}")
    print(f"   • Products with sales_rank = 0: {zero_count}")
    print(f"   • Products with missing sales_rank: {missing_count}")
    print(f"   • Minimum sales_rank (excluding 0): {min_rank:,}")
    print(f"   • Maximum sales_rank: {max_rank:,}")

    if zero_count == 0 and missing_count == 0:
        print(f"\n✅ SUCCESS: All products have valid sales ranks!")
        print(f"   Sales rank range: {min_rank:,} to {max_rank:,}")
    else:
        print(f"\n❌ WARNING: Found {zero_count + missing_count} products with invalid sales ranks!")

    # Create distribution plot
    all_ranks = []
    for product in products:
        salesrank = product.get('salesrank', None)
        if salesrank and salesrank > 0:
            all_ranks.append(salesrank)

    if all_ranks:
        plt.figure(figsize=(12, 6))

        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        plt.hist(all_ranks, bins=50, edgecolor='black', alpha=0.7, color='blue')
        plt.xlabel('Sales Rank')
        plt.ylabel('Frequency')
        plt.title(f'Sales Rank Distribution\n(All {len(all_ranks)} Products)')
        plt.axvline(np.mean(all_ranks), color='red', linestyle='--', label=f'Mean: {np.mean(all_ranks):,.0f}')
        plt.axvline(np.median(all_ranks), color='green', linestyle='--', label=f'Median: {np.median(all_ranks):,.0f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Log scale
        plt.subplot(1, 2, 2)
        plt.hist(all_ranks, bins=50, edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Sales Rank (log scale)')
        plt.ylabel('Frequency')
        plt.title('Sales Rank Distribution (Log Scale)')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)

        plt.suptitle('Verification: Sales Rank Distribution After Filtering', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('verification_salesrank_distribution.png', dpi=100, bbox_inches='tight')
        print(f"\n📈 Saved distribution plot as 'verification_salesrank_distribution.png'")

    return zero_count, missing_count, min_rank, max_rank


def compare_before_after():
    """Compare statistics before and after filtering."""

    print("\n" + "=" * 70)
    print("COMPARISON: BEFORE vs AFTER FILTERING")
    print("=" * 70)

    # Check if backup exists
    import os
    import glob

    backup_files = glob.glob('products_backup_*.json')

    if not backup_files:
        print("⚠️  No backup file found for comparison")
        return

    backup_file = backup_files[0]  # Use most recent backup

    # Count products in backup (before filtering)
    before_total = 0
    before_with_rank = 0
    before_without_rank = 0

    with open(backup_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    product = json.loads(line)
                    before_total += 1
                    salesrank = product.get('salesrank', None)
                    if salesrank and salesrank > 0:
                        before_with_rank += 1
                    else:
                        before_without_rank += 1
                except json.JSONDecodeError:
                    continue

    # Count products after filtering
    after_total = 0
    with open('products.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                after_total += 1

    print(f"\n📊 Before Filtering (from {backup_file}):")
    print(f"   • Total products: {before_total}")
    print(f"   • With valid sales_rank: {before_with_rank}")
    print(f"   • Without valid sales_rank: {before_without_rank}")
    print(f"   • Percentage with valid ranks: {before_with_rank/before_total*100:.1f}%")

    print(f"\n📊 After Filtering (current products.json):")
    print(f"   • Total products: {after_total}")
    print(f"   • All should have valid sales_rank: {after_total}")
    print(f"   • Products removed: {before_total - after_total}")
    print(f"   • Removal rate: {(before_total - after_total)/before_total*100:.1f}%")

    if after_total == before_with_rank:
        print(f"\n✅ VERIFIED: Filtered file contains exactly {after_total} products with valid sales ranks")
    else:
        print(f"\n⚠️  WARNING: Expected {before_with_rank} products but found {after_total}")


def main():
    print("\n" + "=" * 70)
    print("FILTERING VERIFICATION REPORT")
    print("=" * 70)

    # Verify no zeros or missing values
    zero_count, missing_count, min_rank, max_rank = verify_no_zeros()

    # Compare before and after
    compare_before_after()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    if zero_count == 0 and missing_count == 0:
        print("\n🎉 All verification checks passed!")
        print("   • No products with sales_rank = 0")
        print("   • No products with missing sales_rank")
        print("   • All products have valid sales ranks")
        print(f"   • Valid range: {min_rank:,} to {max_rank:,}")
    else:
        print("\n⚠️  Issues found - please review the output above")

    return zero_count == 0 and missing_count == 0


if __name__ == "__main__":
    success = main()