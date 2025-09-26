import json
import shutil
from datetime import datetime


def filter_products():
    """Remove products with sales_rank equal to 0 or missing."""

    # Backup original file
    backup_filename = f'products_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    shutil.copy('products.json', backup_filename)
    print(f"✅ Created backup: {backup_filename}")

    # Load and filter products
    all_products = []
    removed_products = []
    kept_products = []

    print("\n📂 Loading products from products.json...")

    with open('products.json', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    product = json.loads(line)
                    all_products.append(product)

                    # Check if product has valid sales rank
                    salesrank = product.get('salesrank', None)

                    if salesrank is None or salesrank == 0:
                        # Remove this product
                        removed_products.append({
                            'title': product.get('title', 'Unknown'),
                            'category': product.get('category', 'Unknown'),
                            'salesrank': salesrank
                        })
                    else:
                        # Keep this product
                        kept_products.append(product)

                except json.JSONDecodeError as e:
                    print(f"⚠️  Error parsing line {line_num}: {e}")

    # Save filtered products back to products.json
    print("\n💾 Saving filtered products to products.json...")

    with open('products.json', 'w', encoding='utf-8') as f:
        for product in kept_products:
            f.write(json.dumps(product, separators=(',', ':')) + '\n')

    return all_products, kept_products, removed_products, backup_filename


def print_statistics(all_products, kept_products, removed_products):
    """Print detailed statistics about the filtering operation."""

    print("\n" + "=" * 70)
    print("FILTERING STATISTICS")
    print("=" * 70)

    print(f"\n📊 Overall Statistics:")
    print(f"   • Total products before filtering: {len(all_products)}")
    print(f"   • Products kept (valid sales_rank): {len(kept_products)}")
    print(f"   • Products removed: {len(removed_products)}")
    print(f"   • Removal rate: {len(removed_products)/len(all_products)*100:.1f}%")

    # Analyze removed products by category
    category_removals = {}
    for product in removed_products:
        category = product['category']
        if category not in category_removals:
            category_removals[category] = 0
        category_removals[category] += 1

    print(f"\n📁 Removed Products by Category:")
    # Sort by count of removals
    sorted_categories = sorted(category_removals.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories[:10]:  # Show top 10
        print(f"   • {category}: {count} products removed")

    if len(sorted_categories) > 10:
        print(f"   ... and {len(sorted_categories) - 10} more categories")

    # Show some examples of removed products
    print(f"\n📝 Examples of Removed Products (first 5):")
    for i, product in enumerate(removed_products[:5], 1):
        rank_str = 'None' if product['salesrank'] is None else str(product['salesrank'])
        print(f"   {i}. {product['title'][:50]}...")
        print(f"      Category: {product['category']}, Sales Rank: {rank_str}")

    # Analyze kept products
    print(f"\n✅ Kept Products Analysis:")

    # Category distribution of kept products
    category_kept = {}
    for product in kept_products:
        category = product.get('category', 'Unknown')
        if category not in category_kept:
            category_kept[category] = 0
        category_kept[category] += 1

    print(f"   • Total unique categories: {len(category_kept)}")
    print(f"   • Top categories by product count:")

    sorted_kept = sorted(category_kept.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_kept[:5]:
        print(f"     - {category}: {count} products")


def main():
    print("\n" + "=" * 70)
    print("PRODUCT FILTERING - REMOVING INVALID SALES RANKS")
    print("=" * 70)

    # Filter products
    all_products, kept_products, removed_products, backup_filename = filter_products()

    # Print statistics
    print_statistics(all_products, kept_products, removed_products)

    print("\n" + "=" * 70)
    print("OPERATION COMPLETE")
    print("=" * 70)

    print(f"\n✅ Successfully filtered products.json")
    print(f"📁 Original file backed up as: {backup_filename}")
    print(f"📊 Final result: {len(kept_products)} products with valid sales ranks")

    return kept_products, removed_products


if __name__ == "__main__":
    kept, removed = main()