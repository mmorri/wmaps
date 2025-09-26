import json
import pandas as pd

products = []
with open('products.json', 'r') as f:
    for line in f:
        try:
            products.append(json.loads(line.strip()))
        except:
            pass

df = pd.DataFrame(products)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Filter for books only
books = df[df['category'].str.contains('books', case=False, na=False)]
books_with_price = books[books['price'].notna() & (books['price'] > 0) & (books['price'] < 200)]

print('Book Price Statistics:')
print(f'Mean: ${books_with_price["price"].mean():.2f}')
print(f'Median: ${books_with_price["price"].median():.2f}')
print(f'Min: ${books_with_price["price"].min():.2f}')
print(f'Max: ${books_with_price["price"].max():.2f}')
print(f'25th percentile: ${books_with_price["price"].quantile(0.25):.2f}')
print(f'75th percentile: ${books_with_price["price"].quantile(0.75):.2f}')
print(f'\nNumber of books with reasonable prices: {len(books_with_price)}')