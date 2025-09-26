import json

products = []
with open('products.json', 'r') as f:
    for line in f:
        try:
            products.append(json.loads(line.strip()))
        except:
            pass

# Check price values
prices = [p.get('price') for p in products if p.get('price') is not None]
print(f'Total products with price field: {len(prices)}')
print(f'Sample prices (first 20): {prices[:20]}')

# Check if prices are in cents
if prices:
    avg_price = sum(prices) / len(prices)
    if avg_price > 100:
        print(f'\nPrices appear to be in cents (average: {avg_price:.0f} = ${avg_price/100:.2f})')
    else:
        print(f'\nPrices appear to be in dollars (average: ${avg_price:.2f})')

# Check books specifically
book_prices = []
game_prices = []
for p in products:
    category = p.get('category', '').lower()
    price = p.get('price')
    if price is not None:
        if 'books' in category:
            book_prices.append(price)
        elif 'games' in category:
            game_prices.append(price)

print(f'\nBook prices (first 10): {book_prices[:10]}')
if book_prices:
    print(f'Book price range: {min(book_prices)} - {max(book_prices)}')
    if min(book_prices) > 100:  # Likely in cents
        print(f'In dollars: ${min(book_prices)/100:.2f} - ${max(book_prices)/100:.2f}')
    print(f'Average book price: {sum(book_prices)/len(book_prices):.0f}')

print(f'\nGame prices (first 10): {game_prices[:10]}')
if game_prices:
    print(f'Game price range: {min(game_prices)} - {max(game_prices)}')
    if min(game_prices) > 100:  # Likely in cents
        print(f'In dollars: ${min(game_prices)/100:.2f} - ${max(game_prices)/100:.2f}')
    print(f'Average game price: {sum(game_prices)/len(game_prices):.0f}')