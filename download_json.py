import requests


def download_json(github_url, local_path):
    resp = requests.get(github_url)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded {github_url} to {local_path}")


download_json(
    "https://raw.githubusercontent.com/mrubash1/ecommerce-sample-data/refs/heads/master/preload_data/products.json",
    "products.json",
)
download_json(
    "https://raw.githubusercontent.com/mrubash1/ecommerce-sample-data/refs/heads/master/preload_data/categories.json",
    "categories.json",
)
