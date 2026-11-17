import requests


def download_json(github_url, local_path, expected_sha256=None):
    resp = requests.get(github_url, timeout=30)
    resp.raise_for_status()
    data = resp.content
    if expected_sha256:
        import hashlib
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError(f"{local_path}: SHA-256 mismatch — refusing to write")
    with open(local_path, "wb") as f:
        f.write(data)
    print(f"Downloaded {github_url} to {local_path}")


download_json(
    "https://raw.githubusercontent.com/mrubash1/ecommerce-sample-data/refs/heads/master/preload_data/products.json",
    "products.json",
)
download_json(
    "https://raw.githubusercontent.com/mrubash1/ecommerce-sample-data/refs/heads/master/preload_data/categories.json",
    "categories.json",
)
