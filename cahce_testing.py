import os

CACHE_PATH = "cache\test_embeddings.pkl"

# Check if file exists
if os.path.exists(CACHE_PATH):
    print(f"✓ Cache file found at: {CACHE_PATH}")
    print(f"File size: {os.path.getsize(CACHE_PATH)} bytes")
else:
    print(f"✗ Cache file NOT found at: {CACHE_PATH}")
    
    # Check if cache directory exists
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        print(f"✓ Cache directory exists")
        print("Files in cache directory:")
        for file in os.listdir(cache_dir):
            print(f"  - {file}")
    else:
        print(f"✗ Cache directory '{cache_dir}' does not exist")