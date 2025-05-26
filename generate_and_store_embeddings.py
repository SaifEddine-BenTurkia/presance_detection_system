import os
import random
import numpy as np
import pickle
from deepface import DeepFace
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import tensorflow as tf

# ---- CONFIGURATION ----
dataset_dir = "lfw-deepfunneled/lfw-deepfunneled"
index_name = "face_embeddings_cosine"  # existing index in Kibana
model_name = "Facenet512"
es_host = "http://localhost:9200"
BATCH_SIZE = 32
CACHE_DIR = "cache"
KNOWN_CACHE_PATH = os.path.join(CACHE_DIR, "known_embeddings_cache.pkl")
TEST_OUTPUT_PATH = os.path.join(CACHE_DIR, "test_embeddings.pkl")

# ---- PREPARE CACHE DIRECTORY ----
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- CHECK GPU AVAILABILITY ----
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU detected: {len(gpu_devices)} GPU(s) available." if gpu_devices else "No GPU detected; running on CPU.")

# ---- CONNECT TO ELASTICSEARCH ----
es = Elasticsearch(es_host)

# ---- CLEAR EXISTING DOCUMENTS IN INDEX ----nprint(f"Clearing existing documents in '{index_name}'...")
es.delete_by_query(index=index_name, body={"query": {"match_all": {}}}, refresh=True, ignore=[404])

# ---- HELPER FUNCTION FOR EMBEDDINGS ----
def compute_embeddings_batch(image_paths):
    """Extract and normalize embeddings for a batch of image paths."""
    embeddings = []
    for path in image_paths:
        try:
            res = DeepFace.represent(
                path,
                model_name=model_name,
                enforce_detection=True,
                detector_backend='retinaface',
                align=True
            )
            emb = np.array(res[0]['embedding'])
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
        except Exception as e:
            print(f"Skipping {path}: {e}")
            embeddings.append(None)
    return embeddings

# ---- BULK INDEX FUNCTION ----
def store_embeddings_bulk(embeddings_data):
    """Store tuples of (person_name, file_path, embedding) into Elasticsearch."""
    actions = [
        {
            "_index": index_name,
            "_source": {
                "person_name": name,
                "file_path": path,
                "embedding": emb
            }
        }
        for name, path, emb in embeddings_data if emb is not None
    ]
    if actions:
        bulk(es, actions)

# ---- COLLECT DATA AND SPLIT ----
all_images = {}
for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        all_images[person] = [os.path.join(person_dir, f) for f in os.listdir(person_dir)]

known_images, test_images = [], []
for person, imgs in all_images.items():
    random.shuffle(imgs)
    split = max(1, int(len(imgs) * 0.7))
    known_images.extend((person, p) for p in imgs[:split])
    test_images.extend((person, p) for p in imgs[split:])
print(f"Data split: {len(known_images)} known, {len(test_images)} test")

# ---- PROCESS AND INDEX KNOWN IMAGES IN BATCHES ----
print("Processing and indexing known images in batches...")
batched = []
for person, path in tqdm(known_images, desc="Known batches"):
    batched.append((person, path))
    if len(batched) >= BATCH_SIZE:
        names, paths = zip(*batched)
        embs = compute_embeddings_batch(paths)
        store_embeddings_bulk(list(zip(names, paths, embs)))
        batched = []
# Final batch
if batched:
    names, paths = zip(*batched)
    embs = compute_embeddings_batch(paths)
    store_embeddings_bulk(list(zip(names, paths, embs)))
print("✅ All known embeddings indexed.")

# ---- PROCESS TEST IMAGES IN BATCHES AND SAVE ----
print("Processing test images in batches (not indexing)...")
test_results = []
batched = []
for person, path in tqdm(test_images, desc="Test batches"):
    batched.append((person, path))
    if len(batched) >= BATCH_SIZE:
        names, paths = zip(*batched)
        embs = compute_embeddings_batch(paths)
        test_results.extend(list(zip(names, paths, embs)))
        batched = []
# Final test batch
if batched:
    names, paths = zip(*batched)
    embs = compute_embeddings_batch(paths)
    test_results.extend(list(zip(names, paths, embs)))

with open(TEST_OUTPUT_PATH, 'wb') as f:
    pickle.dump(test_results, f)
print(f"✅ Saved test embeddings file at {TEST_OUTPUT_PATH}.")

print("Pipeline completed successfully.")
