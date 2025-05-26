import os
import numpy as np
from deepface import DeepFace
from elasticsearch import Elasticsearch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import pickle
import tensorflow as tf

# --- Configurations ---
es = Elasticsearch("http://127.0.0.1:9200" )
index_name = "face_embeddings"
image_db_directory = "lfw-deepfunneled/lfw-deepfunneled"
thresholds = np.arange(0.90, 1.0, 0.01)
EMBEDDING_CACHE_FILE = ".\cache\embeddings_cache.pkl"
BATCH_SIZE = 32  # Default batch size

# Check for GPU availability with TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    USE_GPU = True
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")
    print(f"Using GPU: {gpus}")
    tf.config.set_visible_devices(gpus[0], 'GPU')
    BATCH_SIZE = 32
else:
    USE_GPU = False
    print("No GPU found, using CPU")
    BATCH_SIZE = 2048

class FaceEmbeddingDataset:
    def __init__(self, image_directory, batch_size=32):
        self.image_directory = image_directory
        self.batch_size = batch_size
        self.image_paths = []
        self.person_names = []
        self._build_dataset()
        
    def _build_dataset(self):
        """Build lists of all images and corresponding person names"""
        print("Building dataset from directory structure...")
        for person_name in os.listdir(self.image_directory):
            person_path = os.path.join(self.image_directory, person_name)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, img_name)
                    self.image_paths.append(image_path)
                    self.person_names.append(person_name)
        
        print(f"Found {len(self.image_paths)} images from {len(set(self.person_names))} unique people")
    
    def get_batched_data(self):
        """Get data in batches without using TF Dataset API"""
        num_samples = len(self.image_paths)
        for i in range(0, num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, num_samples)
            yield self.image_paths[i:end_idx], self.person_names[i:end_idx]

def process_batch(batch_paths, batch_persons):
    """Process a batch of images with DeepFace"""
    results = {}
    
    for i, (image_path, person_name) in enumerate(zip(batch_paths, batch_persons)):
        try:
            result = DeepFace.represent(
                image_path, 
                model_name="Facenet512", 
                enforce_detection=False,
                detector_backend='retinaface',
                align=True,
            )
            embedding = np.array(result[0]["embedding"])
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            results[image_path] = (person_name, embedding)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results

def precompute_embeddings(use_cache=True):
    """Precompute all embeddings with batched processing and optional caching"""
    if use_cache and os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, 'rb') as f:
            print("Loading embeddings from cache...")
            cache_data = pickle.load(f)
            # Check if cache is empty
            if cache_data and len(cache_data) > 0:
                print(f"Using {len(cache_data)} embeddings from cache")
                return cache_data
            else:
                print("Cache file exists but is empty, regenerating embeddings...")
    
    embedding_cache = {}
    
    # Create dataset and get batches
    dataset = FaceEmbeddingDataset(image_db_directory, batch_size=BATCH_SIZE)
    total_batches = len(dataset.image_paths) // BATCH_SIZE + (1 if len(dataset.image_paths) % BATCH_SIZE > 0 else 0)
    
    # Process in batches using simple Python iterators instead of TF Dataset
    for i, (paths_batch, persons_batch) in enumerate(tqdm(dataset.get_batched_data(), total=total_batches, desc="Processing batches")):
        # Process the batch
        batch_results = process_batch(paths_batch, persons_batch)
        embedding_cache.update(batch_results)
        
        # Periodically save cache
        if (i + 1) % 20 == 0:
            with open(EMBEDDING_CACHE_FILE, 'wb') as f:
                pickle.dump(embedding_cache, f)
            print(f"Saved checkpoint: {len(embedding_cache)}/{len(dataset.image_paths)} images processed")
    
    # Final cache save
    with open(EMBEDDING_CACHE_FILE, 'wb') as f:
        pickle.dump(embedding_cache, f)
    
    return embedding_cache

# --- Elasticsearch Functions ---
def search_es_for_similarities(query_embedding):
    try:
        response = es.search(
            index=index_name,
            knn={
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 5,
                "num_candidates": 100
            }
        )
        return response['hits']['hits']
    except Exception as e:
        import traceback
        print(f"Traceback {traceback.print_exc()}")
        print(f"Elasticsearch error: {e}")
        return []

# --- Evaluation Functions ---
def evaluate_batch(batch_items, threshold):
    """Process a batch of items for evaluation with finer threshold control"""
    y_true_batch, y_pred_batch = [], []
    
    for image_path, (person_name, embedding) in batch_items:
        hits = search_es_for_similarities(embedding)
        
        recognized_name = "Unknown"
        
        if hits:
            top_hit = hits[0]
            top_score = top_hit['_score']
            top_hit_name = top_hit['_source']['person_name']
            
            # Apply threshold directly (no normalization needed)
            if top_score >= threshold:
                recognized_name = top_hit_name
            
            # Optional debugging for every 100th item
            if len(y_true_batch) % 100 == 0:
                print(f"Sample score: {top_score:.4f}, Threshold: {threshold:.4f}, Match: {top_score >= threshold}")
        
        y_true_batch.append(person_name)
        y_pred_batch.append(recognized_name)
    
    return y_true_batch, y_pred_batch

def evaluate_with_threshold(embedding_cache, threshold):
    """Evaluate performance with batched processing"""
    if not embedding_cache:
        print("Error: No embeddings to evaluate")
        return compute_metrics([], [])
    
    y_true, y_pred = [], []
    items = list(embedding_cache.items())
    
    # Process in batches
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    
    for batch in tqdm(batches, desc=f"Evaluating threshold {threshold:.1f}"):
        y_true_batch, y_pred_batch = evaluate_batch(batch, threshold)
        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)
    
    return compute_metrics(y_true, y_pred)

def compute_metrics(y_true, y_pred):
    """Optimized metric computation"""
    if not y_true or not y_pred:
        print("Warning: Empty evaluation data")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'TN': 0,
            'sensitivity': 0
        }
        
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Vectorized computation
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = np.where((TP + FN) > 0, TP / (TP + FN), 0)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'TP': TP.sum(),
        'FP': FP.sum(),
        'FN': FN.sum(),
        'TN': TN.sum(),
        'sensitivity': sensitivity.mean()
    }

# --- Visualization Functions ---
def plot_results(results):
    """Plot all metrics and results"""
    # Extract data for plotting
    thresholds = [r[0] for r in results]
    metrics = [r[1] for r in results]
    
    # Plot performance metrics
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, [m['accuracy'] for m in metrics], label='Accuracy', marker='o')
    plt.plot(thresholds, [m['precision'] for m in metrics], label='Precision', marker='s')
    plt.plot(thresholds, [m['recall'] for m in metrics], label='Recall', marker='^')
    plt.plot(thresholds, [m['f1'] for m in metrics], label='F1 Score', marker='x')
    plt.title('Performance Metrics vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create metrics table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [
        [
            m['TP'], m['FP'], m['FN'], m['TN'], 
            f"{m['sensitivity']:.4f}",
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}"
        ] for m in metrics
    ]
    
    columns = ["TP", "FP", "FN", "TN", "Sensitivity", "Accuracy", "Precision", "Recall", "F1"]
    ax.table(
        cellText=table_data,
        rowLabels=[f"Thresh {t:.1f}" for t in thresholds],
        colLabels=columns,
        loc='center'
    )
    plt.title("Detailed Performance Metrics", pad=20)
    plt.tight_layout()
    plt.savefig("threshold_metrics.png")
    plt.show()

# --- Main Execution ---
def main():
    start_time = time()
    
    try:
        # Step 1: Ensure TensorFlow is using GPU
        if USE_GPU:
            print(f"TensorFlow is using GPU: {tf.test.is_gpu_available()}")
            print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
        # # Step 2: Precompute all embeddings (with caching)
        # embedding_cache = precompute_embeddings(use_cache=True)
        # print(f"Loaded {len(embedding_cache)} embeddings")
        
        # # Exit if no embeddings were found
        # if not embedding_cache:
        #     print("Error: No embeddings available for evaluation")
        #     return
        
        #load the cached test emebeddings data
        with open(r'cache\test_embeddings.pkl', 'rb') as f:
            embedding_cache = pickle.load(f)

        embedding_cache = [i[1:] for i in embedding_cache]
        embedding_cache= dict(embedding_cache)
        # Step 3: Evaluate for all thresholds
        results = []
        for threshold in thresholds:
            metrics = evaluate_with_threshold(embedding_cache, threshold)
            results.append((threshold, metrics))
            print(f"\nThreshold {threshold:.1f} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            print(f"TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']} | TN: {metrics['TN']}")
        
        # Step 4: Plot results
        plot_results(results)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up GPU resources
        if USE_GPU:
            tf.keras.backend.clear_session()
        elapsed_time = time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()