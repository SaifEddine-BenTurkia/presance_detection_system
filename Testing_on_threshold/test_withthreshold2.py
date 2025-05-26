# import os
# import numpy as np
# from elasticsearch import Elasticsearch
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from time import time
# import pickle
# import tensorflow as tf
# import seaborn as sns

# # --- Configurations ---
# es = Elasticsearch("http://127.0.0.1:9200")
# index_name = "face_embeddings_index "
# thresholds = np.arange(0.8, 1.0, 0.03)
# CACHE_PATH = r'cache\test_embeddings.pkl'
# BATCH_SIZE = 32

# # Check for GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
#     tf.config.set_visible_devices(gpus[0], 'GPU')
#     USE_GPU = True
# else:
#     USE_GPU = False

# # --- Elasticsearch KNN query ---
# def search_es_for_similarities(query_embedding):
#     try:
#         response = es.search(
#             index=index_name,
#             knn={
#                 "field": "embedding",
#                 "query_vector": query_embedding.tolist(),
#                 "k": 5,
#                 "num_candidates": 100,
#                 "similarity":"cosine"
#             }
#         )
#         return response['hits']['hits']
#     except Exception:
#         return []

# # --- Evaluation Functions ---
# def evaluate_batch(batch_items, threshold):
#     y_true, y_pred,scores  = [], [],[]
#     for image_path, (person_name, embedding) in batch_items:
#         hits = search_es_for_similarities(embedding)
#         score = hits[0]['_score'] if hits else 0
#         recog = hits[0]['_source']['person_name'] if hits and hits[0]['_score'] >= threshold else "Unknown"
#         y_true.append(person_name)
#         y_pred.append(recog)
#         scores.append(score)
#     return y_true, y_pred,scores

# def evaluate_with_threshold(embedding_cache, threshold):
#     items = list(embedding_cache.items())
#     batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
#     y_true_all, y_pred_all, all_scores = [], [], []
#     for batch in tqdm(batches, desc=f"Evaluating @ thr={threshold:.2f}", unit="batch"):
#         y_t, y_p, scores = evaluate_batch(batch, threshold)
#         y_true_all.extend(y_t)
#         y_pred_all.extend(y_p)
#         all_scores.extend(scores)
#     metrics = compute_metrics(y_true_all, y_pred_all)
#     metrics['scores'] = all_scores  # Add scores to the metrics dict
#     return metrics


# def compute_metrics(y_true, y_pred):
#     if not y_true:
#         return dict.fromkeys(('accuracy','precision','recall','f1','TP','FP','FN','TN','sensitivity'), 0)
#     labels = sorted(set(y_true + y_pred))
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     TP = np.diag(cm)
#     FP = cm.sum(axis=0) - TP
#     FN = cm.sum(axis=1) - TP
#     TN = cm.sum() - (TP + FP + FN)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         sens = np.where((TP+FN)>0, TP/(TP+FN), 0)
#     return {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
#         'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
#         'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
#         'TP': TP.sum(), 'FP': FP.sum(), 'FN': FN.sum(), 'TN': TN.sum(),
#         'sensitivity': sens.mean()
#     }

# def plot_score_histogram(score_list, threshold):
#     plt.figure(figsize=(8,4))
#     sns.histplot(score_list, bins=20, kde=True)
#     plt.title(f"Score Distribution @ Threshold={threshold:.2f}")
#     plt.xlabel("ES _score"); plt.ylabel("Frequency")
#     plt.grid(True); plt.tight_layout()
#     plt.show()

# # --- Plotting ---
# def plot_results(results):
#     ths = [r[0] for r in results]
#     mets = [r[1] for r in results]
#     plt.figure(figsize=(12,6))
#     plt.plot(ths, [m['accuracy'] for m in mets],    marker='o', label='Accuracy')
#     plt.plot(ths, [m['precision'] for m in mets],   marker='s', label='Precision')
#     plt.plot(ths, [m['recall'] for m in mets],      marker='^', label='Recall')
#     plt.plot(ths, [m['f1'] for m in mets],          marker='x', label='F1')
#     plt.xlabel("Threshold"); plt.ylabel("Score")
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.show()

# # --- Main Execution ---
# def main():
#     start = time()
#     # Load cache
#     with open(CACHE_PATH, 'rb') as f:
#         raw = pickle.load(f)  # [(name, path, emb_list), ...]
#     embedding_cache = {path: (name, np.array(emb)) for name, path, emb in raw}

#     # Sweep thresholds
#     results = []
#     for thr in tqdm(thresholds, desc="Threshold sweep", unit="thr", position = 0 ):
#         metrics = evaluate_with_threshold(embedding_cache, thr)
#         results.append((thr, metrics))
#         print(f"Thr={thr:.2f} → Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f},"
#               f" Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
#         plot_score_histogram(metrics['scores'], thr)

#     # Plot
#     plot_results(results)

#     print(f"Done in {time()-start:.1f}s")

# if __name__ == "__main__":
#     main()



import os
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import pickle
import tensorflow as tf
import seaborn as sns

# --- Configurations ---
# Use the same index name as used during embedding generation
es = Elasticsearch("http://127.0.0.1:9200")
index_name = "face_embeddings_cosine"
# Thresholds for cosine similarity scores
thresholds = np.arange(0.8, 1.0, 0.1)
# Load the test embeddings saved by the generation script
CACHE_PATH = os.path.join("cache", "test_embeddings.pkl")
BATCH_SIZE = 32

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    USE_GPU = True
else:
    USE_GPU = False

# --- Elasticsearch KNN query ---
def search_es_for_similarities(query_embedding):
    try:
        response = es.search(
            index=index_name,
            knn={
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": 5,
                "num_candidates": 100,
                "similarity": "cosine"
            }
        )
        return response['hits']['hits']
    except Exception:
        return []

# --- Evaluation Functions ---
def evaluate_batch(batch_items, threshold):
    y_true, y_pred, scores = [], [], []
    for image_path, (person_name, embedding) in batch_items:
        hits = search_es_for_similarities(embedding)
        score = hits[0]['_score'] if hits else 0
        recog = hits[0]['_source']['person_name'] if hits and hits[0]['_score'] >= threshold else "Unknown"
        y_true.append(person_name)
        y_pred.append(recog)
        scores.append(score)
    return y_true, y_pred, scores


def evaluate_with_threshold(embedding_cache, threshold):
    items = list(embedding_cache.items())
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    y_true_all, y_pred_all, all_scores = [], [], []
    for batch in tqdm(batches, desc=f"Evaluating @ thr={threshold:.2f}", unit="batch"):
        y_t, y_p, scores = evaluate_batch(batch, threshold)
        y_true_all.extend(y_t)
        y_pred_all.extend(y_p)
        all_scores.extend(scores)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics['scores'] = all_scores
    return metrics


def compute_metrics(y_true, y_pred):
    if not y_true:
        return dict.fromkeys(('accuracy','precision','recall','f1','TP','FP','FN','TN','sensitivity'), 0)
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.where((TP+FN)>0, TP/(TP+FN), 0)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'TP': TP.sum(),
        'FP': FP.sum(),
        'FN': FN.sum(),
        'TN': TN.sum(),
        'sensitivity': sens.mean()
    }


def plot_score_histogram(score_list, threshold):
    plt.figure(figsize=(8,4))
    sns.histplot(score_list, bins=20, kde=True)
    plt.title(f"Score Distribution @ Threshold={threshold:.2f}")
    plt.xlabel("ES _score"); plt.ylabel("Frequency")
    plt.grid(True); plt.tight_layout()
    plt.show()


def plot_results(results):
    ths = [r[0] for r in results]
    mets = [r[1] for r in results]
    plt.figure(figsize=(12,6))
    plt.plot(ths, [m['accuracy'] for m in mets],    marker='o', label='Accuracy')
    plt.plot(ths, [m['precision'] for m in mets],   marker='s', label='Precision')
    plt.plot(ths, [m['recall'] for m in mets],      marker='^', label='Recall')
    plt.plot(ths, [m['f1'] for m in mets],          marker='x', label='F1')
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()


def main():
    start = time()
    # Load test embeddings
    with open(CACHE_PATH, 'rb') as f:
        raw = pickle.load(f)  # [(name, path, emb), ...]
    # Build cache dict: path -> (name, emb=array)
    embedding_cache = {path: (name, np.array(emb)) for name, path, emb in raw}

    results = []
    for thr in tqdm(thresholds, desc="Threshold sweep", unit="thr", position=0):
        metrics = evaluate_with_threshold(embedding_cache, thr)
        results.append((thr, metrics))
        print(f"Thr={thr:.2f} → Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f},"
              f" Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        plot_score_histogram(metrics['scores'], thr)

    plot_results(results)
    print(f"Done in {time()-start:.1f}s")

if __name__ == "__main__":
    main()
