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
from collections import defaultdict, Counter

# --- Configurations ---
es = Elasticsearch("http://127.0.0.1:9200")
index_name = "face_embeddings_cosine"

# FIXED: Proper threshold range - will be adjusted after analyzing score distribution
INITIAL_THRESHOLDS = np.linspace(0.1, 0.9, 17)  # Start with wide range

# Load the test embeddings saved by the generation script
CACHE_PATH = "cache/test_embeddings.pkl"
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

print(f"Using GPU: {USE_GPU}")

# --- Elasticsearch KNN query ---
def search_es_for_similarities(query_embedding, k=10):
    """Search for similar embeddings with increased k for better analysis"""
    try:
        # Since your index has similarity: "cosine" in mapping, we don't need to specify it in query
        # The similarity parameter in knn query is for overriding the index setting, not required
        response = es.search(
            index=index_name,
            knn={
                "field": "embedding",
                "query_vector": query_embedding.tolist(),
                "k": k,
                "num_candidates": 200
            }
        )
        return response['hits']['hits']
    except Exception as e:
        print(f"ES query error: {e}")
        return []

# --- NEW: Data Validation and Analysis ---
def validate_data(embedding_cache):
    """Validate and analyze the test data"""
    print("=== DATA VALIDATION ===")
    print(f"Total test samples: {len(embedding_cache)}")
    
    # Check for person distribution
    person_counts = Counter()
    for _, (person_name, _) in embedding_cache.items():
        person_counts[person_name] += 1
    
    print(f"Number of unique people: {len(person_counts)}")
    print(f"Samples per person: min={min(person_counts.values())}, max={max(person_counts.values())}, avg={np.mean(list(person_counts.values())):.1f}")
    
    # Show distribution
    if len(person_counts) <= 20:
        print("Person distribution:")
        for person, count in person_counts.most_common():
            print(f"  {person}: {count} samples")
    
    # Check embedding dimensions and properties
    sample_embedding = next(iter(embedding_cache.values()))[1]
    print(f"Embedding dimension: {sample_embedding.shape}")
    print(f"Embedding range: [{sample_embedding.min():.3f}, {sample_embedding.max():.3f}]")
    print(f"Embedding norm: {np.linalg.norm(sample_embedding):.3f}")
    print()
    
    return person_counts

def analyze_score_distribution(embedding_cache, sample_size=100):
    """Analyze the actual ES score distribution to determine proper thresholds"""
    print("=== SCORE DISTRIBUTION ANALYSIS ===")
    
    # Sample some embeddings for analysis
    sample_items = list(embedding_cache.items())[:sample_size]
    
    all_scores = []
    correct_person_scores = []
    wrong_person_scores = []
    
    print("Analyzing score distribution (this may take a moment)...")
    
    for image_path, (person_name, embedding) in tqdm(sample_items, desc="Analyzing scores"):
        hits = search_es_for_similarities(embedding, k=10)
        
        if hits:
            # All scores
            all_scores.extend([hit['_score'] for hit in hits])
            
            # Separate correct vs wrong person scores
            for hit in hits:
                score = hit['_score']
                if hit['_source']['person_name'] == person_name:
                    correct_person_scores.append(score)
                else:
                    wrong_person_scores.append(score)
    
    if not all_scores:
        print("ERROR: No scores found! Check your Elasticsearch connection and index.")
        return None
    
    print(f"Total scores analyzed: {len(all_scores)}")
    print(f"Score range: [{min(all_scores):.3f}, {max(all_scores):.3f}]")
    print(f"Score mean: {np.mean(all_scores):.3f}, std: {np.std(all_scores):.3f}")
    print(f"Score percentiles: 25%={np.percentile(all_scores, 25):.3f}, 50%={np.percentile(all_scores, 50):.3f}, 75%={np.percentile(all_scores, 75):.3f}")
    
    if correct_person_scores and wrong_person_scores:
        print(f"Correct person scores - mean: {np.mean(correct_person_scores):.3f}, std: {np.std(correct_person_scores):.3f}")
        print(f"Wrong person scores - mean: {np.mean(wrong_person_scores):.3f}, std: {np.std(wrong_person_scores):.3f}")
    
    # Plot distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.title("All ES Scores Distribution")
    plt.xlabel("ES Score")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    if correct_person_scores and wrong_person_scores:
        plt.subplot(1, 3, 2)
        plt.hist(correct_person_scores, bins=30, alpha=0.7, label='Correct Person', color='green')
        plt.hist(wrong_person_scores, bins=30, alpha=0.7, label='Wrong Person', color='red')
        plt.title("Score Distribution by Correctness")
        plt.xlabel("ES Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.boxplot([correct_person_scores, wrong_person_scores], labels=['Correct', 'Wrong'])
        plt.title("Score Distribution Boxplot")
        plt.ylabel("ES Score")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Suggest threshold range
    score_25th = np.percentile(all_scores, 25)
    score_75th = np.percentile(all_scores, 75)
    suggested_min = max(0.1, score_25th - 0.1)
    suggested_max = min(1.0, score_75th + 0.1)
    
    print(f"Suggested threshold range: [{suggested_min:.2f}, {suggested_max:.2f}]")
    return {
        'all_scores': all_scores,
        'correct_scores': correct_person_scores,
        'wrong_scores': wrong_person_scores,
        'suggested_range': (suggested_min, suggested_max)
    }

def manual_verification(embedding_cache, num_samples=5):
    """Manually verify a few examples to understand the matching behavior"""
    print("=== MANUAL VERIFICATION ===")
    sample_items = list(embedding_cache.items())[:num_samples]
    
    for i, (image_path, (person_name, embedding)) in enumerate(sample_items):
        print(f"\n--- Sample {i+1} ---")
        print(f"Query person: {person_name}")
        print(f"Image path: {os.path.basename(image_path)}")
        
        hits = search_es_for_similarities(embedding, k=5)
        
        if hits:
            print("Top 5 matches:")
            for j, hit in enumerate(hits):
                match_name = hit['_source']['person_name']
                score = hit['_score']
                is_correct = "✓" if match_name == person_name else "✗"
                print(f"  {j+1}. {match_name} (score: {score:.3f}) {is_correct}")
        else:
            print("No matches found!")

# --- IMPROVED: Better Evaluation Logic ---
def evaluate_batch_improved(batch_items, threshold):
    """Improved evaluation logic that properly tracks correct vs incorrect recognition"""
    y_true, y_pred, scores, detailed_results = [], [], [], []
    
    for image_path, (person_name, embedding) in batch_items:
        hits = search_es_for_similarities(embedding, k=10)
        
        if not hits:
            # No matches found
            y_true.append(person_name)
            y_pred.append("Unknown")
            scores.append(0.0)
            detailed_results.append({
                'query_person': person_name,
                'predicted': "Unknown", 
                'score': 0.0,
                'correct': False
            })
            continue
        
        # Find the best matching score for the correct person
        correct_person_best_score = 0
        for hit in hits:
            if hit['_source']['person_name'] == person_name:
                correct_person_best_score = max(correct_person_best_score, hit['_score'])
        
        # Get the overall best match
        best_match = hits[0]
        best_match_name = best_match['_source']['person_name']
        best_match_score = best_match['_score']
        
        # Decision logic
        if correct_person_best_score >= threshold:
            # Correct person found with sufficient score
            prediction = person_name
            final_score = correct_person_best_score
            is_correct = True
        elif best_match_score >= threshold:
            # Wrong person recognized with high confidence
            prediction = best_match_name
            final_score = best_match_score
            is_correct = False
        else:
            # No confident match
            prediction = "Unknown"
            final_score = best_match_score
            is_correct = False
        
        y_true.append(person_name)
        y_pred.append(prediction)
        scores.append(final_score)
        detailed_results.append({
            'query_person': person_name,
            'predicted': prediction,
            'score': final_score,
            'correct': is_correct,
            'correct_person_score': correct_person_best_score,
            'best_match_score': best_match_score
        })
    
    return y_true, y_pred, scores, detailed_results

def evaluate_with_threshold_improved(embedding_cache, threshold):
    """Improved threshold evaluation with detailed metrics"""
    items = list(embedding_cache.items())
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    
    y_true_all, y_pred_all, all_scores, all_details = [], [], [], []
    
    for batch in tqdm(batches, desc=f"Evaluating @ thr={threshold:.3f}", unit="batch"):
        y_t, y_p, scores, details = evaluate_batch_improved(batch, threshold)
        y_true_all.extend(y_t)
        y_pred_all.extend(y_p)
        all_scores.extend(scores)
        all_details.extend(details)
    
    metrics = compute_metrics_improved(y_true_all, y_pred_all, all_details)
    metrics['scores'] = all_scores
    metrics['details'] = all_details
    
    return metrics

def compute_metrics_improved(y_true, y_pred, details):
    """Improved metrics computation with more detailed analysis"""
    if not y_true:
        return dict.fromkeys(('accuracy', 'precision', 'recall', 'f1', 'recognition_rate', 
                            'false_acceptance_rate', 'false_rejection_rate'), 0)
    
    # Standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Face recognition specific metrics
    total_queries = len(details)
    correct_recognitions = sum(1 for d in details if d['correct'])
    unknown_predictions = sum(1 for pred in y_pred if pred == "Unknown")
    
    recognition_rate = correct_recognitions / total_queries if total_queries > 0 else 0
    
    # False acceptance: wrong person recognized with high confidence
    false_acceptances = sum(1 for d in details if not d['correct'] and d['predicted'] != "Unknown")
    false_acceptance_rate = false_acceptances / total_queries if total_queries > 0 else 0
    
    # False rejection: correct person not recognized (predicted as Unknown when they should be recognized)
    false_rejections = sum(1 for d in details if d['predicted'] == "Unknown" and d['correct_person_score'] > 0)
    false_rejection_rate = false_rejections / total_queries if total_queries > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'recognition_rate': recognition_rate,
        'false_acceptance_rate': false_acceptance_rate,
        'false_rejection_rate': false_rejection_rate,
        'total_queries': total_queries,
        'correct_recognitions': correct_recognitions,
        'unknown_predictions': unknown_predictions,
        'false_acceptances': false_acceptances,
        'false_rejections': false_rejections
    }

def plot_detailed_results(results):
    """Plot comprehensive results analysis"""
    if not results:
        print("No results to plot!")
        return
    
    thresholds = [r[0] for r in results]
    metrics = [r[1] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Standard metrics
    axes[0, 0].plot(thresholds, [m['accuracy'] for m in metrics], marker='o', label='Accuracy')
    axes[0, 0].plot(thresholds, [m['precision'] for m in metrics], marker='s', label='Precision')
    axes[0, 0].plot(thresholds, [m['recall'] for m in metrics], marker='^', label='Recall')
    axes[0, 0].plot(thresholds, [m['f1'] for m in metrics], marker='x', label='F1')
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Standard Classification Metrics")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Face recognition specific metrics
    axes[0, 1].plot(thresholds, [m['recognition_rate'] for m in metrics], marker='o', label='Recognition Rate', color='green')
    axes[0, 1].plot(thresholds, [m['false_acceptance_rate'] for m in metrics], marker='s', label='False Acceptance Rate', color='red')
    axes[0, 1].plot(thresholds, [m['false_rejection_rate'] for m in metrics], marker='^', label='False Rejection Rate', color='orange')
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("Face Recognition Metrics")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Count metrics
    axes[1, 0].plot(thresholds, [m['correct_recognitions'] for m in metrics], marker='o', label='Correct Recognitions', color='green')
    axes[1, 0].plot(thresholds, [m['false_acceptances'] for m in metrics], marker='s', label='False Acceptances', color='red')
    axes[1, 0].plot(thresholds, [m['false_rejections'] for m in metrics], marker='^', label='False Rejections', color='orange')
    axes[1, 0].plot(thresholds, [m['unknown_predictions'] for m in metrics], marker='x', label='Unknown Predictions', color='gray')
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Prediction Counts")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined optimization metric (you can adjust weights based on your needs)
    combined_scores = []
    for m in metrics:
        # Optimize for high recognition rate, low false acceptance
        combined = m['recognition_rate'] * 0.7 - m['false_acceptance_rate'] * 0.3
        combined_scores.append(combined)
    
    axes[1, 1].plot(thresholds, combined_scores, marker='o', color='purple')
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Combined Score")
    axes[1, 1].set_title("Combined Optimization Score\n(0.7*Recognition - 0.3*FalseAcceptance)")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Find and highlight optimal threshold
    optimal_idx = np.argmax(combined_scores)
    optimal_threshold = thresholds[optimal_idx]
    axes[1, 1].axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].text(optimal_threshold, max(combined_scores)*0.5, f'Optimal: {optimal_threshold:.3f}', 
                   rotation=90, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    # Print optimal threshold info
    optimal_metrics = metrics[optimal_idx]
    print(f"\n=== OPTIMAL THRESHOLD: {optimal_threshold:.3f} ===")
    print(f"Recognition Rate: {optimal_metrics['recognition_rate']:.3f}")
    print(f"False Acceptance Rate: {optimal_metrics['false_acceptance_rate']:.3f}")
    print(f"False Rejection Rate: {optimal_metrics['false_rejection_rate']:.3f}")
    print(f"F1 Score: {optimal_metrics['f1']:.3f}")
    print(f"Accuracy: {optimal_metrics['accuracy']:.3f}")

def test_elasticsearch_connection():
    """Test ES connection and get version info"""
    try:
        info = es.info()
        print(f"Elasticsearch version: {info['version']['number']}")
        
        # Test if index exists
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' exists")
            
            # Get index info
            mapping = es.indices.get_mapping(index=index_name)
            print("Index mapping found")
            
            # Test a simple count
            count = es.count(index=index_name)
            print(f"Total documents in index: {count['count']}")
            
            return True
        else:
            print(f"ERROR: Index '{index_name}' does not exist!")
            return False
            
    except Exception as e:
        print(f"ERROR: Cannot connect to Elasticsearch: {e}")
        return False

def main():
    start_time = time()
    
    print("Testing Elasticsearch connection...")
    if not test_elasticsearch_connection():
        return
    
    print("\nLoading test embeddings...")
    try:
        with open(CACHE_PATH, 'rb') as f:
            raw = pickle.load(f)  # [(name, path, emb), ...]
    except FileNotFoundError:
        print(f"ERROR: Could not find {CACHE_PATH}")
        print("Make sure you have generated test embeddings first.")
        return
    
    # Build cache dict: path -> (name, emb=array)
    embedding_cache = {path: (name, np.array(emb)) for name, path, emb in raw}
    
    print("Step 1: Data validation")
    person_counts = validate_data(embedding_cache)
    
    print("Step 2: Score distribution analysis")
    score_analysis = analyze_score_distribution(embedding_cache)
    
    if score_analysis is None:
        return
    
    print("Step 3: Manual verification")
    manual_verification(embedding_cache)
    
    # Determine thresholds based on analysis
    suggested_min, suggested_max = score_analysis['suggested_range']
    thresholds = np.linspace(suggested_min, suggested_max, 15)
    
    print(f"\nStep 4: Threshold evaluation with range [{suggested_min:.3f}, {suggested_max:.3f}]")
    print("This may take several minutes...")
    
    results = []
    for threshold in tqdm(thresholds, desc="Threshold sweep", unit="threshold"):
        metrics = evaluate_with_threshold_improved(embedding_cache, threshold)
        results.append((threshold, metrics))
        
        print(f"Threshold {threshold:.3f}: Rec={metrics['recognition_rate']:.3f}, "
              f"FAR={metrics['false_acceptance_rate']:.3f}, FRR={metrics['false_rejection_rate']:.3f}, "
              f"F1={metrics['f1']:.3f}")
    
    print("Step 5: Results visualization")
    plot_detailed_results(results)
    
    print(f"\nEvaluation completed in {time() - start_time:.1f} seconds")
    
    # Save results for further analysis
    results_path = os.path.join("cache", "evaluation_results.pkl")
    os.makedirs("cache", exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()