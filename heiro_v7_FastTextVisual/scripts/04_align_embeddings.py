import logging
import json
import numpy as np
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
FUSED_MODEL_PATH = BASE_DIR / "models/fused_embeddings_1068d.kv"
GLOVE_PATH = BASE_DIR.parent / "heiro_v5_getdata/data/processed/glove.6B.300d.txt"
ANCHORS_PATH = BASE_DIR.parent / "heiro_v6_BERT/data/processed/anchors.json"
RESULTS_PATH = BASE_DIR / "data/processed/alignment_results_v7.json"

def main():
    # 1. Load Models
    print(f"Loading Fused Model from {FUSED_MODEL_PATH}...")
    hiero_kv = KeyedVectors.load(str(FUSED_MODEL_PATH))
    
    print(f"Loading GloVe from {GLOVE_PATH}...")
    # Load GloVe into KeyedVectors (might take a while, but standard way)
    # Or load manually to dictionary for speed if we only need anchors?
    # But we need full vocab for nearest neighbor search during evaluation.
    # So we must load full GloVe.
    glove_kv = KeyedVectors.load_word2vec_format(str(GLOVE_PATH), binary=False, no_header=True)
    
    # 2. Load Anchors
    print(f"Loading Anchors from {ANCHORS_PATH}...")
    with open(ANCHORS_PATH, 'r') as f:
        anchors = json.load(f)
        
    # 3. Prepare Data
    print("Preparing alignment data...")
    X = []
    Y = []
    valid_anchors = []
    
    for anchor in anchors:
        h_word = anchor['hieroglyphic']
        e_word = anchor['english']
        
        # Normalize English word? GloVe is lowercase.
        e_word = e_word.lower()
        
        # Check if words exist
        if h_word in hiero_kv and e_word in glove_kv:
            X.append(hiero_kv[h_word])
            Y.append(glove_kv[e_word])
            valid_anchors.append((h_word, e_word))
            
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Valid Anchors: {len(X)} / {len(anchors)}")
    
    # 4. Split Data
    X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
        X, Y, valid_anchors, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 5. Train Alignment (Linear Regression / Ridge)
    print("Training Linear Alignment...")
    # Ridge regression is better for high-dim source
    aligner = Ridge(alpha=1.0)
    aligner.fit(X_train, Y_train)
    
    print(f"R^2 Score on Train: {aligner.score(X_train, Y_train):.4f}")
    print(f"R^2 Score on Test: {aligner.score(X_test, Y_test):.4f}")
    
    # 6. Evaluate
    print("Evaluating on Test Set...")
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = len(X_test)
    
    # Predict all test vectors
    Y_pred = aligner.predict(X_test)
    
    for i in tqdm(range(total)):
        pred_vec = Y_pred[i]
        true_word = anchors_test[i][1]
        
        # Find nearest neighbors in GloVe
        # most_similar returns [(word, score), ...]
        neighbors = glove_kv.similar_by_vector(pred_vec, topn=10)
        neighbor_words = [w for w, s in neighbors]
        
        if true_word == neighbor_words[0]:
            correct_top1 += 1
        if true_word in neighbor_words[:5]:
            correct_top5 += 1
        if true_word in neighbor_words[:10]:
            correct_top10 += 1
            
    acc_top1 = correct_top1 / total * 100
    acc_top5 = correct_top5 / total * 100
    acc_top10 = correct_top10 / total * 100
    
    print(f"\nResults (Test Set N={total}):")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-5 Accuracy: {acc_top5:.2f}%")
    print(f"Top-10 Accuracy: {acc_top10:.2f}%")
    
    # Save results
    results = {
        "model": "V7 FastText + Visuals (Fused 1068d -> 300d)",
        "test_samples": total,
        "top1_accuracy": acc_top1,
        "top5_accuracy": acc_top5,
        "top10_accuracy": acc_top10,
        "r2_train": aligner.score(X_train, Y_train),
        "r2_test": aligner.score(X_test, Y_test)
    }
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
