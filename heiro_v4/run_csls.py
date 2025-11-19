import os
import pickle
import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "heiro_v4/data"
MODELS_DIR = "heiro_v4/models"
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors.pkl")
HIEROGLYPHIC_MODEL_FILE = os.path.join(MODELS_DIR, "hieroglyphic_fasttext.model")
ENGLISH_MODEL_FILE = os.path.join(MODELS_DIR, "english_word2vec.model")

print("Loading models...")
hier_model = FastText.load(HIEROGLYPHIC_MODEL_FILE)
eng_model = Word2Vec.load(ENGLISH_MODEL_FILE)

print("Loading anchors...")
with open(ANCHOR_FILE, 'rb') as f:
    anchors = pickle.load(f)
    
print(f"Loaded {len(anchors)} anchors.")

# Prepare matrices
valid_anchors = []
X_list = []
Y_list = []

for anchor in anchors:
    h_word = anchor['hieroglyphic']
    e_word = anchor['english']
    if e_word in eng_model.wv:
        valid_anchors.append((h_word, e_word))
        X_list.append(hier_model.wv[h_word])
        Y_list.append(eng_model.wv[e_word])

X = np.array(X_list)
Y = np.array(Y_list)

# Split
X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
    X, Y, valid_anchors, test_size=0.2, random_state=42
)

# SVD for Rotation
U, S, Vt = np.linalg.svd(Y_train.T @ X_train)
R = U @ Vt

print("Rotation matrix R calculated.")

def get_csls_scores(source_vecs, target_vecs, k=10):
    """
    Compute CSLS scores between source and target vectors.
    """
    # Normalize vectors for cosine similarity
    source_norm = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)
    target_norm = target_vecs / np.linalg.norm(target_vecs, axis=1, keepdims=True)
    
    # Compute Cosine Similarity Matrix (N x M)
    sim_matrix = np.dot(source_norm, target_norm.T)
    
    # Calculate r_T (average sim to k nearest neighbors in target for each source)
    r_T = np.mean(np.sort(sim_matrix, axis=1)[:, -k:], axis=1)
    
    # Calculate r_S (average sim to k nearest neighbors in source for each target)
    r_S = np.mean(np.sort(sim_matrix, axis=0)[-k:, :], axis=0)
    
    # CSLS = 2*cos - r_T - r_S
    csls_scores = 2 * sim_matrix - r_T[:, np.newaxis] - r_S[np.newaxis, :]
    
    return csls_scores

# Prepare Target Space (All English Words)
target_words = eng_model.wv.index_to_key[:20000]
target_vecs = np.array([eng_model.wv[w] for w in target_words])

print(f"Target space prepared: {len(target_words)} words.")

def evaluate_csls(X_test, anchors_test, R, top_k=10):
    # Project Test Vectors
    X_projected = X_test @ R.T
    
    # Compute CSLS Matrix
    scores = get_csls_scores(X_projected, target_vecs)
    
    hits = 0
    total = len(anchors_test)
    
    for i, (h_word, true_e_word) in enumerate(anchors_test):
        top_indices = np.argsort(scores[i])[-top_k:][::-1]
        candidates = [target_words[idx] for idx in top_indices]
        
        if true_e_word in candidates:
            hits += 1
            
    return hits / total

print("Evaluating CSLS...")
csls_acc_1 = evaluate_csls(X_test, anchors_test, R, top_k=1)
csls_acc_10 = evaluate_csls(X_test, anchors_test, R, top_k=10)

print(f"CSLS Top-1 Accuracy: {csls_acc_1:.2%}")
print(f"CSLS Top-10 Accuracy: {csls_acc_10:.2%}")

def translate_csls(h_word, top_k=5):
    if h_word not in hier_model.wv:
        print(f"'{h_word}' not in vocab.")
        return
        
    vec = hier_model.wv[h_word]
    proj_vec = (vec @ R.T).reshape(1, -1)
    
    scores = get_csls_scores(proj_vec, target_vecs)
    top_indices = np.argsort(scores[0])[-top_k:][::-1]
    
    print(f"\nCSLS Translation for '{h_word}':")
    for idx in top_indices:
        print(f"  -> {target_words[idx]} ({scores[0][idx]:.3f})")

translate_csls("nfr")
translate_csls("pr-aa")
translate_csls("ra")
