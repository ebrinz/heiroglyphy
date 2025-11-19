import os
import pickle
import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "heiro_v3/data"
MODELS_DIR = "heiro_v3/models"
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

print(f"Filtered to {len(valid_anchors)} valid anchors.")
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

X_train, X_test, Y_train, Y_test, anchors_train, anchors_test = train_test_split(
    X, Y, valid_anchors, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} pairs, Testing on {len(X_test)} pairs.")

def learn_rotation(X, Y):
    # SVD of Y.T @ X
    U, S, Vt = np.linalg.svd(Y.T @ X)
    # R = U @ Vt
    R = U @ Vt
    return R

print("Learning rotation matrix...")
R = learn_rotation(X_train, Y_train)
print("Done.")

def evaluate(X, anchors, R, top_k=10):
    hits = 0
    total = len(anchors)
    
    # Project all X to English space
    X_projected = X @ R.T
    
    for i, (h_word, true_e_word) in enumerate(anchors):
        pred_vec = X_projected[i]
        
        # Find nearest neighbors in English model
        similar = eng_model.wv.most_similar([pred_vec], topn=top_k)
        candidates = [w for w, score in similar]
        
        if true_e_word in candidates:
            hits += 1
            
    accuracy = hits / total
    return accuracy

print("Evaluating on Test Set...")
acc_1 = evaluate(X_test, anchors_test, R, top_k=1)
acc_5 = evaluate(X_test, anchors_test, R, top_k=5)
acc_10 = evaluate(X_test, anchors_test, R, top_k=10)

print(f"Top-1 Accuracy: {acc_1:.2%}")
print(f"Top-5 Accuracy: {acc_5:.2%}")
print(f"Top-10 Accuracy: {acc_10:.2%}")

def translate(h_word, top_k=5):
    # Get vector (FastText handles OOV)
    vec = hier_model.wv[h_word]
    # Project
    proj_vec = vec @ R.T
    # Find neighbors
    similar = eng_model.wv.most_similar([proj_vec], topn=top_k)
    
    print(f"\nTranslation for '{h_word}':")
    for w, score in similar:
        print(f"  -> {w} ({score:.3f})")

# Famous words
translate("nfr")       # Good/Beautiful
translate("pr-aa")     # Pharaoh (Great House)
translate("ankh")      # Life
translate("maat")      # Truth/Order
translate("ra")        # Sun God
translate("suten")     # King (nswt)
translate("netjer")    # God (nTr)
