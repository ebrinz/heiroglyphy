"""
Script to execute v11 Phase 3: Advanced Training (MLP)
"""
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def run_training():
    # Paths
    PROJECT_ROOT = Path("heiro_v11")
    REPO_ROOT = Path(".")
    
    V11_DATA = PROJECT_ROOT / 'data'
    GLOVE_PATH = REPO_ROOT / 'heiro_v5_getdata/data/processed/glove.6B.300d.txt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # 1. Load Data
    print('Loading fused embeddings...')
    with open(V11_DATA / 'processed/fused_embeddings_v11.pkl', 'rb') as f:
        fused_embeddings = pickle.load(f)
    print(f'✓ Loaded {len(fused_embeddings)} fused embeddings')
    
    print('Loading anchors...')
    with open(V11_DATA / 'processed/v11_anchors.json', 'r') as f:
        anchors = json.load(f)
    print(f'✓ Loaded {len(anchors)} anchors')
    
    print('Loading GloVe...')
    english_embeddings = KeyedVectors.load_word2vec_format(
        str(GLOVE_PATH), binary=False, no_header=True
    )
    print(f'✓ Loaded {len(english_embeddings)} English embeddings')
    
    # 2. Prepare Dataset
    X_data = []
    Y_data = []
    valid_pairs = []
    
    for anchor in anchors:
        egy = anchor['hieroglyphic']
        eng = anchor['english'].lower()
        
        if egy in fused_embeddings and eng in english_embeddings:
            X_data.append(fused_embeddings[egy])
            Y_data.append(english_embeddings[eng])
            valid_pairs.append((egy, eng))
    
    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    
    print(f'Valid pairs: {len(X_data)} / {len(anchors)}')
    
    X_train, X_test, Y_train, Y_test, pairs_train, pairs_test = train_test_split(
        X_data, Y_data, valid_pairs, test_size=0.2, random_state=42
    )
    
    class HieroglyphDataset(Dataset):
        def __init__(self, X, Y):
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    
    train_loader = DataLoader(HieroglyphDataset(X_train, Y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(HieroglyphDataset(X_test, Y_test), batch_size=32, shuffle=False)
    
    # 3. Define Model
    class AlignmentMLP(nn.Module):
        def __init__(self, input_dim=1536, hidden_dim=512, output_dim=300):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, x):
            return self.net(x)
    
    model = AlignmentMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 4. Train
    epochs = 50
    print('Starting training...')
    
    for epoch in range(epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} complete')
            
    # 5. Evaluate
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    
    print('Evaluating accuracy...')
    
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(device)
        Y_pred = model(X_test_tensor).cpu().numpy()
        
        for i in tqdm(range(len(Y_pred))):
            neighbors = english_embeddings.similar_by_vector(Y_pred[i], topn=10)
            neighbor_words = [w for w, s in neighbors]
            true_word = pairs_test[i][1]
            
            if true_word == neighbor_words[0]: correct_top1 += 1
            if true_word in neighbor_words[:5]: correct_top5 += 1
            if true_word in neighbor_words[:10]: correct_top10 += 1
    
    n = len(X_test)
    acc_top1 = correct_top1/n*100
    acc_top5 = correct_top5/n*100
    acc_top10 = correct_top10/n*100
    
    print(f'\nResults (N={n}):')
    print(f'Top-1: {acc_top1:.2f}%')
    print(f'Top-5: {acc_top5:.2f}%')
    print(f'Top-10: {acc_top10:.2f}%')
    
    # Save Results
    results = {
        'version': 'v11',
        'technique': 'MLP + Clean Data + N-Grams',
        'anchors': len(X_data),
        'top1': acc_top1,
        'top5': acc_top5,
        'top10': acc_top10
    }
    
    with open(V11_DATA / 'processed/results_v11.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    torch.save(model.state_dict(), V11_DATA / 'processed/mlp_model_v11.pth')
    print('✓ Saved model and results')

if __name__ == "__main__":
    run_training()
