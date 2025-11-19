#!/usr/bin/env python3
"""
Follow-up analysis: Compare TLA hieroglyphic embeddings and their TLA German→English translations in the hieroglyphic embedding space, versus the same English translations in the sentence-transformer (Wikipedia) space.

This script computes and reports the alignment between:
- Each TLA item in hieroglyphic space
- Its English translation in hieroglyphic space (via FastText)
- The same English translation in the Wikipedia/sentence-transformer space

It measures how well the semantic alignment is preserved across spaces.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
import torch

from heiro import Config, EmbeddingManager, GermanTranslationProcessor
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alignment")

def main():
    config = Config()
    embedding_manager = EmbeddingManager(config)

    # Load TLA parallel data
    german_processor = GermanTranslationProcessor(config)
    parallel_data = german_processor.process_tla_translations()

    # Load hieroglyphic embeddings (FastText)
    hiero = embedding_manager.load_embeddings('hieroglyphic')
    hiero_embs = hiero['embeddings']
    hiero_texts = hiero['texts']

    # Load TLA English embeddings (in hieroglyphic space)
    tla_english = embedding_manager.load_embeddings('tla_english')
    tla_english_embs = tla_english['embeddings']
    tla_english_texts = tla_english['texts']

    # Load Wikipedia English embeddings (sentence-transformer space)
    wikipedia = embedding_manager.load_embeddings('wikipedia')
    wikipedia_embs = wikipedia['embeddings']
    wikipedia_texts = wikipedia['texts']

    # Prepare a sentence-transformer model for fresh English embeddings
    st_model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    # Load the trained vec2vec model
    import torch
    from heiro import MultiSpaceVec2VecModel
    model_path = config.MODELS_DIR / "multi_space_vec2vec_model.pt"
    device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiSpaceVec2VecModel(
        checkpoint['input_dims'],
        checkpoint['latent_dim'],
        checkpoint['num_spaces']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Computing cross-space alignment for TLA items using vec2vec model...")

    similarities_latent = []
    all_pred_hiero = []
    all_actual_hiero = []
    all_indices = []
    for idx, item in enumerate(parallel_data):
        if idx >= len(hiero_embs) or idx >= len(tla_english_embs):
            break
        # Hieroglyphic embedding (FastText space)
        hiero_emb = torch.FloatTensor(hiero_embs[idx:idx+1]).to(device)
        # TLA English embedding (sentence-transformer space)
        tla_eng_emb = torch.FloatTensor(tla_english_embs[idx:idx+1]).to(device)
        with torch.no_grad():
            # Encode TLA English to latent, decode to hieroglyphic space
            tla_latent = model.encoders[1](tla_eng_emb)
            pred_hiero_emb = model.decoders[2](tla_latent)
            # Cosine similarity between predicted hieroglyphic and actual
            sim_latent = cosine_similarity(pred_hiero_emb.cpu().numpy(), hiero_emb.cpu().numpy())[0][0]
            similarities_latent.append(sim_latent)
            all_pred_hiero.append(pred_hiero_emb.cpu().numpy()[0])
            all_actual_hiero.append(hiero_emb.cpu().numpy()[0])
            all_indices.append(idx)
        if idx < 5:
            logger.info(f"Sample {idx}")
            logger.info(f"Hieroglyphic: {item['hieroglyphic']}")
            logger.info(f"English: {item['english']}")
            logger.info(f"Vec2vec-pred Hiero↔Actual Hiero similarity: {sim_latent:.3f}")
    logger.info(f"\nAverage Vec2vec-pred Hieroglyphic↔Actual Hieroglyphic similarity: {np.mean(similarities_latent):.4f}")
    logger.info(f"{len(similarities_latent)} items analyzed.")

    # Log the 10 worst-aligned pairs
    logger.info("\nWorst-aligned pairs (lowest similarity):")
    sim_arr = np.array(similarities_latent)
    worst_idxs = np.argsort(sim_arr)[:10]
    for rank, widx in enumerate(worst_idxs):
        item = parallel_data[all_indices[widx]]
        logger.info(f"#{rank+1} | Similarity: {sim_arr[widx]:.3f}")
        logger.info(f"  Hieroglyphic: {item['hieroglyphic']}")
        logger.info(f"  English: {item['english']}")

    # Visualization: 2D scatter plot (PCA)
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    all_pred_hiero = np.stack(all_pred_hiero)
    all_actual_hiero = np.stack(all_actual_hiero)
    # Concatenate for joint PCA
    all_concat = np.concatenate([all_pred_hiero, all_actual_hiero], axis=0)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_concat)
    pred_2d = all_2d[:len(all_pred_hiero)]
    actual_2d = all_2d[len(all_pred_hiero):]
    # Color by similarity
    sim_norm = (sim_arr - sim_arr.min()) / (np.ptp(sim_arr) + 1e-6)
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_2d[:, 0], actual_2d[:, 1], c=sim_norm, cmap='Reds', label='Actual Hieroglyphic', alpha=0.5, s=10)
    plt.scatter(pred_2d[:, 0], pred_2d[:, 1], c=sim_norm, cmap='Blues', label='Predicted (from English)', alpha=0.5, s=10)
    plt.colorbar(label='Vec2vec Alignment (cosine sim)')
    plt.legend()
    plt.title('Hieroglyphic Embedding Alignment (Actual vs. Predicted)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig('alignment_scatter.png', dpi=150)
    logger.info("Saved 2D alignment scatter plot as 'alignment_scatter.png'.")

if __name__ == "__main__":
    main()
