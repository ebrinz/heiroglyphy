import logging
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from tqdm import tqdm

from heiro import (
    Config,
    EmbeddingManager,
    GermanTranslationProcessor,
    EgyptologyDataCollector,
    MultilingualEmbeddingSpace,
    HieroglyphicDataProcessor,
    HieroglyphicEmbeddingSpace,
    MultiSpaceVec2VecTrainer,
    MultiSpaceVec2VecModel,
    logger
)

class FollowUpTaskManager:
    """Enable various follow-up tasks using the trained embeddings and models.
    NOTE: All code that was previously at the class level has been moved into the setup() method to prevent execution at import time.
    """
    
    def __init__(self, config):
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        # Device selection
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon GPU) for follow-up tasks")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Using CUDA GPU for follow-up tasks")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for follow-up tasks")

    def setup(self):
        """Setup method to prepare all follow-up task data. Call explicitly when needed."""
        self.german_processor = GermanTranslationProcessor(self.config)
        self.parallel_data = self.german_processor.process_tla_translations()
        self.hieroglyphic_texts = [item['hieroglyphic'] for item in self.parallel_data]
        self.german_texts = [item['german'] for item in self.parallel_data]
        self.english_translated_texts = [item['english'] for item in self.parallel_data]
        self.english_collector = EgyptologyDataCollector(self.config)
        self.wikipedia_texts = self.english_collector.prepare_english_corpus()
        self.wikipedia_text_list = [item['text'] for item in self.wikipedia_texts]
        self.embedder = MultilingualEmbeddingSpace(self.config)
        self.wikipedia_embeddings = self.embedder.create_embeddings(self.wikipedia_text_list, language="en")
        self.tla_english_embeddings = self.embedder.create_embeddings(self.english_translated_texts, language="en")
        self.german_embeddings = self.embedder.create_embeddings(self.german_texts, language="de")
        self.hieroglyphic_processor = HieroglyphicDataProcessor(self.config)
        self.hieroglyphic_corpus_file = self.hieroglyphic_processor.prepare_hieroglyphic_corpus()
        self.hieroglyphic_embedder = HieroglyphicEmbeddingSpace(self.config)
        self.hieroglyphic_embeddings = self.hieroglyphic_embedder.create_embeddings(self.hieroglyphic_texts)
        self.min_samples = min(
            len(self.wikipedia_embeddings),
            len(self.tla_english_embeddings), 
            len(self.hieroglyphic_embeddings)
        )
        self.wikipedia_embeddings = self.wikipedia_embeddings[:self.min_samples]
        self.tla_english_embeddings = self.tla_english_embeddings[:self.min_samples]
        self.hieroglyphic_embeddings = self.hieroglyphic_embeddings[:self.min_samples]
        self.vec2vec_trainer = MultiSpaceVec2VecTrainer(self.config)
        self.vec2vec_model = self.vec2vec_trainer.train(
            self.wikipedia_embeddings,      # Space 0: Wikipedia English
            self.tla_english_embeddings,    # Space 1: TLA German→English (ground truth)
            self.hieroglyphic_embeddings    # Space 2: Hieroglyphic FastText
        )
    
    def load_vec2vec_model(self):
        """Load trained vec2vec model"""
        model_path = self.config.MODELS_DIR / "multi_space_vec2vec_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"vec2vec model not found: {model_path}")
        
        # Load checkpoint with proper device mapping for MPS
        if self.device.type == 'mps':
            checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU first for MPS
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
        
        model = MultiSpaceVec2VecModel(
            checkpoint['input_dims'],
            checkpoint['latent_dim'],
            checkpoint['num_spaces']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint
    
    def semantic_search(self, query_text, target_space="hieroglyphic", top_k=10):
        """Semantic search across embedding spaces"""
        logger.info(f"Performing semantic search: '{query_text}' in {target_space} space")
        
        # Load model and embeddings
        model, checkpoint = self.load_vec2vec_model()
        
        # Load target embeddings
        target_data = self.embedding_manager.load_embeddings(target_space)
        target_embeddings = target_data['embeddings']
        target_texts = target_data['texts']
        
        # Encode query (assuming it's English)
        embedder = MultilingualEmbeddingSpace(self.config)
        query_embedding = embedder.create_embeddings([query_text], language="en")
        
        # If target is hieroglyphic, translate query through vec2vec
        if target_space == "hieroglyphic":
            query_tensor = torch.FloatTensor(query_embedding)
            # ... (rest of method omitted for brevity)
        # ... (rest of method omitted for brevity)

    def cross_space_similarity(self, text1, space1, text2, space2):
        """Calculate similarity between texts in different embedding spaces"""
        logger.info(f"Calculating cross-space similarity: {space1} ↔ {space2}")
        # ... (rest of method omitted for brevity)

    def concept_exploration(self, concept_keywords, num_samples=20):
        """Explore how concepts are represented across all embedding spaces"""
        logger.info(f"Exploring concept: {concept_keywords}")
        # ... (rest of method omitted for brevity)

    def translation_quality_analysis(self):
        """Analyze translation quality between spaces"""
        logger.info("Analyzing translation quality...")
        # ... (rest of method omitted for brevity)

    def export_embeddings_for_analysis(self, format='csv'):
        """Export embeddings in various formats for external analysis"""
        logger.info(f"Exporting embeddings in {format} format...")
        # ... (rest of method omitted for brevity)

def demonstrate_follow_up_tasks(config):
    """Demonstrate various follow-up tasks"""
    task_manager = FollowUpTaskManager(config)
    
    logger.info("Available embeddings:")
    available = task_manager.embedding_manager.list_available_embeddings()
    for emb in available:
        logger.info(f"  - {emb['name']}: {emb['shape']} ({emb['timestamp']})")
    
    # Example 1: Semantic search
    logger.info("\n--- Example 1: Semantic Search ---")
    try:
        results = task_manager.semantic_search("pharaoh temple", "hieroglyphic", top_k=5)
        logger.info("Top 5 hieroglyphic texts similar to 'pharaoh temple':")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. [{result['similarity']:.3f}] {result['text'][:100]}...")
    except Exception as e:
        logger.error(f"Semantic search example failed: {e}")
    
    # Example 2: Concept exploration
    logger.info("\n--- Example 2: Concept Exploration ---")
    try:
        concept_results = task_manager.concept_exploration(["god", "temple", "priest"], num_samples=3)
        for space, results in concept_results.items():
            logger.info(f"\n{space.upper()} space:")
            for result in results[:3]:
                logger.info(f"  [{result['similarity']:.3f}] {result['text'][:80]}...")
    except Exception as e:
        logger.error(f"Concept exploration example failed: {e}")
    
    # Example 3: Translation quality analysis
    logger.info("\n--- Example 3: Translation Quality Analysis ---")
    try:
        quality_results = task_manager.translation_quality_analysis()
        logger.info("Best aligned translations (TLA English ↔ Hieroglyphic):")
        for result in quality_results[:5]:
            logger.info(f"  Similarity: {result['latent_similarity']:.3f}")
            logger.info(f"  English: {result['tla_text'][:60]}...")
            logger.info(f"  Hieroglyphic: {result['hieroglyphic_text']}")
            logger.info()
    except Exception as e:
        logger.error(f"Translation quality analysis failed: {e}")
    
    # Example 4: Export embeddings
    logger.info("\n--- Example 4: Exporting Embeddings ---")
    try:
        export_dir = task_manager.export_embeddings_for_analysis('csv')
        logger.info(f"Embeddings exported to: {export_dir}")
        
        # List exported files
        exported_files = list(export_dir.glob("*.csv"))
        logger.info(f"Exported {len(exported_files)} files:")
        for file in exported_files:
            logger.info(f"  - {file.name}")
    except Exception as e:
        logger.error(f"Export example failed: {e}")

def run_follow_up_tasks_only():
    """Entry point for running only follow-up tasks with existing embeddings"""
    config = Config()
    task_manager = FollowUpTaskManager(config)
    
    logger.info("=== Running Follow-up Tasks Only ===")
    
    # Check available embeddings
    available = task_manager.embedding_manager.list_available_embeddings()
    if len(available) < 3:
        logger.error("Insufficient embeddings found. Please run main() first.")
        return
    
    demonstrate_follow_up_tasks(config)

if __name__ == "__main__":
    run_follow_up_tasks_only()
