#!/usr/bin/env python3
"""
========
Egyptian Hieroglyphic-English Embedding Bridge using vec2vec
============================================================

This script creates three embedding spaces for comparison:
1. English space: Sentence-transformers trained on Wikipedia + Egyptology papers
2. German space: Sentence-transformers on TLA German translations
3. Hieroglyphic space: FastText trained on TLA hieroglyphic dataset
4. Uses vec2vec to bridge between all spaces and compare alignments

Requirements:
pip install datasets sentence-transformers fasttext torch transformers scikit-learn numpy pandas tqdm
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from tqdm import tqdm
import logging
import pickle


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_device_availability():
    """Check and log available compute devices"""
    logger.info("=== Device Availability ===") 
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("✅ MPS (Apple Silicon GPU) is available")
    else:
        logger.info("❌ MPS not available")
        
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("❌ CUDA not available")
        
    logger.info(f"📱 CPU cores available: {torch.get_num_threads()}")
    logger.info("=" * 30)

class Config:
    """Configuration parameters"""
    # Paths
    DATA_DIR = Path("./data")
    MODELS_DIR = Path("./models")
    CACHE_DIR = Path("./cache")
    
    # English embedding config
    SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
    WIKIPEDIA_SAMPLE_SIZE = 100000  # Adjust based on resources
    
    # German translation config
    GERMAN_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-de-en"
    
    # FastText config
    FASTTEXT_DIM = 300
    FASTTEXT_EPOCHS = 10
    FASTTEXT_MIN_COUNT = 2
    
    # vec2vec config
    LATENT_DIM = 256
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0002
    NUM_EPOCHS = 100
    
    def __init__(self):
        # Create directories
        for path in [self.DATA_DIR, self.MODELS_DIR, self.CACHE_DIR]:
            path.mkdir(exist_ok=True)

class EgyptologyDataCollector:
    """Collect and prepare Egyptology-related English texts"""
    
    def __init__(self, config):
        self.config = config
        
    def collect_wikipedia_data(self, sample_size=None):
        """Collect Wikipedia articles, filtered for Egyptology content"""
        logger.info("Loading Wikipedia dataset...")
        
        # Try different Wikipedia dataset versions/names
        dataset_configs = [
            ("wikimedia/wikipedia", "20231101.en"),
            ("wikimedia/wikipedia", "20230901.en"),
            ("wikipedia", "20220301.en"), 
            ("wikipedia", "20210320.en"),
            ("legacy-datasets/wikipedia", "20220301.en")
        ]
        
        wiki = None
        for dataset_name, config_name in dataset_configs:
            try:
                logger.info(f"Trying to load {dataset_name} with config {config_name}...")
                wiki = load_dataset(dataset_name, config_name, split="train", streaming=True, trust_remote_code=True)
                logger.info(f"Successfully loaded {dataset_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Final fallback - try simple Wikipedia without streaming
        if wiki is None:
            try:
                logger.info("Trying non-streaming Wikipedia dataset...")
                wiki_data = load_dataset("wikipedia", "20220301.simple", split="train[:1000]")  # Small sample
                # Convert to generator-like structure
                wiki = iter(wiki_data)
                logger.info("Successfully loaded simple Wikipedia")
            except Exception as e:
                logger.warning(f"Failed to load simple Wikipedia: {e}")
        
        if wiki is None:
            logger.warning("Could not load any Wikipedia dataset. Using synthetic Egyptology data only.")
            return self.create_extended_synthetic_data()
        
        egyptology_keywords = [
            "egypt", "egyptian", "pharaoh", "pyramid", "hieroglyph", "ancient egypt",
            "nile", "cairo", "alexandria", "luxor", "karnak", "temple", "tomb",
            "mummy", "papyrus", "sphinx", "cleopatra", "ramesses", "tutankhamun",
            "dynasty", "ptolemy", "coptic", "demotic", "hieratic", "rosetta stone",
            "champollion", "archaeology", "egyptology"
        ]
        
        collected_texts = []
        sample_size = sample_size or self.config.WIKIPEDIA_SAMPLE_SIZE
        
        logger.info(f"Collecting {sample_size} Wikipedia articles...")
        
        count = 0
        try:
            for article in tqdm(wiki, desc="Processing Wikipedia"):
                if count >= sample_size:
                    break
                    
                text = article['text'].lower()
                title = article['title'].lower()
                
                # Check if article is relevant to Egyptology
                if any(keyword in text or keyword in title for keyword in egyptology_keywords):
                    collected_texts.append({
                        'text': article['text'],
                        'title': article['title'],
                        'source': 'wikipedia'
                    })
                    count += 1
                
                # Also collect some general articles for broader context
                elif count < sample_size * 0.7:  # 70% Egyptology, 30% general
                    collected_texts.append({
                        'text': article['text'][:2000],  # Truncate general articles
                        'title': article['title'],
                        'source': 'wikipedia_general'
                    })
                    count += 1
        except Exception as e:
            logger.warning(f"Error during Wikipedia processing: {e}")
            logger.info(f"Continuing with {len(collected_texts)} articles collected so far...")
        
        # If we got very few articles, supplement with synthetic data
        if len(collected_texts) < sample_size * 0.1:  # Less than 10% of target
            logger.warning("Very few Wikipedia articles collected. Adding synthetic data.")
            synthetic_data = self.create_extended_synthetic_data()
            collected_texts.extend(synthetic_data)
        
        logger.info(f"Collected {len(collected_texts)} Wikipedia articles")
        return collected_texts
    
    def prepare_english_corpus(self):
        """Prepare the full English corpus"""
        cache_file = self.config.CACHE_DIR / "english_corpus.pkl"
        
        if cache_file.exists():
            logger.info("Loading cached English corpus...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Collect Wikipedia data
        texts = self.collect_wikipedia_data()
        
        # Add some synthetic Egyptology sentences for better alignment
        synthetic_texts = self.create_extended_synthetic_data()
        texts.extend(synthetic_texts)
        
        logger.info(f"Total English corpus size: {len(texts)} texts")
        
        # Cache the corpus
        with open(cache_file, 'wb') as f:
            pickle.dump(texts, f)
        
        return texts
    
    def create_extended_synthetic_data(self):
        """Create extended synthetic English sentences about Egyptian concepts"""
        logger.info("Creating extended synthetic Egyptology corpus...")
        
        synthetic_data = [
            # Basic concepts
            {"text": "Ancient Egyptian hieroglyphs were used for religious and administrative texts", "source": "synthetic"},
            {"text": "The pharaoh was considered a divine ruler in ancient Egypt", "source": "synthetic"},
            {"text": "Papyrus was the primary writing material in ancient Egypt", "source": "synthetic"},
            {"text": "Egyptian temples were centers of religious worship and learning", "source": "synthetic"},
            {"text": "The Rosetta Stone was key to deciphering hieroglyphic writing", "source": "synthetic"},
            {"text": "Ancient Egyptian priests were responsible for maintaining religious rituals", "source": "synthetic"},
            {"text": "Egyptian tombs contained hieroglyphic inscriptions about the afterlife", "source": "synthetic"},
            {"text": "The ancient Egyptian language evolved into Coptic", "source": "synthetic"},
            {"text": "Egyptian scribes were highly educated members of society", "source": "synthetic"},
            {"text": "Hieroglyphic texts often contained religious and funerary content", "source": "synthetic"},
            
            # Gods and religion
            {"text": "Ra was the ancient Egyptian sun god worshipped throughout the kingdom", "source": "synthetic"},
            {"text": "Anubis guided souls through the afterlife in Egyptian mythology", "source": "synthetic"},
            {"text": "Isis was a powerful goddess of magic and motherhood in Egypt", "source": "synthetic"},
            {"text": "The Book of the Dead contained spells for the Egyptian afterlife", "source": "synthetic"},
            {"text": "Horus was depicted as a falcon-headed god in Egyptian art", "source": "synthetic"},
            
            # Architecture and monuments
            {"text": "The Great Pyramid of Giza was built for pharaoh Khufu", "source": "synthetic"},
            {"text": "Egyptian temples featured massive stone columns and hieroglyphic decorations", "source": "synthetic"},
            {"text": "The Sphinx guards the pyramids of Giza with its lion body and human head", "source": "synthetic"},
            {"text": "Valley of the Kings contains many royal tombs from ancient Egypt", "source": "synthetic"},
            {"text": "Karnak Temple complex was one of the largest religious sites in Egypt", "source": "synthetic"},
            
            # Daily life and culture
            {"text": "Ancient Egyptians practiced mummification to preserve bodies for the afterlife", "source": "synthetic"},
            {"text": "The Nile River was essential for Egyptian agriculture and transportation", "source": "synthetic"},
            {"text": "Egyptian artisans created beautiful jewelry, pottery, and sculptures", "source": "synthetic"},
            {"text": "Hieratic script was used for everyday writing in ancient Egypt", "source": "synthetic"},
            {"text": "Egyptian medicine included surgical procedures and herbal remedies", "source": "synthetic"},
            
            # Historical periods
            {"text": "The Old Kingdom period saw the construction of the great pyramids", "source": "synthetic"},
            {"text": "Middle Kingdom Egypt reunified after a period of political fragmentation", "source": "synthetic"},
            {"text": "New Kingdom pharaohs expanded Egyptian territory through military campaigns", "source": "synthetic"},
            {"text": "Ptolemaic Egypt was ruled by Greek pharaohs after Alexander's conquest", "source": "synthetic"},
            {"text": "Roman rule ended the last dynasty of pharaonic Egypt", "source": "synthetic"},
            
            # Language and writing
            {"text": "Egyptian hieroglyphs combine phonetic and ideographic writing elements", "source": "synthetic"},
            {"text": "Demotic script replaced hieratic for administrative documents in later periods", "source": "synthetic"},
            {"text": "Coptic Christianity preserved the Egyptian language using Greek letters", "source": "synthetic"},
            {"text": "Jean-François Champollion deciphered hieroglyphs using the Rosetta Stone", "source": "synthetic"},
            {"text": "Egyptian texts were written in columns or horizontal lines without spaces", "source": "synthetic"},
        ]
        
        # Duplicate some entries to reach target size if needed
        target_size = min(1000, self.config.WIKIPEDIA_SAMPLE_SIZE // 10)
        while len(synthetic_data) < target_size:
            synthetic_data.extend(synthetic_data[:min(len(synthetic_data), target_size - len(synthetic_data))])
        
        logger.info(f"Created {len(synthetic_data)} synthetic Egyptology texts")
        return synthetic_data[:target_size]

class GermanTranslationProcessor:
    """Process German translations from TLA dataset and translate to English"""
    
    def __init__(self, config):
        self.config = config
        self.translator = None
        
    def setup_translator(self):
        """Initialize German to English translator"""
        logger.info("Loading German→English translator...")
        
        # Determine device - prioritize MPS for Apple Silicon
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            logger.info("Using MPS (Apple Silicon GPU) for translation")
        elif torch.cuda.is_available():
            device = 0
            logger.info("Using CUDA GPU for translation")
        else:
            device = -1
            logger.info("Using CPU for translation")
            
        self.translator = pipeline(
            "translation", 
            model=self.config.GERMAN_TRANSLATOR_MODEL,
            device=device
        )
        
    def translate_german_texts(self, german_texts, batch_size=32):
        """Translate German texts to English"""
        if not self.translator:
            self.setup_translator()
            
        logger.info(f"Translating {len(german_texts)} German texts to English...")
        
        english_translations = []
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(german_texts), batch_size), desc="Translating"):
            batch = german_texts[i:i+batch_size]
            
            try:
                # Translate batch
                translations = self.translator(batch, max_length=512)
                batch_translations = [t['translation_text'] for t in translations]
                english_translations.extend(batch_translations)
                
            except Exception as e:
                logger.warning(f"Translation error for batch {i}: {e}")
                # Add placeholder translations for failed batches
                english_translations.extend(["[Translation failed]"] * len(batch))
        
        return english_translations
    
    def process_tla_translations(self):
        """Process all German translations from TLA dataset"""
        cache_file = self.config.CACHE_DIR / "german_english_translations.pkl"
        
        if cache_file.exists():
            logger.info("Loading cached German→English translations...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load TLA dataset
        dataset = load_dataset(
            "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium",
            split="train"
        )
        
        # Extract German translations
        german_texts = []
        parallel_data = []
        
        for item in dataset:
            if item['translation'] and item['transliteration']:
                german_texts.append(item['translation'])
                parallel_data.append({
                    'hieroglyphic': item['transliteration'],
                    'german': item['translation'],
                    'hieroglyphs': item.get('hieroglyphs', ''),
                    'lemmatization': item.get('lemmatization', ''),
                    'date_not_before': item.get('dateNotBefore', ''),
                    'date_not_after': item.get('dateNotAfter', '')
                })
        
        # Translate to English
        english_translations = self.translate_german_texts(german_texts)
        
        # Combine with parallel data
        for i, english_text in enumerate(english_translations):
            if i < len(parallel_data):
                parallel_data[i]['english'] = english_text
        
        # Cache results
        with open(cache_file, 'wb') as f:
            pickle.dump(parallel_data, f)
        
        logger.info(f"Processed {len(parallel_data)} German→English translations")
class HieroglyphicDataProcessor:
    """Process TLA hieroglyphic dataset for FastText training"""
    
    def __init__(self, config):
        self.config = config
        
    def load_tla_dataset(self):
        """Load TLA dataset from HuggingFace"""
        logger.info("Loading TLA dataset...")
        
        dataset = load_dataset(
            "thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium",
            split="train"
        )
        
        return dataset
    
    def prepare_hieroglyphic_corpus(self):
        """Prepare hieroglyphic texts for FastText training"""
        cache_file = self.config.CACHE_DIR / "hieroglyphic_corpus.txt"
        
        if cache_file.exists():
            logger.info("Using cached hieroglyphic corpus...")
            return str(cache_file)
        
        dataset = self.load_tla_dataset()
        
        logger.info("Processing hieroglyphic texts...")
        
        # Prepare different text representations
        texts = []
        
        for item in tqdm(dataset, desc="Processing TLA data"):
            # Use transliteration as primary text (more suitable for FastText)
            if item['transliteration']:
                texts.append(item['transliteration'])
            
            # Also include lemmatization for better word understanding
            if item['lemmatization']:
                # Extract lemma transliterations (after the | symbol)
                lemmas = []
                for lemma_entry in item['lemmatization'].split():
                    if '|' in lemma_entry:
                        lemma = lemma_entry.split('|')[1]
                        lemmas.append(lemma)
                if lemmas:
                    texts.append(' '.join(lemmas))
        
        # Write to file for FastText
        with open(cache_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        logger.info(f"Prepared {len(texts)} hieroglyphic texts")
        return str(cache_file)
    
    def get_parallel_data(self):
        """Extract parallel hieroglyphic-German data for alignment"""
        dataset = self.load_tla_dataset()
        
        parallel_data = []
        for item in dataset:
            if item['transliteration'] and item['translation']:
                parallel_data.append({
                    'hieroglyphic': item['transliteration'],
                    'german': item['translation'],
                    'hieroglyphs': item.get('hieroglyphs', ''),
                    'lemmatization': item.get('lemmatization', '')
                })
        
        return parallel_data

class MultilingualEmbeddingSpace:
    """Create embedding spaces for different languages using sentence transformers"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def create_embeddings(self, texts, language="en"):
        """Create embeddings for texts in specified language"""
        logger.info(f"Loading sentence transformer model for {language}...")
        
        # Use language-specific model if available, otherwise use multilingual
        if language == "de":
            model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        else:
            model_name = self.config.SENTENCE_TRANSFORMER_MODEL
            
        self.model = SentenceTransformer(model_name)
        
        logger.info(f"Creating embeddings for {len(texts)} {language} texts...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        return embeddings
    
    def save_model(self, language="en"):
        """Save the sentence transformer model"""
        model_path = self.config.MODELS_DIR / f"{language}_sentence_transformer"
        self.model.save(str(model_path))
        return model_path

class HieroglyphicEmbeddingSpace:
    """Create hieroglyphic embedding space using FastText"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def train_fasttext_model(self, corpus_file):
        """Train FastText model on hieroglyphic corpus"""
        logger.info("Training FastText model on hieroglyphic texts...")
        
        self.model = fasttext.train_unsupervised(
            corpus_file,
            model='skipgram',
            dim=self.config.FASTTEXT_DIM,
            epoch=self.config.FASTTEXT_EPOCHS,
            minCount=self.config.FASTTEXT_MIN_COUNT,
            thread=4
        )
        
        model_path = self.config.MODELS_DIR / "hieroglyphic_fasttext.bin"
        self.model.save_model(str(model_path))
        
        logger.info(f"FastText model saved to {model_path}")
        return model_path
    
    def create_embeddings(self, texts):
        """Create embeddings for hieroglyphic texts"""
        if not self.model:
            # Load model if not already loaded
            model_path = self.config.MODELS_DIR / "hieroglyphic_fasttext.bin"
            self.model = fasttext.load_model(str(model_path))
        
        logger.info("Creating hieroglyphic embeddings...")
        embeddings = []
        
        for text in tqdm(texts, desc="Embedding hieroglyphic texts"):
            # Get sentence embedding by averaging word embeddings
            words = text.split()
            if words:
                word_embeddings = [self.model.get_word_vector(word) for word in words]
                sentence_embedding = np.mean(word_embeddings, axis=0)
                embeddings.append(sentence_embedding)
            else:
                # Handle empty text
                embeddings.append(np.zeros(self.config.FASTTEXT_DIM))
        
        return np.array(embeddings)

# vec2vec Implementation (enhanced for multiple spaces)
class MultiSpaceVec2VecDataset(Dataset):
    """Dataset for vec2vec training with multiple embedding spaces"""
    
    def __init__(self, *embedding_spaces):
        # Find minimum length across all spaces
        min_len = min(len(embeddings) for embeddings in embedding_spaces)
        
        # Truncate all spaces to same length and convert to tensors
        self.embeddings = []
        for embeddings in embedding_spaces:
            self.embeddings.append(torch.FloatTensor(embeddings[:min_len]))
        
    def __len__(self):
        return len(self.embeddings[0])
    
    def __getitem__(self, idx):
        return tuple(embeddings[idx] for embeddings in self.embeddings)

class MultiSpaceVec2VecModel(nn.Module):
    """Enhanced vec2vec model for multiple embedding spaces"""
    
    def __init__(self, input_dims, latent_dim, num_spaces=3):
        super().__init__()
        
        self.num_spaces = num_spaces
        self.input_dims = input_dims
        
        # Encoders to shared latent space
        self.encoders = nn.ModuleList()
        for i, input_dim in enumerate(input_dims):
            encoder = nn.Sequential(
                nn.Linear(input_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim * 2, latent_dim),
                nn.ReLU()
            )
            self.encoders.append(encoder)
        
        # Decoders from shared latent space
        self.decoders = nn.ModuleList()
        for i, input_dim in enumerate(input_dims):
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim * 2, input_dim)
            )
            self.decoders.append(decoder)
        
    def forward(self, *inputs):
        # Encode all inputs to latent space
        latents = []
        for i, x in enumerate(inputs):
            z = self.encoders[i](x)
            latents.append(z)
        
        outputs = {'latents': latents}
        
        # Reconstruction: decode each latent back to its original space
        for i, z in enumerate(latents):
            outputs[f'recon_{i}'] = self.decoders[i](z)
        
        # Cross-translations: decode each latent to all other spaces
        for i, z_source in enumerate(latents):
            for j in range(len(latents)):
                if i != j:
                    outputs[f'cross_{i}_to_{j}'] = self.decoders[j](z_source)
        
        return outputs

class MultiSpaceVec2VecTrainer:
    """Trainer for multi-space vec2vec model"""
    
    def __init__(self, config):
        self.config = config
        
        # Prioritize MPS for Apple Silicon, then CUDA, then CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            logger.info("Using MPS (Apple Silicon GPU) for vec2vec training")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Using CUDA GPU for vec2vec training")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for vec2vec training")
        
    def train(self, *embedding_spaces):
        """Train multi-space vec2vec model"""
        logger.info("Setting up multi-space vec2vec training...")
        
        # Create dataset
        dataset = MultiSpaceVec2VecDataset(*embedding_spaces)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Initialize model
        input_dims = [embeddings.shape[1] for embeddings in embedding_spaces]
        num_spaces = len(embedding_spaces)
        
        model = MultiSpaceVec2VecModel(input_dims, self.config.LATENT_DIM, num_spaces)
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        logger.info(f"Training {num_spaces}-space vec2vec model...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0
            
            for batch_idx, inputs in enumerate(dataloader):
                inputs = [x.to(self.device) for x in inputs]
                
                optimizer.zero_grad()
                
                outputs = model(*inputs)
                
                loss = 0
                
                # Reconstruction losses
                for i in range(num_spaces):
                    recon_loss = nn.MSELoss()(outputs[f'recon_{i}'], inputs[i])
                    loss += recon_loss
                
                # Cross-translation cycle consistency losses
                for i in range(num_spaces):
                    for j in range(num_spaces):
                        if i != j:
                            # Cycle: space_i → space_j → space_i
                            cycle_loss = nn.MSELoss()(outputs[f'cross_{i}_to_{j}'], inputs[j])
                            loss += 0.5 * cycle_loss
                
                # Latent alignment losses (encourage similar latent representations)
                latent_alignment_loss = 0
                latents = outputs['latents']
                for i in range(num_spaces):
                    for j in range(i+1, num_spaces):
                        alignment_loss = nn.MSELoss()(latents[i], latents[j])
                        latent_alignment_loss += alignment_loss
                
                loss += 0.1 * latent_alignment_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model
        model_path = self.config.MODELS_DIR / "multi_space_vec2vec_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dims': input_dims,
            'latent_dim': self.config.LATENT_DIM,
            'num_spaces': num_spaces
        }, model_path)
        
        return model

class EmbeddingManager:
    """Manage saving/loading of embeddings and enable follow-up tasks"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings_dir = config.CACHE_DIR / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
    def save_embeddings(self, embeddings, texts, name, metadata=None):
        """Save embeddings with associated texts and metadata"""
        save_data = {
            'embeddings': embeddings,
            'texts': texts,
            'metadata': metadata or {},
            'shape': embeddings.shape,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        filepath = self.embeddings_dir / f"{name}_embeddings.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved {name} embeddings: {embeddings.shape} to {filepath}")
        return filepath
    
    def load_embeddings(self, name):
        """Load embeddings with associated data"""
        filepath = self.embeddings_dir / f"{name}_embeddings.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {name} embeddings: {data['shape']} from {filepath}")
        return data
    
    def list_available_embeddings(self):
        """List all available saved embeddings"""
        embedding_files = list(self.embeddings_dir.glob("*_embeddings.pkl"))
        available = []
        
        for filepath in embedding_files:
            name = filepath.stem.replace("_embeddings", "")
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                available.append({
                    'name': name,
                    'shape': data['shape'],
                    'timestamp': data.get('timestamp', 'unknown'),
                    'metadata': data.get('metadata', {})
                })
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")
        
        return available
    
    def save_vec2vec_model_info(self, model_path, space_names, input_dims):
        """Save vec2vec model information for later use"""
        model_info = {
            'model_path': str(model_path),
            'space_names': space_names,
            'input_dims': input_dims,
            'latent_dim': self.config.LATENT_DIM,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        info_path = self.config.CACHE_DIR / "vec2vec_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
    
def evaluate_embedding_spaces(config, wikipedia_embeddings, tla_english_embeddings, hieroglyphic_embeddings,
                            wikipedia_texts, tla_english_texts, hieroglyphic_texts, parallel_data):
    """Comprehensive evaluation of embedding spaces and vec2vec alignment"""
    logger.info("Starting comprehensive evaluation...")
    
    # Basic statistics
    logger.info(f"Wikipedia embeddings: {wikipedia_embeddings.shape}")
    logger.info(f"TLA English embeddings: {tla_english_embeddings.shape}")
    logger.info(f"Hieroglyphic embeddings: {hieroglyphic_embeddings.shape}")
    
    # Load trained model for evaluation
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() 
                             else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        model_path = config.MODELS_DIR / "multi_space_vec2vec_model.pt"
        if device.type == 'mps':
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = torch.load(model_path, map_location=device)
        
        model = MultiSpaceVec2VecModel(
            checkpoint['input_dims'],
            checkpoint['latent_dim'],
            checkpoint['num_spaces']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info("Successfully loaded trained vec2vec model for evaluation")
        
        # Sample embeddings for evaluation
        sample_size = min(100, len(tla_english_embeddings))
        indices = np.random.choice(len(tla_english_embeddings), sample_size, replace=False)
        
        wiki_sample = torch.FloatTensor(wikipedia_embeddings[indices]).to(device)
        tla_sample = torch.FloatTensor(tla_english_embeddings[indices]).to(device)
        hier_sample = torch.FloatTensor(hieroglyphic_embeddings[indices]).to(device)
        
        with torch.no_grad():
            outputs = model(wiki_sample, tla_sample, hier_sample)
            
            # Calculate latent space similarities
            latents = outputs['latents']
            
            # Wikipedia ↔ TLA English alignment
            wiki_tla_sim = cosine_similarity(latents[0].cpu().numpy(), latents[1].cpu().numpy())
            wiki_tla_avg = np.mean(np.diag(wiki_tla_sim))
            
            # TLA English ↔ Hieroglyphic alignment (ground truth)
            tla_hier_sim = cosine_similarity(latents[1].cpu().numpy(), latents[2].cpu().numpy())
            tla_hier_avg = np.mean(np.diag(tla_hier_sim))
            
            # Wikipedia ↔ Hieroglyphic alignment (our main bridge)
            wiki_hier_sim = cosine_similarity(latents[0].cpu().numpy(), latents[2].cpu().numpy())
            wiki_hier_avg = np.mean(np.diag(wiki_hier_sim))
            
            logger.info("=== Latent Space Alignment Results ===")
            logger.info(f"Wikipedia ↔ TLA English: {wiki_tla_avg:.4f}")
            logger.info(f"TLA English ↔ Hieroglyphic (ground truth): {tla_hier_avg:.4f}")
            logger.info(f"Wikipedia ↔ Hieroglyphic (main bridge): {wiki_hier_avg:.4f}")
            
            # Show some example alignments
            logger.info("\n=== Example Alignments ===")
            for i in range(min(5, sample_size)):
                idx = indices[i]
                logger.info(f"\nSample {i+1}:")
                logger.info(f"Wikipedia: {wikipedia_texts[idx][:80]}...")
                logger.info(f"TLA English: {tla_english_texts[idx][:80]}...")
                logger.info(f"Hieroglyphic: {hieroglyphic_texts[idx]}")
                logger.info(f"Similarities - Wiki↔TLA: {wiki_tla_sim[i,i]:.3f}, TLA↔Hier: {tla_hier_sim[i,i]:.3f}, Wiki↔Hier: {wiki_hier_sim[i,i]:.3f}")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        logger.info("Proceeding with basic embedding statistics...")
    
    # Basic embedding space analysis
    logger.info("\n=== Embedding Space Statistics ===")
    
    # Calculate some basic similarities between spaces
    sample_size = min(50, len(tla_english_embeddings))
    sample_indices = np.random.choice(len(tla_english_embeddings), sample_size, replace=False)
    
    # Direct similarity between TLA English and Hieroglyphic (should be high - same source)
    tla_sample = tla_english_embeddings[sample_indices]
    hier_sample = hieroglyphic_embeddings[sample_indices]
    
    # Since they're different dimensions, we can't directly compare
    # This would be done through the vec2vec model in practice
    logger.info(f"TLA English embedding dims: {tla_sample.shape[1]}")
    logger.info(f"Hieroglyphic embedding dims: {hier_sample.shape[1]}")
    logger.info("Direct comparison requires vec2vec translation (different dimensions)")
    
    # Show some sample texts
    logger.info("\n=== Sample Parallel Texts ===")
    for i in range(min(3, len(parallel_data))):
        item = parallel_data[i]
        logger.info(f"\nSample {i+1}:")
        logger.info(f"German: {item['german']}")
        logger.info(f"English: {item['english']}")
        logger.info(f"Hieroglyphic: {item['hieroglyphic']}")
        
    logger.info("\n=== Evaluation Complete ===")

def run_evaluation_only(config, embedding_manager):
    """
    Load all embeddings and run evaluation using existing data.
    """
    # Load embeddings
    wikipedia = embedding_manager.load_embeddings('wikipedia')
    tla_english = embedding_manager.load_embeddings('tla_english')
    hieroglyphic = embedding_manager.load_embeddings('hieroglyphic')
    german = embedding_manager.load_embeddings('german')

    # Load parallel data (for sample texts)
    german_processor = GermanTranslationProcessor(config)
    parallel_data = german_processor.process_tla_translations()

    # Call evaluation
    evaluate_embedding_spaces(
        config,
        wikipedia['embeddings'],
        tla_english['embeddings'],
        hieroglyphic['embeddings'],
        wikipedia['texts'],
        tla_english['texts'],
        hieroglyphic['texts'],
        parallel_data
    )

def main():
    import sys
    config = Config()
    embedding_manager = EmbeddingManager(config)

    # Smart logic: check for fresh argument
    force_fresh = len(sys.argv) > 1 and sys.argv[1] == 'fresh'
    logger.info("🏺 Egyptian Hieroglyphic-English Embedding Bridge (Smart Mode)")
    if force_fresh:
        logger.info("🔄 FRESH MODE: Will recreate all data")
    else:
        logger.info("⚡ SMART MODE: Will reuse existing data")
    check_device_availability()
    available_embeddings = embedding_manager.list_available_embeddings()
    print("[DEBUG] Available embeddings:", [emb['name'] for emb in available_embeddings])
    logger.info(f"[DEBUG] Available embeddings: {[emb['name'] for emb in available_embeddings]}")
    vec2vec_model_path = config.MODELS_DIR / "multi_space_vec2vec_model.pt"
    vec2vec_exists = vec2vec_model_path.exists()
    logger.info(f"\n📊 Current Status:")
    logger.info(f"Found {len(available_embeddings)} embedding spaces:")
    for emb in available_embeddings:
        logger.info(f"  ✅ {emb['name']}: {emb['shape']} ({emb['timestamp'][:19]})")
    if vec2vec_exists:
        logger.info(f"✅ vec2vec model: {vec2vec_model_path}")
    else:
        logger.info("❌ vec2vec model: Not found")
    required_embeddings = ['wikipedia', 'tla_english', 'german', 'hieroglyphic']
    existing_embeddings = [emb['name'] for emb in available_embeddings]
    missing_embeddings = [name for name in required_embeddings if name not in existing_embeddings]
    # Decision logic
    if not force_fresh and not missing_embeddings and vec2vec_exists:
        logger.info("\n🎉 All data exists! Proceeding directly to evaluation...")
        return run_evaluation_only(config, embedding_manager)
    # Create missing data
    logger.info(f"\n🚧 Creating missing data...")
    if force_fresh or missing_embeddings:
        logger.info(f"Missing: {missing_embeddings if not force_fresh else 'All (fresh mode)'}")
    return run_full_pipeline(config, embedding_manager, existing_embeddings, force_fresh)


def test_translation(config, english_embeddings, hieroglyphic_embeddings, 
                    english_texts, hieroglyphic_texts):
    """Test the translation between embedding spaces"""
    
    # Load trained model
    model_path = config.MODELS_DIR / "vec2vec_model.pt"
    
    english_dim = english_embeddings.shape[1]
    hieroglyphic_dim = hieroglyphic_embeddings.shape[1]
    
    model = Vec2VecModel(english_dim, hieroglyphic_dim, config.LATENT_DIM)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test translation
    with torch.no_grad():
        # Sample some embeddings
        sample_indices = np.random.choice(len(english_embeddings), 5, replace=False)
        
        for idx in sample_indices:
            english_emb = torch.FloatTensor(english_embeddings[idx:idx+1]).to(device)
            hieroglyphic_emb = torch.FloatTensor(hieroglyphic_embeddings[idx:idx+1]).to(device)
            
            outputs = model(english_emb, hieroglyphic_emb)
            
            # Calculate similarity between latent representations
            z1 = outputs['z1'].cpu().numpy()
            z2 = outputs['z2'].cpu().numpy()
            similarity = cosine_similarity(z1, z2)[0][0]
            
            logger.info(f"\nSample {idx}:")
            logger.info(f"English: {english_texts[idx][:100]}...")
            logger.info(f"Hieroglyphic: {hieroglyphic_texts[idx]}")
            logger.info(f"Latent similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
