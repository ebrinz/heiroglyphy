# Egyptian Hieroglyphic-English Embedding Bridge

A comprehensive system that bridges ancient Egyptian hieroglyphic texts with modern English knowledge using state-of-the-art embedding techniques and vec2vec translation.

## 🏺 Overview

This project creates three interconnected embedding spaces:
1. **Wikipedia English Space**: Modern Egyptology knowledge from Wikipedia articles
2. **TLA German→English Space**: Ground truth translations from the Thesaurus Linguae Aegyptiae
3. **Hieroglyphic FastText Space**: Ancient Egyptian transliterations using subword embeddings

The system uses vec2vec (unsupervised embedding translation) to bridge these spaces, enabling semantic search and analysis across 4,000 years of language evolution.

## 📋 Requirements

### Python Dependencies
```bash
pip install datasets sentence-transformers fasttext torch transformers scikit-learn numpy pandas tqdm
```

### System Requirements
- **RAM**: 16GB+ recommended (for large Wikipedia dataset processing)
- **Storage**: ~5GB for datasets and models
- **GPU**: Optional but recommended for faster training
  - **Apple Silicon (M1/M2/M3)**: MPS acceleration supported ✅
  - **NVIDIA GPUs**: CUDA acceleration supported ✅
  - **CPU fallback**: Works on any system
- **Time**: Initial training takes 2-8 hours depending on hardware
  - Apple Silicon (MPS): ~2-4 hours
  - NVIDIA GPU (CUDA): ~2-3 hours  
  - CPU only: ~6-8 hours

## 🚀 Quick Start

### Initial Setup
```bash
cd /Users/crashy/Desktop/heiro
pip install datasets sentence-transformers fasttext torch transformers scikit-learn numpy pandas tqdm
```

### First Run (Full Training Pipeline)
```bash
python heiro.py
```

This will:
1. ✅ Download TLA dataset (~12,773 hieroglyphic sentences)
2. ✅ Translate German→English using Helsinki-NLP model
3. ✅ Collect Egyptology-relevant Wikipedia articles (~100K)
4. ✅ Create Wikipedia English embeddings (sentence-transformers)
5. ✅ Create TLA German→English embeddings (ground truth)
6. ✅ Train FastText on hieroglyphic transliterations
7. ✅ Train 3-way vec2vec bridge between all spaces
8. ✅ Save all embeddings and models
9. ✅ Run evaluation and demonstration

### Follow-up Analysis (Using Saved Embeddings)
```bash
python heiro.py follow-up
```

This demonstrates:
- 🔍 Semantic search across embedding spaces
- 🌐 Cross-space concept exploration
- 📊 Translation quality analysis
- 📁 Data export for external tools

## 📁 Project Structure

After running, your directory will contain:

```
/Users/crashy/Desktop/heiro/
├── heiro.py                      # Main implementation
├── README.md                     # This file
├── cache/                        # Cached data and embeddings
│   ├── embeddings/              
│   │   ├── wikipedia_embeddings.pkl      # Wikipedia English vectors
│   │   ├── tla_english_embeddings.pkl    # TLA German→English vectors
│   │   ├── german_embeddings.pkl         # Original German vectors
│   │   └── hieroglyphic_embeddings.pkl   # Hieroglyphic FastText vectors
│   ├── german_english_translations.pkl   # Processed TLA translations
│   ├── english_corpus.pkl               # Wikipedia corpus
│   ├── hieroglyphic_corpus.txt          # FastText training data
│   └── vec2vec_model_info.json          # Model metadata
├── models/                       # Trained models
│   ├── multi_space_vec2vec_model.pt     # Main vec2vec bridge
│   ├── hieroglyphic_fasttext.bin        # Hieroglyphic FastText model
│   ├── wikipedia_en/                    # Wikipedia sentence transformer
│   ├── tla_english/                     # TLA English sentence transformer
│   └── german/                          # German sentence transformer
└── data/                         # Processed data and exports
    └── exports/                  # CSV/NumPy exports for analysis
        ├── wikipedia_embeddings.csv
        ├── tla_english_embeddings.csv
        ├── hieroglyphic_embeddings.csv
        └── *_metadata.csv
```

## 🔬 Research Applications

### 1. Semantic Search Across Time
Find hieroglyphic texts semantically similar to modern English concepts:

```python
from heiro import FollowUpTaskManager, Config

config = Config()
task_manager = FollowUpTaskManager(config)

# Search hieroglyphic space for English concepts
results = task_manager.semantic_search("pharaoh temple", "hieroglyphic", top_k=10)
for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Hieroglyphic: {result['text']}")
```

### 2. Cross-Space Concept Analysis
Compare how concepts appear across different embedding spaces:

```python
# Explore concept across all spaces
concept_results = task_manager.concept_exploration(["god", "temple", "priest"], num_samples=10)
for space, results in concept_results.items():
    print(f"\n{space.upper()} space:")
    for result in results[:3]:
        print(f"  [{result['similarity']:.3f}] {result['text'][:80]}...")
```

### 3. Translation Quality Validation
Analyze how well the vec2vec bridge aligns with ground truth German→English translations:

```python
quality_results = task_manager.translation_quality_analysis()
print("Best aligned translations (TLA English ↔ Hieroglyphic):")
for result in quality_results[:5]:
    print(f"Latent Similarity: {result['latent_similarity']:.3f}")
    print(f"English: {result['tla_text']}")
    print(f"Hieroglyphic: {result['hieroglyphic_text']}")
```

### 4. Data Export for External Analysis
Export embeddings for use in R, MATLAB, or other tools:

```python
# Export as CSV
export_dir = task_manager.export_embeddings_for_analysis('csv')

# Export as NumPy arrays
export_dir = task_manager.export_embeddings_for_analysis('npy')
```

## 📊 Technical Details

### Embedding Spaces
1. **Wikipedia English**: 768-dim sentence-transformers (`all-mpnet-base-v2`)
2. **TLA German→English**: 768-dim sentence-transformers (translated via `Helsinki-NLP/opus-mt-de-en`)
3. **Hieroglyphic FastText**: 300-dim FastText trained on transliterations

### vec2vec Architecture
- **Shared latent space**: 256 dimensions
- **Loss functions**: Reconstruction + cycle consistency + latent alignment
- **Training**: 100 epochs with adaptive learning rate

### Dataset Sources
- **TLA Dataset**: `thesaurus-linguae-aegyptiae/tla-Earlier_Egyptian_original-v18-premium` (HuggingFace)
- **Wikipedia**: English articles filtered for Egyptology content
- **Translation**: German→English via Helsinki-NLP transformer

## 🎯 Research Questions You Can Explore

### Historical Linguistics
- How do ancient Egyptian concepts map to modern understanding?
- Which hieroglyphic terms have direct modern equivalents?
- How does semantic drift appear across millennia?

### Digital Humanities
- Can we identify previously unknown connections between texts?
- How do religious vs. administrative texts cluster semantically?
- What insights emerge from cross-temporal concept analysis?

### Computational Methods
- How effective is vec2vec for low-resource ancient languages?
- Does German→English ground truth validate Wikipedia→Hieroglyphic alignment?
- Can subword embeddings capture hieroglyphic morphology?

## 🔧 Customization Options

### Adjust Dataset Sizes
Edit `Config` class in `heiro.py`:
```python
class Config:
    WIKIPEDIA_SAMPLE_SIZE = 50000  # Reduce for faster processing
    FASTTEXT_EPOCHS = 5           # Reduce for quicker training
    NUM_EPOCHS = 50               # Reduce vec2vec epochs
```

### Change Models
```python
# Use different sentence transformers
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Faster, smaller

# Use different German translator
GERMAN_TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-de-en"
```

### Add New Embedding Spaces
Extend the `MultiSpaceVec2VecModel` to include additional languages or text types.

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Reduce batch sizes and sample sizes in Config class
BATCH_SIZE = 32  # Reduce from 64
WIKIPEDIA_SAMPLE_SIZE = 25000  # Reduce from 100000
```

**2. GPU Errors (MPS/CUDA)**
```python
# Force CPU usage if GPU issues occur
import torch

# Disable MPS
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# Disable CUDA  
torch.cuda.is_available = lambda: False
```

**3. MPS Memory Issues (Apple Silicon)**
```python
# If you get MPS memory errors, reduce batch size
class Config:
    BATCH_SIZE = 32  # Reduce from 64
    # Or force CPU for problematic operations
```

**3. Dataset Download Failures**
- Check internet connection
- Try rerunning - HuggingFace datasets sometimes timeout
- Clear cache: `rm -rf ~/.cache/huggingface/`

**4. FastText Training Issues**
```bash
# Install FastText from source if pip version fails
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

### Performance Optimization

**For Faster Development:**
- Use smaller Wikipedia sample sizes
- Reduce vec2vec epochs
- Use CPU-only mode for initial testing

**For Production:**
- **Apple Silicon (M1/M2/M3)**: Automatically uses MPS acceleration
- **NVIDIA GPUs**: Automatically uses CUDA acceleration
- **CPU systems**: Increase batch sizes if memory allows
- Use full datasets for best results

**Device Priority (Automatic):**
1. MPS (Apple Silicon GPU) - **Recommended for M1/M2/M3 Macs**
2. CUDA (NVIDIA GPU)
3. CPU (fallback)

## 📈 Expected Results

### Training Metrics
- **vec2vec Loss**: Should decrease to <0.1 after 100 epochs
- **Latent Similarities**: >0.7 for well-aligned concepts
- **Cross-space Search**: Top-10 accuracy varies by concept

### Semantic Search Examples
Modern query: "temple priest ritual"
Expected hieroglyphic results:
- `ḥm-nṯr pr-aA` (priest of pharaoh)
- `ḥwt-nṯr nt ra` (temple of Ra)
- `ir.t tp-ra` (performing daily ritual)

## 🤝 Contributing

This research tool is designed for:
- Digital humanities researchers
- Egyptologists interested in computational methods
- NLP researchers working on low-resource languages
- Anyone exploring cross-temporal semantic analysis

### Extending the Project
- Add new ancient languages (Sumerian, Akkadian)
- Incorporate image analysis of actual hieroglyphs
- Develop interactive visualization tools
- Create API for external applications

## 📚 References

### Core Papers
- **vec2vec**: "Harnessing the Universal Geometry of Embeddings" (Jha et al., 2025)
- **TLA Dataset**: Thesaurus Linguae Aegyptiae project documentation
- **FastText**: "Enriching Word Vectors with Subword Information" (Bojanowski et al., 2017)

### Datasets
- **TLA**: Berlin-Brandenburg Academy of Sciences
- **Wikipedia**: Wikimedia Foundation
- **Helsinki-NLP**: University of Helsinki translation models

## 📄 License

This project uses datasets and models with various licenses:
- **TLA Dataset**: CC BY-SA 4.0
- **Wikipedia**: CC BY-SA 3.0
- **Sentence Transformers**: Apache 2.0
- **FastText**: MIT License

Please cite appropriate sources when using this research tool.

---

**Happy exploring the intersection of ancient wisdom and modern AI!** 🏺🤖

For questions or issues, check the troubleshooting section above or examine the detailed logging output during execution.
