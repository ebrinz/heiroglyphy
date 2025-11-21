# V6 Strategy: FastText vs. BERT

## FastText (V5 Baseline)
**Pros:**
- **Efficiency**: Trains in minutes on CPU.
- **Data Efficiency**: Works well with small datasets (104k texts).
- **Subword Information**: Captures morphology via n-grams (crucial for hieroglyphs).
- **Simplicity**: Static embeddings are easy to align with Procrustes.

**Cons:**
- **Context-Blind**: "A1" (Man) has the same vector whether he is a king or a captive.
- **Shallow**: Cannot capture complex syntax or long-range dependencies.
- **Visual Integration**: Limited to simple concatenation of visual vectors.

## BERT (V6 HieroBERT)
**Pros:**
- **Context-Aware**: The vector for "A1" changes based on surrounding glyphs. This is the core goal of V6.
- **Deep Integration**: Can use visual embeddings as input tokens or attention biases.
- **State-of-the-Art**: Standard for modern NLP; 768d aligns with our visual features.
- **Fine-Tuning**: Can be fine-tuned for specific tasks (translation, classification).

**Cons:**
- **Data Hungry**: 104k texts is "small data" for transformers. Risk of overfitting.
- **Compute Intensive**: Requires GPU for training; slower inference.
- **Alignment Complexity**: Contextual embeddings are dynamic; aligning them to static English vectors requires pooling or advanced techniques.

## Verdict
**Winner: BERT (HieroBERT)**
Despite the data size risk, BERT is the only way to achieve "Context-Aware" alignment.
- **Mitigation**: We will train a *small* BERT (DistilBERT or TinyBERT) to prevent overfitting.
- **Synergy**: The 768d visual embeddings we generated are a perfect match for BERT Base.