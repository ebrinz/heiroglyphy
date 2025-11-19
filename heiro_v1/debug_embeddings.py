#!/usr/bin/env python3
"""
Debug script to check what embeddings actually exist
"""

import sys
from pathlib import Path

# Import from the original file
try:
    from heiro import *
except ImportError:
    print("Please make sure heiro.py is in the same directory")
    sys.exit(1)

def debug_embeddings():
    """Debug what embeddings actually exist"""
    config = Config()
    embedding_manager = EmbeddingManager(config)
    
    print("🔍 DEBUGGING EXISTING EMBEDDINGS")
    print("=" * 50)
    
    # Check directories
    cache_dir = config.CACHE_DIR
    embeddings_dir = cache_dir / "embeddings"
    models_dir = config.MODELS_DIR
    
    print(f"📁 Cache directory: {cache_dir}")
    print(f"   Exists: {cache_dir.exists()}")
    
    print(f"📁 Embeddings directory: {embeddings_dir}")
    print(f"   Exists: {embeddings_dir.exists()}")
    
    if embeddings_dir.exists():
        files = list(embeddings_dir.glob("*.pkl"))
        print(f"   Files found: {len(files)}")
        for file in files:
            print(f"     - {file.name} ({file.stat().st_size / 1024 / 1024:.1f}MB)")
    
    print(f"📁 Models directory: {models_dir}")
    print(f"   Exists: {models_dir.exists()}")
    
    if models_dir.exists():
        files = list(models_dir.glob("*"))
        print(f"   Files/dirs found: {len(files)}")
        for file in files:
            if file.is_file():
                print(f"     - {file.name} ({file.stat().st_size / 1024 / 1024:.1f}MB)")
            else:
                print(f"     - {file.name}/ (directory)")
    
    # Try to list available embeddings
    print("\\n🔍 TESTING EMBEDDING MANAGER")
    print("=" * 30)
    
    try:
        available_embeddings = embedding_manager.list_available_embeddings()
        print(f"✅ Found {len(available_embeddings)} embeddings via manager:")
        for emb in available_embeddings:
            print(f"   - {emb['name']}: {emb['shape']}")
    except Exception as e:
        print(f"❌ Error listing embeddings: {e}")
        import traceback
        traceback.print_exc()
    
    # Check vec2vec model
    vec2vec_path = config.MODELS_DIR / "multi_space_vec2vec_model.pt"
    print(f"\\n🤖 VEC2VEC MODEL CHECK")
    print(f"Path: {vec2vec_path}")
    print(f"Exists: {vec2vec_path.exists()}")
    if vec2vec_path.exists():
        print(f"Size: {vec2vec_path.stat().st_size / 1024 / 1024:.1f}MB")
    
    print("\\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    debug_embeddings()
