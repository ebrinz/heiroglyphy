#!/usr/bin/env python3
"""
Force evaluation mode - directly load existing data and run evaluation
"""

from heiro import *

def force_evaluation():
    """Force evaluation using existing data"""
    config = Config()
    embedding_manager = EmbeddingManager(config)
    
    print("🎯 FORCE EVALUATION MODE")
    print("Loading all existing data directly...")
    
    try:
        # Load embeddings directly from files
        cache_dir = config.CACHE_DIR / "embeddings"
        
        print(f"📁 Looking in: {cache_dir}")
        
        # Check what files exist
        files = list(cache_dir.glob("*_embeddings.pkl"))
        print(f"Found {len(files)} embedding files:")
        for f in files:
            print(f"  - {f.name}")
        
        # Try to load each one
        embeddings_data = {}
        
        for name in ['wikipedia', 'tla_english', 'hieroglyphic']:
            file_path = cache_dir / f"{name}_embeddings.pkl"
            if file_path.exists():
                print(f"🔄 Loading {name}...")
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                embeddings_data[name] = data
                print(f"✅ {name}: {data['embeddings'].shape}")
            else:
                print(f"❌ {name}: Not found")
        
        if len(embeddings_data) >= 3:
            print("\\n🎉 Sufficient data found! Running evaluation...")
            
            # Extract data
            wiki_data = embeddings_data.get('wikipedia')
            tla_data = embeddings_data.get('tla_english') 
            hier_data = embeddings_data.get('hieroglyphic')
            
            if wiki_data and tla_data and hier_data:
                # Get parallel data for context
                german_processor = GermanTranslationProcessor(config)
                parallel_data = german_processor.process_tla_translations()
                
                # Align sizes
                min_samples = min(
                    len(wiki_data['embeddings']),
                    len(tla_data['embeddings']),
                    len(hier_data['embeddings'])
                )
                
                print(f"📏 Aligned sample size: {min_samples}")
                
                # Run evaluation
                evaluate_embedding_spaces(
                    config,
                    wiki_data['embeddings'][:min_samples],
                    tla_data['embeddings'][:min_samples],
                    hier_data['embeddings'][:min_samples],
                    wiki_data['texts'][:min_samples],
                    tla_data['texts'][:min_samples],
                    hier_data['texts'][:min_samples],
                    parallel_data[:min_samples]
                )
                
                # Run follow-up tasks
                print("\\n=== Follow-up Tasks ===")
                demonstrate_follow_up_tasks(config)
                
                print("\\n✨ Force evaluation complete!")
            else:
                print("❌ Missing required embedding data")
        else:
            print("❌ Insufficient embedding data found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    force_evaluation()
