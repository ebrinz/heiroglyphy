from datasets import load_dataset

print("Loading bbaw_egyptian dataset...")
try:
    # Load the dataset (it might have multiple configs, usually 'default' works)
    dataset = load_dataset("bbaw_egyptian", trust_remote_code=True)
    
    print("\nDataset Structure:")
    print(dataset)
    
    print("\nSample Data (first 3 items from train split):")
    if 'train' in dataset:
        count_with_hieroglyphs = 0
        samples = []
        for item in dataset['train']:
            if item['hieroglyphs'] and item['hieroglyphs'].strip():
                count_with_hieroglyphs += 1
                if len(samples) < 3:
                    samples.append(item)
        
        print(f"\nTotal records with hieroglyphs: {count_with_hieroglyphs}/{len(dataset['train'])}")
        
        print("\nSample records with hieroglyphs:")
        for i, item in enumerate(samples):
            print(f"\n--- Record {i+1} ---")
            print(item)
    else:
        print("No 'train' split found.")
        
except Exception as e:
    print(f"Error loading dataset: {e}")
