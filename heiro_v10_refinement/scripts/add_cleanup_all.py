import json
from pathlib import Path

def add_cleanup_to_notebook(nb_path):
    """Add cleanup cell to a notebook if not already present."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    # Check if cleanup already exists
    has_cleanup = any(
        'gc.collect()' in str(cell.get('source', ''))
        for cell in nb['cells']
    )
    
    if has_cleanup:
        print(f'  ℹ {nb_path.name} already has cleanup')
        return
    
    # Add cleanup cells
    cleanup_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## Cleanup Memory']
    }
    
    gc_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            'import gc\n',
            '\n',
            '# Delete large objects\n',
            'print("Freeing up memory...")\n',
            '\n',
            'del fasttext_model\n',
            'del text_embeddings\n',
            'del visual_embeddings\n',
            'del english_embeddings\n',
            'del fused_embeddings\n',
            'del X, Y, X_train, X_test, Y_train, Y_test, Y_pred\n',
            'del anchors, valid_anchors, anchors_train, anchors_test\n',
            'del aligner\n',
            '\n',
            '# Run garbage collector\n',
            'gc.collect()\n',
            '\n',
            'print("✓ Memory freed!")'
        ]
    }
    
    nb['cells'].append(cleanup_cell)
    nb['cells'].append(gc_cell)
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    print(f'  ✓ Added cleanup to {nb_path.name}')

# Process all v10 notebooks
notebooks_dir = Path('heiro_v10_refinement/notebooks')
for nb_file in sorted(notebooks_dir.glob('*.ipynb')):
    add_cleanup_to_notebook(nb_file)

print('\n✅ All notebooks updated!')
