import json

nb_path = 'heiro_v10_refinement/notebooks/03_fusion_v10.2_lexicon.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Add cleanup cells
cleanup_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': ['## 7. Cleanup Memory']
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

print('✓ Added cleanup cell to notebook')
