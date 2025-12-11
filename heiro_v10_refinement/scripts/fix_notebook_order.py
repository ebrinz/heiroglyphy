import json

nb_path = 'heiro_v10_refinement/notebooks/03_fusion_v10.2_lexicon.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find the results cell (it's the second-to-last cell before cleanup)
# We need to move it before the cleanup cells
cells = nb['cells']

# Remove the last 2 cells (cleanup markdown + code)
cleanup_cells = cells[-2:]
cells = cells[:-2]

# Remove the results cell (should be last now)
results_cell = cells[-1]
cells = cells[:-1]

# Insert results cell back
cells.append(results_cell)

# Add cleanup cells at the end
cells.extend(cleanup_cells)

nb['cells'] = cells

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=2)

print('✓ Fixed notebook - moved results before cleanup')
