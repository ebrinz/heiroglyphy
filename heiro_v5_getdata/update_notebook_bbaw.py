import nbformat
from pathlib import Path

notebook_path = Path('notebooks/01_dataset_assembly.ipynb')
nb = nbformat.read(notebook_path, as_version=4)

# Define the new cell content for loading BBAW data
bbaw_loading_code = """
print("\\nLoading BBAW dataset...")
bbaw_path = Path('../../huggingface/train-00000-of-00001.parquet')

if bbaw_path.exists():
    bbaw_df = pd.read_parquet(bbaw_path)
    print(f"Loaded {len(bbaw_df)} records from BBAW")
    
    # Map columns to match our schema
    # transcription -> transliteration
    # translation -> translation
    # hieroglyphs -> hieroglyphs (new column to preserve)
    
    bbaw_formatted = []
    for _, row in bbaw_df.iterrows():
        record = {
            'transliteration': row['transcription'] if row['transcription'] else "",
            'translation': row['translation'] if row['translation'] else "",
            'source': 'BBAW (HuggingFace)',
            'metadata': {'hieroglyphs': row['hieroglyphs']} if row['hieroglyphs'] else {}
        }
        # Also keep hieroglyphs as a top-level column if we want to use it later easily
        # But for now, putting it in metadata or a separate column in the dataframe is fine.
        # Let's actually add it to the main list and let the dataframe handle the extra column.
        record['hieroglyphs'] = row['hieroglyphs']
        bbaw_formatted.append(record)
        
    data.extend(bbaw_formatted)
    print(f"Added {len(bbaw_formatted)} records from BBAW")
else:
    print(f"BBAW file not found at {bbaw_path}")
"""

# Find the cell where data is loaded (it has "Load the scraped data" comment)
target_cell_index = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and '# Load the scraped data' in cell.source:
        target_cell_index = i
        break

if target_cell_index != -1:
    # Append the BBAW loading code to the existing cell source
    # We insert it before the DataFrame creation
    source = nb.cells[target_cell_index].source
    
    # Split source to insert before df = pd.DataFrame(data)
    parts = source.split('df = pd.DataFrame(data)')
    
    if len(parts) == 2:
        new_source = parts[0] + bbaw_loading_code + '\n\ndf = pd.DataFrame(data)' + parts[1]
        nb.cells[target_cell_index].source = new_source
        print("Successfully injected BBAW loading code.")
    else:
        print("Could not find insertion point 'df = pd.DataFrame(data)'")
else:
    print("Could not find target cell '# Load the scraped data'")

# We also need to update the cleaning function to handle the 'hieroglyphs' column if it exists
# or just ensure it passes through. The current cleaning iterates over rows or applies to 'transliteration'.
# Let's check if we need to update column list display.

# Save the updated notebook
nbformat.write(nb, notebook_path)
print(f"Updated notebook saved to {notebook_path}")
