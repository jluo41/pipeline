import json
import os
from pathlib import Path

def clean_notebook(notebook_path):
    """
    Clean output cells from a single Jupyter notebook.
    
    Args:
        notebook_path (str or Path): Path to the notebook file
    
    Returns:
        bool: True if cleaning was successful, False otherwise
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as file:
            notebook = json.load(file)
        
        # Keep track if we made any changes
        cleaned = False
        
        # Clean output from all cells
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                if 'outputs' in cell and cell['outputs']:
                    cell['outputs'] = []
                    cleaned = True
                if 'execution_count' is not None:
                    cell['execution_count'] = None
                    cleaned = True
        
        # Save the cleaned notebook only if changes were made
        if cleaned:
            with open(notebook_path, 'w', encoding='utf-8') as file:
                json.dump(notebook, file, indent=1)
            print(f"Cleaned: {notebook_path}")
        else:
            print(f"No outputs to clean in: {notebook_path}")
            
        return True
        
    except Exception as e:
        print(f"Error cleaning {notebook_path}: {str(e)}")
        return False

def clean_notebooks_in_folder(folder_path, recursive=True):
    """
    Clean output cells from all Jupyter notebooks in a folder.
    
    Args:
        folder_path (str): Path to the folder containing notebooks
        recursive (bool): Whether to search for notebooks in subfolders
    
    Returns:
        tuple: (number of notebooks processed, number of notebooks with errors)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Define the pattern for finding notebooks
    pattern = '**/*.ipynb' if recursive else '*.ipynb'
    
    # Initialize counters
    processed = 0
    errors = 0
    
    # Process all notebooks
    for notebook_path in folder_path.glob(pattern):
        # Skip checkpoint files
        if '.ipynb_checkpoints' in str(notebook_path):
            continue
            
        if clean_notebook(notebook_path):
            processed += 1
        else:
            errors += 1
    
    return processed, errors

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean output cells from Jupyter notebooks')
    parser.add_argument('folder', help='Folder containing notebooks to clean')
    parser.add_argument('--no-recursive', action='store_true',
                      help='Do not search in subfolders')
    
    args = parser.parse_args()
    
    try:
        processed, errors = clean_notebooks_in_folder(
            args.folder, 
            recursive=not args.no_recursive
        )
        print(f"\nSummary:")
        print(f"Notebooks processed: {processed}")
        if errors > 0:
            print(f"Notebooks with errors: {errors}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)