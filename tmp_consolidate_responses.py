import os
import json
import duckdb
import pandas as pd
from pathlib import Path
from deval.utils import storage_path

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary by concatenating keys with separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def is_sensitive_file(file_path):
    """Check if the file contains sensitive information."""
    sensitive_paths = ['key', 'private', 'secret', 'mnemonic', 'wallet']
    return any(sensitive in file_path.lower() for sensitive in sensitive_paths)

def consolidate_responses():
    # Get the storage path
    storage_dir = Path(storage_path)
    results_dir = storage_dir / 'results'
    
    # List to store all responses
    all_responses = []
    
    # Only process files in the results directory
    if not results_dir.exists():
        print(f"Results directory not found at {results_dir}")
        return
        
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Skip sensitive files
                if is_sensitive_file(file_path):
                    print(f"Skipping sensitive file: {file_path}")
                    continue
                    
                try:
                    with open(file_path, 'r') as f:
                        response_data = json.load(f)
                        
                        # Flatten the params field if it exists
                        if 'params' in response_data:
                            flattened_params = flatten_dict(response_data['params'])
                            # Remove the original params and add flattened version
                            del response_data['params']
                            response_data.update(flattened_params)
                        
                        # Add source file information
                        response_data['source_file'] = file_path
                        all_responses.append(response_data)
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
    
    # Create output directory if it doesn't exist
    output_dir = storage_dir / 'consolidated'
    output_dir.mkdir(exist_ok=True)
    
    # Save consolidated responses as JSON
    output_file_json = output_dir / 'all_responses.json'
    with open(output_file_json, 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    # Convert to DataFrame and save as parquet using duckdb
    df = pd.DataFrame(all_responses)
    output_file_parquet = output_dir / 'all_responses.parquet'
    
    # Create a DuckDB connection
    con = duckdb.connect()
    
    # Register the DataFrame as a table
    con.register('responses', df)
    
    # Save to parquet
    con.execute(f"COPY responses TO '{output_file_parquet}' (FORMAT PARQUET)")
    
    # Close the connection
    con.close()
    
    print(f"Consolidated {len(all_responses)} responses into:")
    print(f"- JSON: {output_file_json}")
    print(f"- Parquet: {output_file_parquet}")

if __name__ == "__main__":
    consolidate_responses() 