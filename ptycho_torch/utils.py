from dataclasses import asdict, is_dataclass
import os
import torch

# Helper function to convert dataclass to JSON serializable dict, filtering Tensors

def config_to_json_serializable_dict(config_obj):
    """
    Convert a config dataclass to a JSON-serializable dictionary.

    Tensor payloads are deliberately omitted because they are not configuration
    identity and have their own persistence boundary.
    """
    if not is_dataclass(config_obj):
        raise TypeError(f"Expected a dataclass instance, got {type(config_obj)}")

    d = asdict(config_obj)
    serializable_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"Warning: Skipping field '{k}' of type torch.Tensor during MLflow param logging.")
            continue # Skip tensors
        # Add checks here for other non-serializable types if needed
        serializable_dict[k] = v
    return serializable_dict

def remove_all_files(directory_path):
    """
    Remove all files in the specified directory without removing the directory itself.
    
    Args:
        directory_path (str): Path to the directory whose files should be removed
        
    Returns:
        int: Number of files removed
    """
    count = 0
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return count
        
    # Loop through all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        # Check if it's a file and remove it
        if os.path.isfile(item_path):
            os.remove(item_path)
            count += 1
            
    print(f"Removed {count} files from '{directory_path}'")
    return count
