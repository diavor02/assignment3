import os

def count_files(directory="/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data") -> int:
    """
    Recursively counts the number of files within a directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        Total number of files found recursively
        
    Raises:
        ValueError: If the path doesn't exist or isn't a directory
    """
    if not os.path.exists(directory):
        raise ValueError(f"Path does not exist: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")

    count = 0
    for entry in os.scandir(directory):
        if entry.is_file():
            count += 1
        elif entry.is_dir():
            count += count_files(entry.path)
    return count

print(count_files() - count_files("/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/2024"))