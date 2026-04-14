import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherLazyDataset(Dataset):
    def __init__(self, data_dir, S=48, fut=24):
        """
        data_dir: Path to the folder containing your X_YYYYMMDDHH.pt files.
        """
        self.seq_length = S + fut
        
        # 1. Get all file paths and sort them chronologically.
        # Sorting is critical so your timesteps are in the correct order!
        search_pattern = os.path.join(data_dir, "**", "*.pt") 
        self.file_paths = sorted(glob.glob(search_pattern, recursive=True))
        
        if len(self.file_paths) < self.seq_length:
            raise ValueError("Not enough files to create a single sequence.")

    def __len__(self):
        # The number of valid sliding windows we can make
        return len(self.file_paths) - self.seq_length + 1

    def __getitem__(self, idx):
        # 2. LAZY LOADING: Only load the files for this specific sequence
        window_paths = self.file_paths[idx : idx + self.seq_length]
        
        # Load each file and stack them
        # Resulting shape: (Timesteps, Height, Width, Channels) -> (72, 450, 449, 7)
        tensors = [torch.load(path) for path in window_paths]
        sequence_tensor = torch.stack(tensors, dim=0)
        
        return sequence_tensor