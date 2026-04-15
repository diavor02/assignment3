import os
import glob
import torch
from torch.utils.data import Dataset

class WeatherLazyDataset(Dataset):
    def __init__(self, data_dir, S=48, horizon=24,
                 original_h=450, original_w=449, in_channels=7):
        self.seq_length = S + horizon
        self.original_h = original_h
        self.original_w = original_w
        self.in_channels = in_channels

        search_pattern = os.path.join(data_dir, "**", "*.pt")
        self.file_paths = sorted(glob.glob(search_pattern, recursive=True))

        if len(self.file_paths) < self.seq_length:
            raise ValueError("Not enough files to create a single sequence.")

    def __len__(self):
        return len(self.file_paths) - self.seq_length + 1

    def __getitem__(self, idx):
        window_paths = self.file_paths[idx : idx + self.seq_length]

        tensors = [
            torch.load(path, map_location="cpu")
            for path in window_paths
        ]
        sequence_tensor = torch.stack(tensors, dim=0)

        assert sequence_tensor.shape == (
            self.seq_length,
            self.original_h,
            self.original_w,
            self.in_channels
        ), f"Got {sequence_tensor.shape}"

        return sequence_tensor


class TabularLazyDataset(Dataset):
    def __init__(self, tensor_dir="./tensors"):
        self.c_future = torch.load(f"{tensor_dir}/c_future.pt", map_location="cpu")
        self.c_hist = torch.load(f"{tensor_dir}/c_hist.pt", map_location="cpu")
        self.y_future = torch.load(f"{tensor_dir}/y_future.pt", map_location="cpu")
        self.y_hist = torch.load(f"{tensor_dir}/y_hist.pt", map_location="cpu")

        assert len(self.c_future) == len(self.c_hist) == len(self.y_hist) == len(self.y_future)

    def __len__(self):
        return len(self.c_future)

    def __getitem__(self, idx):
        return (
            self.y_hist[idx],
            self.y_future[idx],
            self.c_hist[idx],
            self.c_future[idx],
        )
