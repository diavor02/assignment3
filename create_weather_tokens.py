import torch
import torch.nn as nn

from helper import build_file_list
from tokens import SpatialTokenExtractor

PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/"

class WeatherCNN(nn.Module):
    def __init__(self, in_channels=7, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, S=48, horizon=24, mean=None, std=None):
        self.files = file_list
        self.S = S
        self.horizon = horizon
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files) - (self.S + self.horizon)

    def __getitem__(self, idx):
        seq_files = self.files[idx : idx + self.S + self.horizon]

        weather_seq = []
        for path in seq_files:
            x = torch.load(path).float()  # (H, W, C)
            weather_seq.append(x)

        x = torch.stack(weather_seq)  # (T, H, W, C)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        return x

def compute_stats_from_files(file_list):
    sum_ = 0
    sum_sq = 0
    count = 0

    for path in file_list:
        x = torch.load(path).float()  # (H, W, C)

        sum_ += x.sum(dim=(0, 1))
        sum_sq += (x ** 2).sum(dim=(0, 1))
        count += x.shape[0] * x.shape[1]

    mean = sum_ / count
    result = sum_sq / count - mean ** 2
    std = torch.sqrt(torch.as_tensor(result))

    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)

    return mean, std

files = build_file_list(PATH + "weather_data/")

print(len(files))
print(files[0])
print(files[-1])

# compute stats
mean, std = compute_stats_from_files(files)
mean = mean.view(1, 1, 1, -1)
std = std.view(1, 1, 1, -1)

# dataset
dataset = WeatherDataset(files, mean=mean, std=std)

# --- Example Usage ---

# 1. Instantiate your CNN
cnn_module = WeatherCNN()

# 2. Run a dynamic dummy pass to get the downsampled sizes
# Use your ORIGINAL spatial dimensions here (e.g., 450x449)
original_h, original_w = 450, 449
dummy_input = torch.zeros(1, 7, original_h, original_w) 

# Pass it through the CNN and inspect the output shape
dummy_output = cnn_module(dummy_input)
_, _, downsampled_h_size, downsampled_w_size = dummy_output.shape

print(f"Original sizes: {original_h}x{original_w}")
print(f"Downsampled sizes: {downsampled_h_size}x{downsampled_w_size}")

# 3. Create the extractor dynamically
token_extractor = SpatialTokenExtractor(
    cnn_net=cnn_module,
    d_model=64,
    S=48,
    horizon=24,
    downsampled_h=downsampled_h_size,
    downsampled_w=downsampled_w_size
)

# 4. Test with dummy dataloader output
# CRITICAL FIX: Use the ORIGINAL dimensions for the input batch
dummy_batch = torch.randn(16, 72, original_h, original_w, 7) # (B, T, H, W, C)

# Pass it through the extractor
transformer_ready_tokens = token_extractor(dummy_batch)

print(f"Transformer Input Shape: {transformer_ready_tokens.shape}")