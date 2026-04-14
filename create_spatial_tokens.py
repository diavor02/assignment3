import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import WeatherLazyDataset

BATCH_SIZE = 128

class SpatialTokenExtractor(nn.Module):
    def __init__(self, cnn_net, d_model=64, S=48, horizon=24, downsampled_h=32, downsampled_w=32):
        super().__init__()
        self.cnn = cnn_net
        self.d_model = d_model
        
        # P = Number of Spatial Patches (Tokens) per timestep
        self.P = downsampled_h * downsampled_w
        self.T = S + horizon  # Total sequence length

        # Embeddings (Learnable Parameters) as shown in the diagram
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.P, d_model))
        self.timestep_embed = nn.Parameter(torch.randn(1, self.T, 1, d_model))

    def forward(self, x):
        """
        Expects input shape: (B, T, H, W, C)
        Returns unflattened sequence: (B, T, P, d_model)
        """
        B, T, H, W, C = x.shape

        # 1. CNN Downsample
        # Permute to (B, T, C, H, W) and fold T into B for Conv2D
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(B * T, C, H, W)
        
        # Shape becomes (B*T, d_model, downsampled_h, downsampled_w)
        tokens = self.cnn(x) 

        # 2. Extract Spatial Tokens
        # Flatten spatial dimensions H and W into P patches
        tokens = tokens.view(B, T, self.d_model, self.P) # (B, T, d_model, P)
        tokens = tokens.permute(0, 1, 3, 2)              # (B, T, P, d_model)

        # 3. Inject Positional Encodings
        tokens = tokens + self.spatial_embed + self.timestep_embed

        # 4. Return BEFORE flattening so you can add the Tabular Tokens later
        return tokens

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

def create_spatial_tokens(cnn, original_input_channels_size=7, original_h=450, original_w=449):
    dummy_input = torch.zeros(1, original_input_channels_size, original_h, original_w) 
    dummy_output = cnn(dummy_input)
    _, _, downsampled_h_size, downsampled_w_size = dummy_output.shape

    print(f"Original sizes: {original_h}x{original_w}")
    print(f"Downsampled sizes: {downsampled_h_size}x{downsampled_w_size}")

    P = downsampled_h_size * downsampled_w_size

    token_extractor = SpatialTokenExtractor(
        cnn_net=cnn,
        d_model=64,
        S=48,
        horizon=24,
        downsampled_h=downsampled_h_size,
        downsampled_w=downsampled_w_size
    )

    return token_extractor, P


# ----------------------------------------------------------------------------------------------------------------
PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data/"
BATCH_SIZE = 128
S = 48
fut = 24
d_model = 64
original_h = 450
original_w = 449
in_channels = 7

cnn_module = WeatherCNN()

token_extractor_module, P = create_spatial_tokens(cnn_module)

# ------------------------------------------------------- Actual implementation--------------------------------------

# Initialize the lazy dataset
dataset = WeatherLazyDataset(data_dir=PATH, S=S, fut=fut)

# Wrap it in a DataLoader
# num_workers > 0 allows the data loading to happen in the background
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Test it out in a training loop
for batch in dataloader:
    print(f"Batch shape: {batch.shape}") 
    assert batch.shape == (BATCH_SIZE, S + fut, original_h, original_w, in_channels)
    # Expected output: torch.Size([128, 72, 450, 449, 7])
    
    spatial_tokens = token_extractor_module(batch)
    assert spatial_tokens.shape == (BATCH_SIZE, S + fut, P, d_model)

    # --- THIS IS WHERE THE REST OF YOUR ARCHITECTURE GOES ---
    
    # 3. Load your tabular data for this exact same time window
    # y_hist, c_hist, c_future = get_tabular_data_for_batch(...)
    
    # 4. Pass EVERYTHING into the HybridTokenCombiner we wrote earlier
    # final_sequence = combiner(y_hist, c_hist, c_future, spatial_tokens)
    
    # 5. Pass to Transformer -> Calculate Loss -> Backpropagate


# ---------------------------------------------------------Test---------------------------------------------------------
# 2. Create a dummy dataset batch to pass through the module
# Shape: (Batch_Size, Timesteps, Height, Width, Channels)
dummy_batch = torch.zeros(BATCH_SIZE, S + fut, 450, 449, 7)

# 3. Run the forward pass to get the actual token tensors
spatial_tokens = token_extractor_module(dummy_batch)

# 4. Now assert the shape of the resulting tensor
assert spatial_tokens.shape == (BATCH_SIZE, S + fut, P, d_model)

print(f"Successfully generated spatial tokens.")
print(f"Spatial Tokens Tensor Shape: {spatial_tokens.shape}")