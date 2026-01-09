import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Define the same architecture
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))
        x2 = self.pool(x1)
        x3 = torch.relu(self.enc2(x2))
        x4 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
        x5 = torch.relu(self.dec1(x4))
        x6 = self.dec2(x5 + x1)
        return torch.sigmoid(x6)


# Load dataset
D = np.load("data/mri_denoising.npz")
X_clean = torch.tensor(D["clean"][:, None, :, :], dtype=torch.float32)  # (N,1,H,W)
X_noisy = torch.tensor(D["noisy"][:, None, :, :], dtype=torch.float32)  # (N,1,H,W)

# Train/val split, 80/20 train/validate split
idx = torch.randperm(len(X_clean))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

print(f"Length of training dataset: {len(train_idx)}")
print(f"Length of validation dataset: {len(val_idx)}")

train_dl = DataLoader(TensorDataset(X_noisy[train_idx], X_clean[train_idx]),
                      batch_size=32, shuffle=True)
val_dl = DataLoader(TensorDataset(X_noisy[val_idx], X_clean[val_idx]),
                    batch_size=32)

# Autoencoder
model = Denoiser()

# Load the saved weights
model.load_state_dict(torch.load("trained_models/autoencoder_1000epochs_1000images.pt"))    #change the end path if you want to try another model

# Set to evaluation mode
model.eval()

with torch.no_grad():               # disables gradient computation
    recon = model(X_noisy[val_idx[:5]]).squeeze().numpy()        # get denoised images, .squeeze() removes extra dimension, .numpy() converts to numpy array


####################### Plotting images #############################
fig, axes = plt.subplots(2, 3, figsize=(6,6))
for i in range(2):
    axes[i,0].imshow(X_noisy[val_idx[i], 0], cmap="gray")
    axes[i,0].set_title("Noisy")
    axes[i,1].imshow(recon[i], cmap="gray")
    axes[i,1].set_title("Denoised")
    axes[i,2].imshow(X_clean[val_idx[i], 0], cmap="gray")
    axes[i,2].set_title("Clean")
    for j in range(3):
        axes[i,j].axis("off")

plt.tight_layout()
plt.savefig(f"denoising_examples/2x3/denoising_demo.png", dpi=200)
plt.show()      
#####################################################################