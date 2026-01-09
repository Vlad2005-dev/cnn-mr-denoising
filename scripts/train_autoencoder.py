import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x1 = torch.relu(self.enc1(x))      # low-level features
        x2 = self.pool(x1)
        x3 = torch.relu(self.enc2(x2))
        x4 = torch.nn.functional.interpolate(x3, scale_factor=2, mode="bilinear")
        x5 = torch.relu(self.dec1(x4))
        x6 = self.dec2(x5 + x1)             # SKIP CONNECTION
        return torch.sigmoid(x6)

# Load dataset
D = np.load("data/mri_denoising.npz")       # <class 'numpy.lib.npyio.NpzFile'>
X_clean = torch.tensor(D["clean"][:, None, :, :], dtype=torch.float32)  # (N,1,H,W), <class 'torch.Tensor'>
X_noisy = torch.tensor(D["noisy"][:, None, :, :], dtype=torch.float32)  # (N,1,H,W), <class 'torch.Tensor'>

# Train/val split, 80/20 train/validate split
idx = torch.randperm(len(X_clean))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

print(f"Length of training dataset: {len(train_idx)}")
print(f"Length of validation dataset: {len(val_idx)}")
n_images = len(X_clean)
print(f"The total length is: {n_images}")

train_dl = DataLoader(TensorDataset(X_noisy[train_idx], X_clean[train_idx]),
                      batch_size=32, shuffle=True)
val_dl = DataLoader(TensorDataset(X_noisy[val_idx], X_clean[val_idx]),
                    batch_size=32)

# Autoencoder
model = Denoiser()


#####################################################################
####################### Start Training ##############################
#####################################################################

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

n_epochs = 10

MSE = np.zeros(n_epochs)
epochs = np.arange(n_epochs)

for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():           # disables gradient computation (unnecessary during validation, saves memory)
        for xb, yb in val_dl:
            val_loss += loss_fn(model(xb), yb).item()
    val_loss /= len(val_dl)

    MSE[epoch] = val_loss
    print(f"epoch {epoch+1}: val MSE = {val_loss:.4f}")

torch.save(model.state_dict(), f"trained_models/autoencoder_{n_epochs}epochs_{n_images}images.pt")       # saving the model weights for future use

#####################################################################
####################### End of Training #############################
#####################################################################



model.eval()                        # switch model to evaluation mode
with torch.no_grad():               # disables gradient computation
    recon = model(X_noisy[val_idx[:5]]).squeeze().numpy()        # get denoised images, .squeeze() removes extra dimension, .numpy() converts to numpy array



####################### Plotting noisy + denoised + clean images #############################
fig, axes = plt.subplots(5, 3, figsize=(6,6))
for i in range(5):
    axes[i,0].imshow(X_noisy[val_idx[i], 0], cmap="gray")
    axes[i,0].set_title("Noisy")
    axes[i,1].imshow(recon[i], cmap="gray")
    axes[i,1].set_title("Denoised")
    axes[i,2].imshow(X_clean[val_idx[i], 0], cmap="gray")
    axes[i,2].set_title("Clean")
    for j in range(3):
        axes[i,j].axis("off")

plt.tight_layout()
plt.savefig(f"denoising_examples/5x3/denoising_demo_{n_epochs}epochs_{n_images}images.png", dpi=200)
plt.show()
#####################################################################




####################### Plotting MSE vs Epochs #######################
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fig.suptitle(
    f"Validation MSE vs Epochs ({n_epochs} epochs, {n_images} images)",
    fontsize=14
)

# Left plot: linear scale
axes[0].plot(epochs, MSE)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation MSE")

# Right plot: log-log scale
axes[1].loglog(epochs, MSE)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation MSE")
axes[1].set_title("log-log")

plt.tight_layout()
plt.savefig(f"plots/MSE_vs_Epochs_{n_epochs}epochs_{n_images}images.png", dpi=200)
plt.show()
#####################################################################
