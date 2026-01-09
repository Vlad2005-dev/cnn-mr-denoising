import numpy as np
import nibabel as nib       #lib to read NIfTI file format

nii = nib.load("data/EPI_Head_ep2d_bold_moco_10.nii.gz")
V = nii.get_fdata().astype(np.float32)    #extracts 4D data (X, Y, Z, time)
                                            # get_fdata() pulls out pixel values as numpy array
                                            # astype(np.float32) converts them to 32 bit floating point numbers which is standard for neural networks
                                            # shape of V is (64, 64, 30, 300), so 300 time slices

# Take first 100 time slices
V = V[..., :100]  # shape is now (64, 64, 30, 100)    

####################################################################################################################################################
### You can change the number above on line 11 to change the number of time slices taken, BUT DONT GO ABOVE 146 as that's where movement happens ###
####################################################################################################################################################

# Global robust scaling using percentiles (avoids outliers)
vmin, vmax = np.percentile(V, (1, 99))  # for mri each pixel is just an intensity value
V = np.clip(V, vmin, vmax)              # clips all pixels so they fall between vmin and vmax
V = (V - vmin) / (vmax - vmin + 1e-6)   # normalization to keep intensity values between [0,1]

slices = []
# Loop over all 100 time slices
for t in range(V.shape[3]):
    # For each time slice, extract the middle Z slices (10-19)
    for z in range(10, V.shape[2]-10):
        slices.append(V[:, :, z, t])    # get 2D slice at position (X, Y) for z-level z at time t

X_clean = np.stack(slices, axis=0)  # (N,H,W), stacks all 2D slices into a single array
                                    # X_clean.shape is (1000, 64, 64). 100 time slices × 10 z-slices each = 1000 total 2D images

noise_std = 0.06   # slightly gentler than 0.1
rng = np.random.default_rng(0)      # creating random number generator with seed = 0 for reproducibility 
X_noisy = X_clean + noise_std * rng.standard_normal(X_clean.shape, dtype=np.float32)    
# adding gaussian noise scaled by noise_std, this is realistic as it mimics thermal noise from scanners 
# which is also gaussian due to central limit theorem (lots of individual electrons jiggling small random amount)
X_noisy = np.clip(X_noisy, 0, 1)    # clipping dataset back to 0 and 1, just in case extra noise pushed values outside 0 and 1

np.savez_compressed("data/mri_denoising.npz", clean=X_clean.astype(np.float32), noisy=X_noisy.astype(np.float32))
print("Saved:", X_clean.shape)
