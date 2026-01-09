# CNN Denoising of MRI Slices

This project explores convolutional neural networks (CNNs) for denoising MRI images.  
A simple convolutional autoencoder is trained to reconstruct clean MRI slices from synthetically noisy inputs.

The goal is to gain hands-on experience with:
- MRI data handling (NIfTI format)
- Dataset construction from 4D medical images
- Training and evaluating CNN-based autoencoders in PyTorch
- Analysing the effect of training duration on denoising performance

To see examples of denoising refer to denoising_examples/ folder. The figures show noisy, denoised and ground truth images. 3x2 and 5x2 examples are available. Check the file name of the .png for info on epoch and image count used for the model. All 3x2 examples are of the best model of 1000 epochs and 1000 images. 5x2 examples include examples of 10, 100, 1000 epochs, all trained on 1000 images.

To see convergence of MSE for varying epoch numbers see plots/ folder. Check filenames of .png for info on epoch and image count used for the model.

---

## Folder structure 

cnn-denoising/
├── data/
│   ├── EPI_Head_ep2d_bold_moco_10.nii.gz
│   └── mri_denoising.npz
├── scripts/
│   ├── make_dataset.py
│   ├── train_autoencoder.py
│   └── use_trained_autoencoder.py
├── trained_models/
│   ├── autoencoder_10epochs_1000images.pt
│   ├── autoencoder_100epochs_1000images.pt
│   └── autoencoder_1000epochs_1000images.pt
├── plots/
│   └── MSE_vs_Epochs_*.png
├── denoising_examples/
│   └── denoising_demo_*.png
└── README.md


## Dataset

The dataset is constructed from real EPI BOLD MRI data stored in 4D NIfTI format 
(X, Y, Z, time).

**Steps:**
1. Load a 4D EPI BOLD MRI volume
2. Extract the first 100 timepoints (before motion occurs)
3. Select central axial slices from each timepoint
4. Apply robust global intensity normalization using percentiles
5. Add synthetic Gaussian noise to simulate scanner thermal noise
6. Save clean and noisy image pairs in a compressed `.npz` file

This results in **1000 paired 2D MRI slices**.

Dataset generation is handled by: scripts/make_dataset.py


## Model

A lightweight convolutional autoencoder is used:
	•	Encoder: two convolutional layers with max pooling
	•	Decoder: upsampling with bilinear interpolation
	•	Skip connection between early encoder and late decoder layers
	•	Trained using mean squared error (MSE) loss

The architecture is intentionally simple to keep the focus on understanding model behaviour rather than performance tuning. I did experiment with different models initially with small image training count but in the end stuck with this one as it was giving the best results. No checkerboard patterns or over-smoothing.


## Training 

Training is performed using PyTorch:
	•	Optimizer: Adam
	•	Loss: Mean Squared Error (MSE)
	•	Train/validation split: 80/20
	•	Batch size: 32

Training scripts allow easy experimentation with:
	•	Number of epochs (e.g. 10, 100, 1000)
	•	Dataset size
	•	Model convergence behaviour

To train a model: python scripts/train_autoencoder.py

Validation MSE is recorded and plotted versus epoch number.

See plots/ for examples 


## Using a trained model

Previously trained models can be loaded and applied to noisy images for qualitative evaluation.

To run inference with a trained model: python scripts/use_trained_autoencoder.py

You can choose a different model by editing the endpath on line 51 of use_trained_autoencoder.py

3 different models of varying epoch training number are available in /trained_models. All of them are trained on 1000 images. If you train new models with different image or epoch number from the ones existing, they are automatically stored in /trained_models.

This script visualises:
	•	Noisy input
	•	Denoised output
	•	Clean ground truth



## Notes

    •   This project was developed as a learning exercise, combining independent experimentation with the use of modern AI-assisted tools to explore model architectures and training behaviour.
    •	Future extensions could include different noise models, deeper architectures, or quantitative image quality metrics (PSNR, SSIM).