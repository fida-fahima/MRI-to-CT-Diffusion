# MRI-to-CT Translation using Hybrid Diffusion Model

##  Project Overview
This project implements a **Hybrid Denoising Diffusion Probabilistic Model (DDPM)** to perform paired image-to-image translation from **MRI (Magnetic Resonance Imaging)** to **CT (Computed Tomography)** scans.

Accurate MRI-to-CT synthesis is critical in radiation therapy planning, allowing for dose calculation using only MRI scans (MRI-only workflow). This reduces patient radiation exposure and eliminates the costs associated with acquiring separate CT scans.

##  Key Innovation: Hybrid Loss Architecture
Unlike standard Diffusion models that only learn to predict noise, this project utilizes a **Hybrid Loss Function** to ensure both textural realism and anatomical accuracy.

The model optimizes two objectives simultaneously:
1.  **Noise Loss (DDPM):** Ensures the model learns the diffusion process to generate high-frequency details and realistic textures.
2.  **Pixel Consistency Loss (L1):** Enforces structural consistency between the generated CT and the ground truth. This acts as a constraint to prevent the "hallucinations" common in pure generative models, ensuring the anatomy remains faithful to the input MRI.

## Dataset
* **Source:** [SynthRAD2023 Dataset](https://synthrad2023.grand-challenge.org/) (Brain Anatomy)
* **Modality:** Paired MRI and CT brain scans.
* **Preprocessing:** Slices were resized to 256x256, normalized, and augmented using `Albumentations` (Shift, Scale, Rotate, Brightness).

## Tech Stack
* **Framework:** PyTorch
* **Architecture:** UNet with Self-Attention & Residual Blocks (conditioned on MRI input).
* **Scheduler:** Cosine Annealing with Warmup.
* **Metrics:** PSNR, SSIM, MAE, RMSE, PCC, LPIPS, FID.

## Performance
The model was evaluated on unseen test patients, achieving high structural fidelity. The results of three metrices are given below:

* **PSNR:** **24.92 dB** 
* **SSIM:** **0.84** 
* **FID:** **68.35** 
