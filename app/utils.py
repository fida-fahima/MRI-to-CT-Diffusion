
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    img_size = 256
    
    base_channels = 56
    channel_multipliers = (1, 2, 4, 6)
    num_res_blocks = 2
    attention_resolutions = (32, 16,)
    dropout = 0.1

    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    eval_ddim_steps = 50 

HU_MIN, HU_MAX = -1000, 2000 

def ct_scaled_to_hu(ct_scaled):
    """Converts a [-1, 1] scaled array back to Hounsfield Units."""
    return ((ct_scaled + 1.0) / 2.0) * (HU_MAX - HU_MIN) + HU_MIN

def mri_normalize(mri_array):
    """Normalizes an MRI slice to [-1, 1] using percentiles."""
    p1, p99 = np.percentile(mri_array, (0.5, 99.5))
    m = np.clip(mri_array, p1, p99)
    m = (m - p1) / (p99 - p1 + 1e-8) 
    m = 2*m - 1.0                   
    return m.astype(np.float32)

def get_inference_transforms(img_size):
    """
    Returns the validation transforms (Resize + ToTensor) for inference.
    A.Normalize is REMOVED as .npy files are already [-1, 1].
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])