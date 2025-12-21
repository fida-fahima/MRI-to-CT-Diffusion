
import torch
import numpy as np
import nibabel as nib
import albumentations as A
import io
import os
import gzip 

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from .model import UNet, DDPMDiffusion
from .utils import Config, mri_normalize, ct_scaled_to_hu, get_inference_transforms

app = FastAPI(title="MRI-to-CT Diffusion Model")

config = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ")
if not torch.cuda.is_available():
    print("WARNING: CUDA NOT AVAILABLE. INFERENCE WILL BE EXTREMELY SLOW.")
print("Loading model architecture...")

model = UNet(
    in_channels=1, out_channels=1, condition_channels=1,
    base_channels=config.base_channels, 
    channel_multipliers=config.channel_multipliers,
    num_res_blocks=config.num_res_blocks, 
    attention_resolutions=config.attention_resolutions,
    dropout=config.dropout
)
diffusion = DDPMDiffusion(
    model, config.timesteps, config.beta_start, config.beta_end, device
)

MODEL_WEIGHTS_PATH = "weights/best_model_psnr_full.pt"
print(f"Loading weights from {MODEL_WEIGHTS_PATH}...")
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"FATAL ERROR: Model weights file not found at {MODEL_WEIGHTS_PATH}")
    print("Please place 'best_model_psnr_full.pt' in the 'weights' folder.")
    model = None
else:
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=False)
        
        if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
            print("Loading EMA shadow weights...")
            model.load_state_dict(checkpoint['ema_shadow'])
        elif 'model_state_dict' in checkpoint:
            print("Loading 'model_state_dict' weights...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading weights directly from checkpoint root...")
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print(" Model loaded successfully")

    except Exception as e:
        print(f"FATAL ERROR: Could not load model weights. {e}")
        model = None

transform = get_inference_transforms(config.img_size)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Server error.")
    
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".nii.gz") or filename_lower.endswith(".nii")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .nii or .nii.gz file.")

    print(f"Processing file: {file.filename}")
    
    try:
        contents = await file.read()
        decompressed_bytes = None
        if filename_lower.endswith(".nii.gz"):
            print("Decompressing .nii.gz file...")
            file_obj = io.BytesIO(contents)
            decompressed_bytes = gzip.GzipFile(fileobj=file_obj).read()
            print("Decompression complete.")
        elif filename_lower.endswith(".nii"):
            print("Reading .nii file bytes directly...")
            decompressed_bytes = contents
       
        print("Loading NIfTI image from bytes...")
        nii_file = nib.Nifti1Image.from_bytes(decompressed_bytes)
        print("NIfTI image loaded.")

        mri_data = nii_file.get_fdata().astype(np.float32)
        affine = nii_file.affine
        original_shape = mri_data.shape 
        
        generated_ct_volume = np.zeros(original_shape, dtype=np.float32)
        
        resize_back_transform = A.Resize(original_shape[0], original_shape[1])

        print(f"Generating {original_shape[2]} slices (This will be slow on CPU)...")
        
        for i in range(original_shape[2]):
            mri_slice = mri_data[:, :, i]
            
            mri_slice_norm = mri_normalize(mri_slice)
            
            transformed = transform(image=mri_slice_norm)
            mri_tensor = transformed['image'].unsqueeze(0).to(device) 

            with torch.no_grad():
                fake_ct_tensor = diffusion.ddim_sample(
                    mri_tensor, num_steps=config.eval_ddim_steps
                ) 
            
            fake_ct_256 = fake_ct_tensor.squeeze().cpu().numpy() 
            
            fake_ct_orig_shape = resize_back_transform(image=fake_ct_256)['image']
            
            fake_ct_hu = ct_scaled_to_hu(fake_ct_orig_shape)
            
            generated_ct_volume[:, :, i] = fake_ct_hu

        print("Generation complete. Creating NIfTI file...")
        
        output_nii = nib.Nifti1Image(generated_ct_volume, affine)
        
        output_io = io.BytesIO()
        output_nii.to_file_map({'image': output_io})
        output_io.seek(0)

        return StreamingResponse(
            output_io,
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename=generated_ct_from_{file.filename}"}
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

print("FastAPI app initialized.")