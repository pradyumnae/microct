import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
import segmentation_models_pytorch as smp

def run_inference(
    img_dir,
    output_mask_dir,
    output_csv_path,
    model_weights_path,
    num_classes=5,
    patch_size=320,
    stride=160
):
    """
    Run inference on a folder of grayscale CT images using the trained FPN (MiT-B4) model.
    """
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # Ensure output directories exist
    os.makedirs(output_mask_dir, exist_ok=True)

    # Class names mapping
    CLASS_NAMES = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]

    # 2. Re-instantiate the Model Model Structure
    print("Initializing Model...")
    model_eval = smp.FPN(
        encoder_name="mit_b4",
        encoder_weights=None,       # No need to download internet weights for inference
        in_channels=1,              # Grayscale inputs
        classes=num_classes,
    )

    # 3. Load Trained Weights
    print(f"Loading weights from: {model_weights_path}")
    checkpoint = torch.load(model_weights_path, map_location=device)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    model_eval = model_eval.to(device)
    model_eval.eval()

    # 4. Find all Images
    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
    
    print(f"Found {len(img_files)} images to process.")

    results = []

    # 5. Run Inference Loop
    for fname in tqdm(img_files, desc="Inference"):
        img = np.array(Image.open(os.path.join(img_dir, fname)))
        
        # Handle RGB → grayscale if someone passes RGB images by mistake
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8).copy()
        
        img_t = torch.from_numpy(img.copy()).float().unsqueeze(0).unsqueeze(0)
        
        # Z-score standardization on non-zero tissue pixels (MUST MATCH TRAINING!)
        valid_pixels = img_t[img_t > 0]
        if len(valid_pixels) > 0:
            mean, std = valid_pixels.mean(), valid_pixels.std()
            if std > 1e-5:
                img_t = (img_t - mean) / std

        # Pad the image to be a multiple of 32 (required by FPN scaling)
        h, w = img.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')

        img_t = img_t.to(device)
        
        # --- SLIDING WINDOW INFERENCE TO PREVENT OOM ---
        # Splitting large images into patches and averaging the overlaps
        _, _, H_t, W_t = img_t.shape
        pred_prob_accum = torch.zeros((1, num_classes, H_t, W_t), device=device, dtype=torch.float32)
        count_accum     = torch.zeros((1, 1, H_t, W_t), device=device, dtype=torch.float32)
        
        with torch.no_grad():
            # Use mixed precision (bfloat16) to save memory if on GPU, otherwise standard float
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype):
                # Iterate through grid
                for py in range(0, H_t, stride):
                    for px in range(0, W_t, stride):
                        y1 = min(py, H_t - patch_size)
                        y2 = y1 + patch_size
                        x1 = min(px, W_t - patch_size)
                        x2 = x1 + patch_size
                        
                        patch = img_t[:, :, y1:y2, x1:x2]
                        
                        # 1. Original Prediction
                        p_orig = model_eval(patch).softmax(dim=1)
                        
                        # Test Time Augmentation (TTA) for robustness
                        # 2. Horizontal Flip
                        p_hflip = model_eval(torch.flip(patch, dims=[3]))
                        p_hflip = torch.flip(p_hflip.softmax(dim=1), dims=[3])
                        
                        # 3. Vertical Flip
                        p_vflip = model_eval(torch.flip(patch, dims=[2]))
                        p_vflip = torch.flip(p_vflip.softmax(dim=1), dims=[2])
                        
                        # Average the TTA predictions
                        p_avg = (p_orig + p_hflip + p_vflip) / 3.0
                        
                        # Accumulate probabilities in the main image map
                        pred_prob_accum[:, :, y1:y2, x1:x2] += p_avg
                        count_accum[:, :, y1:y2, x1:x2] += 1.0
                        
        # Average the overlaps where patches met
        pred_prob = pred_prob_accum / count_accum
        
        # Take the argmax to get the final class prediction
        pred_cls = torch.argmax(pred_prob, dim=1).squeeze().cpu().numpy()
                
        # Crop back to exact original image size if padding was added earlier
        if pad_h > 0 or pad_w > 0:
            pred_cls = pred_cls[:h, :w]

        # 6. Save Output Mask
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_mask_dir, f"{stem}_pred.png")
        Image.fromarray(pred_cls.astype(np.uint8)).save(out_path)
        
        # 7. Calculate Tissue Fractions
        total = pred_cls.size
        row = {'filename': fname}
        for c, name in enumerate(CLASS_NAMES):
            count = int(np.sum(pred_cls == c))
            row[f"{name}_pixels"] = count
            row[f"{name}_fraction"] = round(count / total, 6)
            
        # Optional: Define porosity as Air_Space fraction
        row['porosity'] = row['Air_Space_fraction']
        results.append(row)

    # 8. Save Dataset Analytics CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    print(f"\nInference Complete!")
    print(f"Masks saved to: {output_mask_dir}")
    print(f"Data saved to:  {output_csv_path}")

if __name__ == "__main__":
    # =========================================================================
    # USER CONFIGURATION: Update these paths for your local machine!
    # =========================================================================
    
    # The folder containing the images you want to segment
    INPUT_IMAGES_DIR = "./sample_images" 
    
    # Path to the pretrained model weights provided to you
    MODEL_WEIGHTS = "./best_model.pth"
    
    # Where you want the segmented masks to drop
    OUTPUT_MASKS_DIR = "./output_masks"
    
    # Where you want the fraction/porosity measurements saved
    OUTPUT_CSV_FILE = "./output_fraction_results.csv"
    
    # =========================================================================
    
    # Check that paths exist before running
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Please create the directory '{INPUT_IMAGES_DIR}' and put images inside it.")
        exit(1)
        
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Could not find model weights at '{MODEL_WEIGHTS}'.")
        exit(1)

    # Run the pipeline
    run_inference(
        img_dir=INPUT_IMAGES_DIR,
        output_mask_dir=OUTPUT_MASKS_DIR,
        output_csv_path=OUTPUT_CSV_FILE,
        model_weights_path=MODEL_WEIGHTS
    )
