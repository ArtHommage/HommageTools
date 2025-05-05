"""
Debug utilities for HTDetectionBatchProcessor
"""
import os
import time
import numpy as np
import torch
from PIL import Image

def save_debug_image(tensor, stage_name, batch_index=0):
    # Force disable debugging
    if os.environ.get("HT_DEBUG_IMAGES", "0") != "1":
        return
    """
    Save an image tensor to disk with timestamp for debugging.
    
    Args:
        tensor: Image tensor to save
        stage_name: Name of the processing stage
        batch_index: Index in the batch to save (default 0)
    """
    # Skip if debugging is disabled
    if not os.environ.get("HT_DEBUG_IMAGES", "0") == "1":
        return
    
    # Create debug directory if it doesn't exist
    os.makedirs("debug_images", exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%H%M%S")
    
    # Handle None or empty tensor
    if tensor is None or tensor.numel() == 0:
        print(f"[DEBUG] Cannot save {stage_name} - empty tensor")
        return
        
    # Ensure we're working with a detached copy
    tensor = tensor.detach().cpu()
    
    # Handle different tensor formats
    if len(tensor.shape) == 4:
        # Batch of images [B,H,W,C] or [B,C,H,W]
        if batch_index < tensor.shape[0]:
            img_tensor = tensor[batch_index]
        else:
            img_tensor = tensor[0]
    else:
        # Single image
        img_tensor = tensor
    
    # Convert to numpy
    img_tensor = img_tensor.numpy()
    
    # Handle channel dimension ordering
    if len(img_tensor.shape) == 3:
        if img_tensor.shape[0] == 3:  # [C,H,W] format
            img_tensor = np.transpose(img_tensor, (1, 2, 0))
        # else assume it's already [H,W,C]
    
    # Scale values to 0-255 range
    if img_tensor.max() <= 1.0 and img_tensor.dtype != np.uint8:
        img_tensor = np.clip(img_tensor * 255, 0, 255).astype(np.uint8)
    else:
        img_tensor = np.clip(img_tensor, 0, 255).astype(np.uint8)
    
    try:
        # Create PIL image
        image = Image.fromarray(img_tensor)
        
        # Save the image with timestamp
        filename = f"debug_images/{timestamp}_{stage_name}.png"
        image.save(filename)
        print(f"[DEBUG] Saved {stage_name} image to {filename}")
    except Exception as e:
        print(f"[DEBUG] Error saving {stage_name} image: {e}")
        print(f"[DEBUG] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"[DEBUG] Values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")