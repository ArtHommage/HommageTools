"""
File: homage_tools/nodes/ht_detection_batch_processor.py
Version: 1.2.0
Description: Node for batch processing detected regions with Stable Diffusion upscaling
"""

import math
import os
import copy
import numpy as np
import torch
from PIL import Image
from enum import Enum
import comfy
import comfy.samplers
from typing import List, Tuple, Dict, Any, Optional, Union

# Define version
VERSION = "1.2.0"

#------------------------------------------------------------------------------
# Section 1: Core Configuration and Enums
#------------------------------------------------------------------------------
# Scale mode enum for determining how to handle scaling
class ScaleMode(Enum):
    MAX = "max"  # Scale up to 1024 regardless
    UP = "up"    # Scale up to next closest bucket
    DOWN = "down"  # Scale down to next closest bucket
    CLOSEST = "closest"  # Scale to closest bucket

# Enum for short edge divisibility
class ShortEdgeDivisibility(Enum):
    DIV_BY_8 = 8
    DIV_BY_64 = 64

# USDU Enums for upscaling
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

# Configuration constants
BUCKET_SIZES = [1024, 768, 512]
MAX_BUCKET_SIZE = 1024
MAX_RESOLUTION = 8192

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}

# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}

#------------------------------------------------------------------------------
# Section 2: Utility Functions
#------------------------------------------------------------------------------
def save_debug_image(tensor, stage_name, batch_index=0):
    """
    Save an image tensor to disk with timestamp for debugging.
    Only active when HT_DEBUG_IMAGES environment variable is set to "1".
    """
    # Skip if debugging is disabled
    if os.environ.get("HT_DEBUG_IMAGES", "0") != "1":
        return
    
    # Ensure tensor is a torch tensor and detach it from graph
    if not isinstance(tensor, torch.Tensor):
        return
        
    tensor = tensor.detach().cpu()
    
    # Create debug directory
    os.makedirs("debug_images", exist_ok=True)
    
    # Handle different tensor formats
    if len(tensor.shape) == 4:  # BHWC
        if batch_index < tensor.shape[0]:
            img_tensor = tensor[batch_index]
        else:
            img_tensor = tensor[0]
    else:
        img_tensor = tensor
    
    # Convert to numpy and proper format
    img_np = img_tensor.numpy()
    if img_np.shape[0] == 3 and len(img_np.shape) == 3:  # If CHW format
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # Scale to 0-255 range
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    try:
        # Create and save image
        import time
        from PIL import Image
        img = Image.fromarray(img_np)
        filename = f"debug_images/{time.strftime('%H%M%S')}_{stage_name}.png"
        img.save(filename)
        print(f"[DEBUG] Saved {stage_name} debug image to {filename}")
    except Exception as e:
        print(f"[DEBUG] Error saving debug image: {e}")

def tensor_to_pil(img_tensor, batch_index=0):
    """Convert a tensor to a PIL image."""
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def pil_to_tensor(image):
    """Convert a PIL image to a tensor."""
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image

def process_segs_with_buckets(segs, original_image, scale_mode, short_edge_div, mask_dilation=1.0, bucket_scale_factor=1.0):
    """
    Process SEGS to fit into specified buckets.
    Returns a list of image crops for flexible processing.
    """
    if segs is None or len(segs[1]) == 0:
        # Create a 1x1 black pixel image tensor for empty SEGS
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return [empty_tensor], [], [], [], []
    
    image_crops = []
    crop_regions = []
    target_dimensions = []
    scale_types = []
    bucket_sizes = []
    
    img_height, img_width = original_image.shape[1:3]
    original_img_size = (img_width, img_height)
    
    for seg in segs[1]:
        # Get crop region
        x1, y1, x2, y2 = seg.crop_region
        
        # Calculate target dimensions based on bucket constraints
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Apply mask dilation
        if mask_dilation != 1.0:
            # Calculate expansion
            width_expansion = int(crop_width * (mask_dilation - 1))
            height_expansion = int(crop_height * (mask_dilation - 1))
            
            # Apply expansion evenly
            x1 = max(0, x1 - width_expansion // 2)
            y1 = max(0, y1 - height_expansion // 2)
            x2 = min(img_width, x2 + width_expansion // 2)
            y2 = min(img_height, y2 + height_expansion // 2)
            
            # Update dimensions
            crop_width = x2 - x1
            crop_height = y2 - y1
        
        # Calculate bucket target dimensions
        is_landscape = crop_width >= crop_height
        long_edge = crop_width if is_landscape else crop_height
        short_edge = crop_height if is_landscape else crop_width
        
        # Determine target bucket size based on scale mode
        if scale_mode == ScaleMode.MAX or long_edge > MAX_BUCKET_SIZE:
            target_long_edge = MAX_BUCKET_SIZE
            scale_type = "down" if long_edge > MAX_BUCKET_SIZE else "up"
        elif scale_mode == ScaleMode.UP:
            larger_buckets = [b for b in BUCKET_SIZES if b > long_edge]
            target_long_edge = min(larger_buckets) if larger_buckets else MAX_BUCKET_SIZE
            scale_type = "up"
        elif scale_mode == ScaleMode.DOWN:
            smaller_buckets = [b for b in BUCKET_SIZES if b < long_edge]
            target_long_edge = max(smaller_buckets) if smaller_buckets else min(BUCKET_SIZES)
            scale_type = "down"
        else:  # CLOSEST
            closest_bucket = min(BUCKET_SIZES, key=lambda b: abs(b - long_edge))
            target_long_edge = closest_bucket
            scale_type = "up" if closest_bucket > long_edge else "down"
            
        # Apply bucket scale factor
        target_long_edge = int(target_long_edge * bucket_scale_factor)
            
        # Calculate scaling factor
        scale_factor = target_long_edge / long_edge
        
        # Calculate short edge size
        raw_short_edge = short_edge * scale_factor
        
        # Adjust to be divisible by divisor
        divisor = short_edge_div.value
        adjusted_short_edge = math.ceil(raw_short_edge / divisor) * divisor
        
        # Calculate final dimensions
        if is_landscape:
            target_width = target_long_edge
            target_height = adjusted_short_edge
        else:
            target_width = adjusted_short_edge
            target_height = target_long_edge
        
        # Store metadata
        crop_regions.append((x1, y1, x2, y2))
        target_dimensions.append((target_width, target_height))
        scale_types.append(scale_type)
        bucket_sizes.append(target_long_edge)
        
        # Extract crop
        try:
            cropped_image = original_image[:, y1:y2, x1:x2, :]
            image_crops.append(cropped_image)
        except Exception as e:
            print(f"[ERROR] Crop extraction failed: {e}")
            continue
    
    # Return the crops
    if len(image_crops) > 0:
        save_debug_image(image_crops[0], "process_segs_example_crop")
        return image_crops, crop_regions, target_dimensions, scale_types, bucket_sizes
    else:
        # Fallback empty tensor
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return [empty_tensor], [], [], [], []

def prepare_crop_batch(image_batch, crop_regions, original_image):
    """
    Prepare cropped batch for output with proper formatting.
    """
    # Empty regions check
    if len(crop_regions) == 0:
        print("[WARNING] No crop regions found")
        marker = original_image.clone()
        marker[..., 0] = 1.0  # Red tint
        return marker
    
    # Ensure proper dimensions
    if isinstance(image_batch, list):
        # Convert list to tensor if needed
        if len(image_batch) > 0:
            # Try to stack if shapes match
            shapes = [img.shape for img in image_batch]
            if all(s == shapes[0] for s in shapes):
                image_batch = torch.cat(image_batch, dim=0)
            else:
                # Return first crop as representative sample
                return image_batch[0]
    
    # Format checking
    if len(image_batch.shape) != 4:
        print(f"[WARNING] Unusual batch shape: {image_batch.shape}")
        if len(image_batch.shape) == 3 and image_batch.shape[-1] == 3:
            image_batch = image_batch.unsqueeze(0)  # Add batch dimension
        elif len(image_batch.shape) == 3 and image_batch.shape[0] == 3:
            # Convert CHW to BHWC
            image_batch = image_batch.permute(1, 2, 0).unsqueeze(0)
    
    # Verify batch size matches crop count
    if image_batch.shape[0] != len(crop_regions):
        print(f"[WARNING] Batch size {image_batch.shape[0]} doesn't match crop count {len(crop_regions)}")
    
    save_debug_image(image_batch, "prepared_crop_batch")
    return image_batch

def upscale_regions(image, crop_regions, target_dimensions, model, positive, negative, vae, seed,
                   steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                   mode_type, tile_width, tile_height, mask_blur, tile_padding,
                   seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                   seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
    """
    Upscale individual regions - this is a simplified version that gives the 
    basic process without trying to actually implement the full upscaling.
    """
    # The full implementation would involve:
    # 1. Setting up SD processing for each crop
    # 2. Processing each with proper settings
    # 3. Integrating the upscaler model
    
    # For simplicity, we'll just create a mock processed result
    processed_crops = []
    info_text = f"Processed {len(crop_regions)} regions with the following dimensions:\n"
    
    # Create a basic info string for each region
    for i, (region, target_dim) in enumerate(zip(crop_regions, target_dimensions)):
        x1, y1, x2, y2 = region
        target_w, target_h = target_dim
        region_info = f"Region {i+1}: {x2-x1}x{y2-y1} → {target_w}x{target_h}"
        info_text += region_info + "\n"
        
        # Extract crop
        if i < len(crop_regions):
            crop = image[:, y1:y2, x1:x2, :]
            # For mock purposes, just resize the crop to target dimensions
            from torchvision.transforms.functional import resize
            # Convert to channels-first for resize
            crop_cf = crop.permute(0, 3, 1, 2)
            resized = resize(crop_cf, [target_h, target_w])
            # Convert back to BHWC
            processed = resized.permute(0, 2, 3, 1)
            processed_crops.append(processed)
    
    # Return all processed crops and info
    if len(processed_crops) > 0:
        return processed_crops, info_text
    else:
        return image, "No regions processed"

#------------------------------------------------------------------------------
# Section 3: Main Node Class
#------------------------------------------------------------------------------
class HTDetectionBatchProcessor:
    """
    ComfyUI node that implements detection-based batch processing with bucket support.
    Detects objects in images and processes them to standardized dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        """Define inputs for the node."""
        required = {
            "image": ("IMAGE",),  # Will accept batch of images
            # Sampling Params
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            # Upscale Params
            "upscale_model": ("UPSCALE_MODEL",),
            "mode_type": (list(MODES.keys()),),
            "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            "tile_padding": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            # Seam fix params
            "seam_fix_mode": (list(SEAM_FIX_MODES.keys()),),
            "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            # Detection Params
            "detection_model": ("BBOX_DETECTOR",),
            "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "detection_dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            # Bucket Params
            "scale_mode": ([mode.value for mode in ScaleMode], {"default": "max"}),
            "short_edge_div": ([div.name for div in ShortEdgeDivisibility], {"default": "DIV_BY_8"}),
            "mask_dilation": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
            "use_buckets": ("BOOLEAN", {"default": True, "label": "Use target buckets"}),
            # Misc
            "force_uniform_tiles": ("BOOLEAN", {"default": True}),
            "tiled_decode": ("BOOLEAN", {"default": False}),
            "bucket_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
        }
        
        optional = {
            "segs": ("SEGS", {"default": None}),
            "use_provided_segs": ("BOOLEAN", {"default": False, "label": "Use provided SEGS instead of detecting"}),
            "segs_upscale_separately": ("BOOLEAN", {"default": True, "label": "Upscale SEGS separately"}),
            "detection_upscale_model": ("UPSCALE_MODEL", {"default": None})
        }

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("upscaled_image", "cropped_batch", "scale_info")
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, seed,
            steps, cfg, sampler_name, scheduler, denoise, upscale_model,
            mode_type, tile_width, tile_height, mask_blur, tile_padding,
            seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
            seam_fix_width, seam_fix_padding, detection_model, detection_threshold,
            detection_dilation, crop_factor, drop_size, scale_mode, short_edge_div,
            mask_dilation, use_buckets, force_uniform_tiles, tiled_decode,
            bucket_scale_factor=1.0, segs=None, use_provided_segs=False, 
            segs_upscale_separately=True, detection_upscale_model=None):
        """
        Main upscaling function that processes either segments or whole images.
        """
        # Print version info
        print(f"HTDetectionBatchProcessor v{VERSION} - Processing")
        
        # Store params for later use (if needed)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur
        
        # Convert string enum choices to actual enum values
        scale_mode_enum = ScaleMode(scale_mode)
        short_edge_div_enum = ShortEdgeDivisibility[short_edge_div]

        # For storing scale info - we use a list for joining later with newlines
        scale_info = []
        
        # Get SEGS - either from provided input or by running detection
        working_segs = None
        try:
            if use_provided_segs and segs is not None:
                working_segs = segs
                print(f"[USDU] Using provided SEGS with {len(segs[1]) if segs and len(segs) > 1 else 0} items")
            else:
                # Run detection using provided detector
                print(f"[USDU] Running detection with threshold {detection_threshold}")
                if detection_model is None:
                    print(f"[USDU] ERROR: No detection model provided")
                    raise ValueError("No detection model provided")
                
                # Run detection
                working_segs = detection_model.detect(image, detection_threshold, detection_dilation, crop_factor, drop_size)
                print(f"[USDU] Detection completed, found {len(working_segs[1]) if working_segs and len(working_segs) > 1 else 0} regions")
            
            # Check for empty SEGS
            if working_segs is None or not isinstance(working_segs, tuple) or len(working_segs) < 2 or not working_segs[1]:
                print("[USDU] No valid regions detected")
                # Return empty result for first output, original image for second
                empty_tensor = torch.zeros(1, 1, 1, 3)
                return (empty_tensor, image, "No regions detected")
                
        except Exception as e:
            print(f"[USDU] Detection error: {str(e)}")
            # Create error indicator
            empty_tensor = torch.zeros(1, 1, 1, 3)
            empty_tensor[0, 0, 0, 0] = 1.0  # Red pixel
            return (empty_tensor, image, f"Detection error: {str(e)}")

        # Process SEGS if available
        if working_segs is not None and len(working_segs[1]) > 0:
            print(f"[USDU] Processing detected regions")
            
            # If using buckets, process SEGS with bucket adjustment
            if use_buckets:
                image_batch, crop_regions, target_dimensions, scale_types, bucket_sizes = process_segs_with_buckets(
                    working_segs, image, scale_mode_enum, short_edge_div_enum, mask_dilation, bucket_scale_factor
                )
                
                # Generate scale info for output
                for i, (region, target_dim, scale_type, bucket_size) in enumerate(zip(crop_regions, target_dimensions, scale_types, bucket_sizes)):
                    x1, y1, x2, y2 = region
                    width, height = target_dim
                    scale_info.append(f"Region {i+1}: {x2-x1}x{y2-y1} → {width}x{height} ({scale_type}, bucket: {bucket_size})")
            else:
                # Without buckets, use default processing
                processed_segs = working_segs
                
                # Extract crops from processed SEGS
                from ..modules.detection_processor import segs_to_image_batch
                image_batch, crop_regions = segs_to_image_batch(processed_segs, image)
                
                # Set default 2x target dimensions
                target_dimensions = []
                for x1, y1, x2, y2 in crop_regions:
                    w = x2 - x1
                    h = y2 - y1
                    target_dimensions.append((w * 2, h * 2))
                    
                scale_info.append("Bucket sizing disabled - using 2x upscale")
            
            # Empty SEGS check
            if len(crop_regions) == 0:
                print("[USDU] No valid regions detected after processing")
                empty_tensor = torch.zeros(1, 1, 1, 3)
                return (empty_tensor, image, "No valid regions detected")
            
            # Prepare crops for display
            if isinstance(image_batch, list):
                # We have a list of tensors - create a visualization
                if len(image_batch) > 0:
                    # Use first crop as representative
                    raw_cropped_batch = image_batch[0]
                    save_debug_image(raw_cropped_batch, "cropped_batch_example")
                else:
                    raw_cropped_batch = torch.zeros(1, 1, 1, 3)
            else:
                # We have a batched tensor
                raw_cropped_batch = prepare_crop_batch(image_batch, crop_regions, image)
                save_debug_image(raw_cropped_batch, "cropped_batch_prepared")
            
            # Process the regions with upscaling
            if segs_upscale_separately:
                # Upscale each region individually
                processed_result, upscale_info = upscale_regions(
                    image, crop_regions, target_dimensions, model, positive, negative, vae, seed,
                    steps, cfg, sampler_name, scheduler, denoise, 
                    detection_upscale_model if detection_upscale_model is not None else upscale_model,
                    mode_type, tile_width, tile_height, mask_blur, tile_padding,
                    seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                    seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode
                )
                
                # Return the processed results
                return (processed_result, raw_cropped_batch, "\n".join(scale_info) + "\n" + upscale_info)
            else:
                # Upscale all regions as a batch (simplified for this version)
                # In a complete implementation, this would handle batch processing
                return (raw_cropped_batch, raw_cropped_batch, "\n".join(scale_info) + "\nBatch processing disabled in this version")
        
        # If we get here, no processing was done
        return (image, image, "No processing performed")