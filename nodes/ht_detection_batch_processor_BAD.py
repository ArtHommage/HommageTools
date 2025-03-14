"""
File: homage_tools/nodes/ht_detection_batch_processor.py
Version: 1.3.3
Description: Node for detecting regions, applying mask dilation, and processing with model-based upscaling/downscaling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
import math
import logging
import time
import traceback  # Added for better error tracking
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2

# ComfyUI imports
import comfy
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy_extras.nodes_upscale_model as model_upscale

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.3.3"  # Updated version for the fixes
MAX_RESOLUTION = 8192
# Standard bucket sizes for auto-scaling
STANDARD_BUCKETS = [512, 768, 1024]

#------------------------------------------------------------------------------
# Section 3: Helper Functions
#------------------------------------------------------------------------------
def get_nearest_divisible_size(size: int, divisor: int) -> int:
    """Calculate nearest size divisible by divisor."""
    return ((size + divisor - 1) // divisor) * divisor

def calculate_target_size(bbox_size: int, mode: str = "Scale Closest") -> int:
    """Calculate target size based on standard buckets."""
    print(f"Calculating target size for {bbox_size} using mode: {mode}")
    
    if mode == "Scale Closest":
        target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
    elif mode == "Scale Up":
        target = next((x for x in STANDARD_BUCKETS if x >= bbox_size), STANDARD_BUCKETS[-1])
    elif mode == "Scale Down":
        target = next((x for x in reversed(STANDARD_BUCKETS) if x <= bbox_size), STANDARD_BUCKETS[0])
    else:  # Scale Max
        target = STANDARD_BUCKETS[-1]
    
    print(f"Selected bucket size: {target}")
    return target

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array with proper format."""
    return tensor.detach().cpu().numpy()

def numpy_to_tensor(array):
    """Convert numpy array to tensor."""
    return torch.from_numpy(array).float()

def dilate_mask_opencv(mask_np, dilation_value):
    """Dilate mask using OpenCV."""
    if dilation_value == 0:
        return mask_np
        
    # Convert to uint8 for OpenCV
    binary_mask = (mask_np * 255).astype(np.uint8)
    
    # Create kernel for dilation/erosion
    kernel_size = abs(dilation_value)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation or erosion
    if dilation_value > 0:
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    else:
        dilated = cv2.erode(binary_mask, kernel, iterations=1)
        
    # Convert back to float [0,1]
    return dilated.astype(float) / 255.0

def tensor_gaussian_blur_mask(mask_tensor, blur_size):
    """Apply gaussian blur to mask tensor using OpenCV."""
    # Extract mask and convert to numpy (handling batch if needed)
    if len(mask_tensor.shape) == 4:  # BHWC
        # Process each batch independently
        result = []
        for b in range(mask_tensor.shape[0]):
            img = mask_tensor[b, ..., 0].detach().cpu().numpy()
            # Apply blur (must be odd size)
            blur_amount = blur_size if blur_size % 2 == 1 else blur_size + 1
            blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
            result.append(torch.from_numpy(blurred).unsqueeze(-1))
        return torch.stack(result, dim=0)
    else:  # HWC
        img = mask_tensor[..., 0].detach().cpu().numpy()
        # Apply blur (must be odd size)
        blur_amount = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        return torch.from_numpy(blurred).unsqueeze(-1)

#------------------------------------------------------------------------------
# Section 4: Tensor Format Validation Functions
#------------------------------------------------------------------------------
def debug_tensor_format(tensor, name="Tensor"):
    """Print detailed debug information about tensor format."""
    print(f"\nDEBUG {name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Value range: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
    
    # Detect format based on shape
    shape_len = len(tensor.shape)
    if shape_len == 4:
        # Determine if BCHW or BHWC
        if tensor.shape[1] <= 4 and tensor.shape[2] >= 8 and tensor.shape[3] >= 8:
            print(f"Appears to be BCHW format: [batch={tensor.shape[0]}, channels={tensor.shape[1]}, height={tensor.shape[2]}, width={tensor.shape[3]}]")
        elif tensor.shape[3] <= 4 and tensor.shape[1] >= 8 and tensor.shape[2] >= 8:
            print(f"Appears to be BHWC format: [batch={tensor.shape[0]}, height={tensor.shape[1]}, width={tensor.shape[2]}, channels={tensor.shape[3]}]")
        else:
            print(f"Unknown 4D format: {tensor.shape}")
    elif shape_len == 3:
        # Determine if CHW or HWC
        if tensor.shape[0] <= 4 and tensor.shape[1] >= 8 and tensor.shape[2] >= 8:
            print(f"Appears to be CHW format: [channels={tensor.shape[0]}, height={tensor.shape[1]}, width={tensor.shape[2]}]")
        elif tensor.shape[2] <= 4 and tensor.shape[0] >= 8 and tensor.shape[1] >= 8:
            print(f"Appears to be HWC format: [height={tensor.shape[0]}, width={tensor.shape[1]}, channels={tensor.shape[2]}]")
        else:
            print(f"Unknown 3D format: {tensor.shape}")

def ensure_bchw_format(tensor, name="Unknown"):
    """
    Ensure tensor is in BCHW format for model operations.
    """
    print(f"Converting {name} to BCHW format:")
    print(f"  Original shape: {tensor.shape}")
    
    # Ensure tensor has 4 dimensions
    if len(tensor.shape) == 3:  # HWC
        # Check if this is CHW already (first dim is channels)
        if tensor.shape[0] <= 4 and tensor.shape[1] >= 8 and tensor.shape[2] >= 8:
            # Already in CHW format, just add batch dimension
            result = tensor.unsqueeze(0)
            print(f"  Added batch to CHW: {result.shape}")
        else:
            # HWC -> BCHW
            result = tensor.permute(2, 0, 1).unsqueeze(0)
            print(f"  Converted HWC to BCHW: {result.shape}")
    elif len(tensor.shape) == 4:
        # Check if already BCHW
        if tensor.shape[1] <= 4 and tensor.shape[2] >= 8 and tensor.shape[3] >= 8:
            # Already BCHW, do nothing
            result = tensor
            print(f"  Already in BCHW format: {result.shape}")
        else:
            # BHWC -> BCHW
            result = tensor.permute(0, 3, 1, 2)
            print(f"  Converted BHWC to BCHW: {result.shape}")
    else:
        raise ValueError(f"Unsupported tensor shape for {name}: {tensor.shape}")
    
    return result

def ensure_bhwc_format(tensor, name="Unknown"):
    """
    Ensure tensor is in BHWC format for ComfyUI operations.
    
    Args:
        tensor: Input tensor
        name: Name for debugging
        
    Returns:
        torch.Tensor: Tensor in BHWC format
    """
    # Debug original format
    print(f"Converting {name} to BHWC format:")
    print(f"  Original shape: {tensor.shape}")
    
    # Ensure tensor has 4 dimensions
    if len(tensor.shape) == 3:  # HWC or CHW
        # Determine if CHW
        if tensor.shape[0] <= 4 and tensor.shape[1] >= 8 and tensor.shape[2] >= 8:
            # CHW -> BHWC (add batch and permute)
            result = tensor.unsqueeze(0).permute(0, 2, 3, 1)
            print(f"  Converted CHW to BHWC: {result.shape}")
        else:
            # HWC -> BHWC (just add batch dimension)
            result = tensor.unsqueeze(0)
            print(f"  Added batch to HWC: {result.shape}")
    elif len(tensor.shape) == 4:  # BCHW or BHWC
        # Check if already BHWC
        if tensor.shape[3] <= 4 and tensor.shape[1] >= 8 and tensor.shape[2] >= 8:
            # Already BHWC
            result = tensor
            print(f"  Already in BHWC format: {result.shape}")
        else:
            # Assume BCHW and convert to BHWC
            result = tensor.permute(0, 2, 3, 1)
            print(f"  Converted BCHW to BHWC: {result.shape}")
    else:
        raise ValueError(f"Unsupported tensor shape for {name}: {tensor.shape}")
    
    # Verify result
    if not (result.shape[3] <= 4 and result.shape[1] >= 8 and result.shape[2] >= 8):
        print(f"  WARNING: Result may not be in correct BHWC format: {result.shape}")
    
    return result

#------------------------------------------------------------------------------
# Section 5: VAE and Sampler Functions
#------------------------------------------------------------------------------
def apply_vae_encode(img, vae, device=None):
    """
    Encode an image using VAE with robust error handling.
    
    Args:
        img: Image tensor in proper format for VAE
        vae: VAE model object
        device: Optional device
        
    Returns:
        Encoded latent tensor
    """
    print("==== VAE ENCODE FUNCTION STARTING ====")
    print(f"Input shape: {img.shape}")
    
    # Ensure input is in the correct format for the VAE (BCHW)
    if len(img.shape) == 4 and img.shape[1] == 3:
        # Already in BCHW format, no need for permutation
        bchw_tensor = img
        print(f"Image already in correct BCHW format: {bchw_tensor.shape}")
    else:
        # Try to convert to proper format if needed
        print("Input may not be in correct format, attempting to fix...")
        if len(img.shape) == 4 and img.shape[3] == 3:  # BHWC format
            bchw_tensor = img.permute(0, 3, 1, 2)
            print(f"Converted from BHWC to BCHW: {bchw_tensor.shape}")
        else:
            bchw_tensor = img
            print(f"Using original tensor shape: {bchw_tensor.shape}")
    
    # Get device from VAE or use provided device
    if device is None:
        if hasattr(vae, 'device'):
            device = vae.device
        else:
            try:
                device = next(vae.parameters()).device
            except:
                try:
                    # Different VAE structure
                    device = next(vae.first_stage_model.parameters()).device
                except:
                    # Fall back to tensor device
                    device = bchw_tensor.device
    
    print(f"VAE device: {device}")
    print(f"Final tensor shape for encoding: {bchw_tensor.shape}")
    
    # Move tensor to VAE's device
    bchw_tensor = bchw_tensor.to(device)
    
    try:
        # First approach: standard encode
        samples = vae.encode(bchw_tensor)
        
        # Handle if samples is a dictionary (some VAE implementations return dict)
        if isinstance(samples, dict) and "samples" in samples:
            print(f"VAE returned a dictionary, extracting 'samples' key")
            return samples["samples"]
        else:
            print(f"VAE encoding successful: {samples.shape}")
            return samples
            
    except Exception as e:
        print(f"VAE encoding error: {e}")
        traceback.print_exc()
        
        # Try alternative approach
        print(f"VAE encoding failed, trying alternative approach: {e}")
        
        try:
            # Unload models to ensure memory availability
            if hasattr(comfy, 'model_management'):
                comfy.model_management.unload_all_models()
                print("Models unloaded to free memory")
            
            # Direct encode
            latent = vae.encode(bchw_tensor)
            
            # Handle dictionary result
            if isinstance(latent, dict) and "samples" in latent:
                print(f"Direct VAE encoding returned a dictionary, extracting 'samples' key")
                latent = latent["samples"]
                
            print(f"Direct VAE encoding successful: {latent.shape}")
            return latent
            
        except Exception as e2:
            print(f"Direct encoding also failed: {e2}")
            traceback.print_exc()
            
            # Final attempt with comfy's direct approach
            try:
                from comfy import model_management
                
                # Try with comfy's specific VAE handling if available
                encoder_device = model_management.get_torch_device()
                bchw_tensor = bchw_tensor.to(encoder_device)
                
                print(f"Final attempt on device: {encoder_device}")
                vae_output = vae.encode(bchw_tensor)
                
                if isinstance(vae_output, dict) and "samples" in vae_output:
                    print(f"Final encoding returned dictionary")
                    return vae_output["samples"]
                
                print(f"Final VAE encoding successful: {vae_output.shape}")
                return vae_output
                
            except Exception as e3:
                print(f"All encoding methods failed: {e3}")
                traceback.print_exc()
                return None

def apply_ksampler(sampler_name, scheduler, latent, seed, steps, cfg, denoise, model, positive, negative, upscaler_strength=1.0):
    """
    Apply KSampler with proper error handling.
    
    Args:
        sampler_name: Name of the sampler to use
        scheduler: Scheduler type
        latent: Latent samples (dict or tensor)
        seed: Random seed
        steps: Number of sampling steps
        cfg: CFG scale value
        denoise: Denoising strength
        model: Diffusion model
        positive: Positive conditioning
        negative: Negative conditioning
        upscaler_strength: Optional strength for upscaler (fallback)
        
    Returns:
        Sampled latent
    """
    print(f"Applying KSampler with steps={steps}, cfg={cfg}, denoise={denoise}")
    
    # Handle if latent is a dictionary
    if isinstance(latent, dict) and "samples" in latent:
        print("Using provided latent dictionary")
        latent_image = latent
    elif torch.is_tensor(latent):
        print("Converting tensor to latent dictionary format")
        latent_image = {"samples": latent}
    else:
        print(f"Unexpected latent type: {type(latent)}")
        raise ValueError(f"Unsupported latent type: {type(latent)}")
    
    try:
        import comfy.sample
        # Try direct approach with comfy's samplers
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"),
            scheduler,
            steps,
            denoise
        )
        
        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)
        
        # Get sampler
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        samples = comfy.sample.sample(
            model.get_model_object("model"),
            noise,
            sigmas,
            sampler,
            positive, 
            negative,
            cfg=cfg,
            latent_image=latent_image["samples"] if denoise < 1.0 else None,
        )
        
        result = {"samples": samples}
        return result
        
    except Exception as e:
        print(f"Direct sampling error: {e}")
        traceback.print_exc()
        
        # Try with node-based approach
        try:
            from nodes import common_ksampler
            
            samples, _ = common_ksampler(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent_image,
                denoise=denoise if denoise < 1.0 else upscaler_strength,
            )
            
            return samples
            
        except Exception as e2:
            print(f"Node-based sampling error: {e2}")
            traceback.print_exc()
            
            # Return original latent as fallback
            return latent_image

#------------------------------------------------------------------------------
# Section 6: Model-based Processing
#------------------------------------------------------------------------------
def process_with_model(
    image: torch.Tensor,
    upscale_model: Any,
    target_size: int,
    scale_factor: float = 1.0,
    divisibility: int = 64,
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """Process image with model-based scaling to target size multiplied by scale factor"""
    try:
        # Ensure image is in BHWC format first
        image = ensure_bhwc_format(image, "process_with_model input")

        # Calculate current dimensions and target dimensions
        input_h, input_w = image.shape[1:3]
        long_edge = max(input_w, input_h)

        # Calculate basic scale to reach target bucket size
        base_scale = target_size / long_edge

        # Apply additional scale factor
        final_scale = base_scale * scale_factor

        print(f"Scaling from {input_w}x{input_h} (BHWC) to bucket size {target_size} with factor {scale_factor}")
        print(f"Final scale factor: {final_scale:.4f}")

        # Calculate output dimensions while maintaining aspect ratio
        if input_w >= input_h:
            output_w = int(input_w * final_scale)
            output_h = int(input_h * final_scale)
        else:
            output_h = int(input_h * final_scale)
            output_w = int(input_w * final_scale)

        # Ensure dimensions are divisible by divisibility factor
        output_w = get_nearest_divisible_size(output_w, divisibility)
        output_h = get_nearest_divisible_size(output_h, divisibility)

        print(f"Final dimensions: {output_w}x{output_h} (divisible by {divisibility})")

        # If downscaling significantly, use bicubic first then model upscale
        if final_scale < 0.5:
            # Intermediate downscale with bicubic
            intermediate_h = int(input_h * final_scale)
            intermediate_w = int(input_w * final_scale)

            # Ensure intermediate dimensions are divisible
            intermediate_h = get_nearest_divisible_size(intermediate_h, divisibility)
            intermediate_w = get_nearest_divisible_size(intermediate_w, divisibility)

            # Convert to BCHW for PyTorch operations
            image_bchw = ensure_bchw_format(image, "downscale_input")

            # Downscale
            result = F.interpolate(
                image_bchw, size=(intermediate_h, intermediate_w),
                mode='bicubic',
                antialias=True,
                align_corners=False
            )

            # Convert back to BHWC and return
            return ensure_bhwc_format(result, "downscale_output")

        # Process with model
        result = image
        
        # Check upscale model validity
        if upscale_model is None:
            logger.warning("No upscale model provided, using bicubic")
            image_bchw = ensure_bchw_format(image, "bicubic_input")
            result_bchw = F.interpolate(
                image_bchw, 
                size=(output_h, output_w),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
            return ensure_bhwc_format(result_bchw, "bicubic_output")
            
        try:
            # Get model scale factor first (for proper tile calculations)
            model_factor = 1.0
            # Test model scale factor with small tensor
            test_input = torch.zeros((1, 3, 32, 32), device=image.device)
            test_output = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, test_input)[0]
            model_factor = test_output.shape[2] / 32.0
            logger.info(f"Model upscale factor: {model_factor}x")
        except Exception as e:
            logger.error(f"Error detecting model scale factor: {e}")
            model_factor = 4.0  # Default assumption
            
        # Process in tiles if image is large
        if input_h > tile_size or input_w > tile_size:
            # Process in tiles
            logger.info(f"Processing large image ({input_w}x{input_h}) in tiles")

            # Create result tensor
            result = torch.zeros((image.shape[0], output_h, output_w, image.shape[3]),
                                device=image.device, dtype=image.dtype)
                                
            # Create weight accumulation tensor for blending
            weights = torch.zeros((image.shape[0], output_h, output_w, 1),
                                 device=image.device, dtype=image.dtype)

            # Calculate tiles with proper overlap
            tiles_x = math.ceil(input_w / (tile_size - overlap))
            tiles_y = math.ceil(input_h / (tile_size - overlap))
            logger.info(f"Processing {tiles_x}x{tiles_y} tiles with {overlap}px overlap")

            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Calculate tile coordinates
                    x1 = x * (tile_size - overlap)
                    y1 = y * (tile_size - overlap)
                    x2 = min(x1 + tile_size, input_w)
                    y2 = min(y1 + tile_size, input_h)

                    # Ensure we don't have tiny tiles at the edge
                    if x2 - x1 < tile_size // 4 or y2 - y1 < tile_size // 4:
                        continue

                    # Extract tile
                    tile = image[:, y1:y2, x1:x2, :]
                    logger.info(f"Processing tile at ({x1},{y1}) to ({x2},{y2}) - Shape: {tile.shape}")

                    # Process tile with model
                    start_time = time.time()
                    
                    try:
                        # Convert to BCHW for model upscaler
                        tile_bchw = ensure_bchw_format(tile, f"tile_{x}_{y}")
                        
                        # Fix: Ensure channels dimension is 3 for model
                        if tile_bchw.shape[1] != 3:
                            logger.warning(f"Fixing channel dimension: {tile_bchw.shape[1]} -> 3")
                            if tile_bchw.shape[1] == 1:
                                # Grayscale to RGB
                                tile_bchw = tile_bchw.repeat(1, 3, 1, 1)
                            elif tile_bchw.shape[1] == 4:
                                # RGBA to RGB
                                tile_bchw = tile_bchw[:, :3, :, :]
                            else:
                                # Unknown format, reshape
                                tile_bchw = tile_bchw[:, :3, :, :]
                                
                        logger.info(f"Upscaling tile with shape: {tile_bchw.shape}")

                        # Upscale with model
                        upscaled_tile = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, tile_bchw)[0]
                        logger.info(f"Upscaled tile shape: {upscaled_tile.shape}")
                        
                    except Exception as e:
                        logger.error(f"Error in model upscaling tile: {e}")
                        # Fall back to bicubic for this tile
                        logger.info("Falling back to bicubic upscaling for this tile")
                        upscaled_tile = F.interpolate(
                            tile_bchw,
                            scale_factor=model_factor,
                            mode='bicubic',
                            align_corners=False
                        )

                    # Calculate output dimensions for this tile
                    tile_out_h = int((y2-y1) * final_scale)
                    tile_out_w = int((x2-x1) * final_scale)
                    
                    # Ensure output dimensions are valid
                    if tile_out_h <= 0 or tile_out_w <= 0:
                        logger.warning(f"Invalid tile output dimensions: {tile_out_w}x{tile_out_h}")
                        continue

                    # Resize to target scale if needed
                    if abs(model_factor - final_scale) > 0.01:
                        logger.info(f"Resizing from {upscaled_tile.shape} to {tile_out_h}x{tile_out_w}")
                        upscaled_tile = F.interpolate(
                            upscaled_tile,
                            size=(tile_out_h, tile_out_w),
                            mode='bicubic',
                            align_corners=False
                        )

                    # Convert back to BHWC
                    upscaled_tile = ensure_bhwc_format(upscaled_tile, f"upscaled_tile_{x}_{y}")

                    # Calculate output coordinates
                    out_x1 = int(x1 * final_scale)
                    out_y1 = int(y1 * final_scale)
                    out_x2 = min(out_x1 + upscaled_tile.shape[2], output_w)
                    out_y2 = min(out_y1 + upscaled_tile.shape[1], output_h)

                    # Verify outputs are within bounds
                    if out_x1 >= output_w or out_y1 >= output_h or out_x2 <= 0 or out_y2 <= 0:
                        logger.warning(f"Tile outside output bounds: ({out_x1},{out_y1}) to ({out_x2},{out_y2})")
                        continue
                        
                    # Adjust if needed
                    out_x1 = max(0, out_x1)
                    out_y1 = max(0, out_y1)
                    
                    # Create weight mask for blending
                    mask_h = out_y2 - out_y1
                    mask_w = out_x2 - out_x1
                    mask = torch.ones((mask_h, mask_w, 1), device=upscaled_tile.device)

                    # Apply feathering to edges for smooth blending
                    if overlap > 0:
                        feather = int(overlap * final_scale)
                        
                        # Left edge
                        if x1 > 0 and feather > 0 and mask_w > 0:
                            for i in range(min(feather, mask_w)):
                                mask[:, i:i+1, :] *= (i / feather)
                                
                        # Right edge
                        if x2 < input_w and feather > 0 and mask_w > 0:
                            for i in range(min(feather, mask_w)):
                                idx = mask_w - i - 1
                                if idx >= 0:
                                    mask[:, idx:idx+1, :] *= (i / feather)
                                    
                        # Top edge
                        if y1 > 0 and feather > 0 and mask_h > 0:
                            for i in range(min(feather, mask_h)):
                                mask[i:i+1, :, :] *= (i / feather)
                                
                        # Bottom edge
                        if y2 < input_h and feather > 0 and mask_h > 0:
                            for i in range(min(feather, mask_h)):
                                idx = mask_h - i - 1
                                if idx >= 0:
                                    mask[idx:idx+1, :, :] *= (i / feather)
                    
                    # Extract appropriate portion of the upscaled tile
                    tile_h = min(upscaled_tile.shape[1], mask_h)
                    tile_w = min(upscaled_tile.shape[2], mask_w)
                    tile_to_paste = upscaled_tile[:, :tile_h, :tile_w, :]
                    
                    # Fix mask dimensions if needed
                    if mask.shape[0] != tile_h or mask.shape[1] != tile_w:
                        mask = mask[:tile_h, :tile_w, :]
                    
                    # Blend tile into result
                    result[:, out_y1:out_y1+tile_h, out_x1:out_x1+tile_w, :] += tile_to_paste * mask
                    weights[:, out_y1:out_y1+tile_h, out_x1:out_x1+tile_w, :] += mask

                    # Log timing
                    logger.info(f"Processed tile {x+1}x{y+1}/{tiles_x}x{tiles_y} in {time.time()-start_time:.2f}s")

            # Normalize by weights to avoid seams
            valid_weights = weights > 0
# Normalize by weights to avoid seams (continued)
            result = torch.where(valid_weights, result / (weights + 1e-8), result)
            
            return result
            
        else:
            # Small image, process whole
            start_time = time.time()
            
            try:
                # Convert to BCHW for model upscaler
                image_bchw = ensure_bchw_format(image, "small_image")
                
                # Fix: Ensure channels dimension is 3 for model
                if image_bchw.shape[1] != 3:
                    logger.warning(f"Fixing channel dimension: {image_bchw.shape[1]} -> 3")
                    if image_bchw.shape[1] == 1:
                        # Grayscale to RGB
                        image_bchw = image_bchw.repeat(1, 3, 1, 1)
                    elif image_bchw.shape[1] == 4:
                        # RGBA to RGB
                        image_bchw = image_bchw[:, :3, :, :]
                    else:
                        # Unknown format, reshape to use 3 channels
                        image_bchw = image_bchw[:, :3, :, :]

                # Model upscale first (preserves details better than bicubic)
                upscaled = model_upscale.ImageUpscaleWithModel().upscale(upscale_model, image_bchw)[0]
                logger.info(f"Model upscaled shape: {upscaled.shape}")

                # Resize to final dimensions using bicubic if needed
                if upscaled.shape[2] != output_h or upscaled.shape[3] != output_w:
                    upscaled = F.interpolate(
                        upscaled,
                        size=(output_h, output_w),
                        mode='bicubic',
                        align_corners=False
                    )

                # Convert back to BHWC
                result = ensure_bhwc_format(upscaled, "upscaled_whole")
                logger.info(f"Processed whole image in {time.time()-start_time:.2f}s")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in whole image upscaling: {e}")
                traceback.print_exc()
                
                # Fall back to bicubic
                logger.info("Falling back to bicubic upscaling")
                try:
                    upscaled = F.interpolate(
                        image_bchw,
                        size=(output_h, output_w),
                        mode='bicubic',
                        align_corners=False
                    )
                    return ensure_bhwc_format(upscaled, "fallback_bicubic")
                except Exception as e2:
                    logger.error(f"Bicubic fallback also failed: {e2}")
                    return image  # Return original as last resort

    except Exception as e:
        logger.error(f"Error in model upscaling: {e}")
        traceback.print_exc()
        
        # Fallback to bicubic with minimal error surface
        logger.info("Falling back to safe bicubic scaling")
        try:
            image_bchw = ensure_bchw_format(image, "fallback_input")
            
            # Ensure values are valid
            if torch.isnan(image_bchw).any() or torch.isinf(image_bchw).any():
                logger.warning("Fixing NaN/Inf values in tensor")
                image_bchw = torch.nan_to_num(image_bchw, nan=0.5, posinf=1.0, neginf=0.0)
                
            # Safe resize
            result = F.interpolate(
                image_bchw,
                size=(output_h, output_w),
                mode='bicubic',
                align_corners=False
            )
            return ensure_bhwc_format(result, "fallback_output")
        except:
            logger.error("All scaling methods failed, returning original")
            return image

def match_mask_to_proportions(mask: torch.Tensor, target_w: int, target_h: int, divisibility: int) -> torch.Tensor:
    """Dilate and adjust mask to match target proportions while ensuring divisibility"""
    # Ensure we have BHWC format
    mask = ensure_bhwc_format(mask, "mask_input")
    
    # Ensure dimensions are divisible
    target_w = get_nearest_divisible_size(target_w, divisibility)
    target_h = get_nearest_divisible_size(target_h, divisibility)
    
    # Resize mask to match target proportions
    mask_bchw = ensure_bchw_format(mask, "mask_for_resize")
    
    # Check for zero dimensions
    if target_h <= 0 or target_w <= 0:
        logger.warning(f"Invalid target dimensions for mask: {target_w}x{target_h}")
        # Return original mask
        return mask
        
    resized_mask = F.interpolate(
        mask_bchw,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to BHWC
    resized_mask = ensure_bhwc_format(resized_mask, "resized_mask")
    
    # Maintain mask integrity after resizing
    resized_mask = (resized_mask > 0.5).float()
    
    return resized_mask

#------------------------------------------------------------------------------
# Section 7: Main Node Implementation
#------------------------------------------------------------------------------
class HTDetectionBatchProcessor:
    """
    Node for detecting regions, processing with mask dilation, and scaling detected regions.
    Combines BBOX detection, masking, and model-based upscaling in one workflow.
    Uses standard bucket sizes (512, 768, 1024) with additional scale factor.
    """
    
    CATEGORY = "HommageTools/Processor"
    FUNCTION = "process"  # This must match the actual method name
    RETURN_TYPES = ("IMAGE", "MASK", "SEGS", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("processed_images", "processed_masks", "segs", "cropped_images", "bypass_image", "bbox_count")
    OUTPUT_IS_LIST = (True, True, False, True, False, False)

    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "scale_mode": (["Scale Closest", "Scale Up", "Scale Down", "Scale Max"], {"default": "Scale Closest"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                "divisibility": (["8", "64"], {"default": "64"}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                # KSampler parameters
                "model": ("MODEL",),
                "vae": ("VAE",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "upscale_model": ("UPSCALE_MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List types of segments to allow, comma-separated"})
            }
        }
    
    def process(self, image: torch.Tensor, detection_threshold: float, mask_dilation: int, crop_factor:float, drop_size: int, tile_size: int, scale_mode: str, scale_factor: float, divisibility: str, mask_blur: int, overlap: int, model, vae, steps: int, cfg: float, denoise: float, sampler_name: str, scheduler: str, seed: int, bbox_detector=None, upscale_model=None, positive=None, negative=None, labels="all") -> Tuple[List[torch.Tensor], List[torch.Tensor], Any, List[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Process the input image through detection, masking, and scaling pipeline.
        Uses standard bucket sizes with additional scale factor.
        """
        logger.info(f"\n===== HTDetectionBatchProcessor v{VERSION} - Starting =====")
        
        # Convert divisibility from string to int
        div_factor = int(divisibility)
        
        # Ensure BHWC format
        image = ensure_bhwc_format(image, "input_image")
            
        # Get dimensions
        batch, height, width, channels = image.shape
        logger.info(f"Input image: {width}x{height}x{channels} (BHWC format)")
        
        # Initialize return values to empty defaults
        processed_images = []
        processed_masks = []
        cropped_images = []
        bypass_image = image  # Initialize bypass to input image
        bbox_count = 0  
        
        # Step 1: Run detection
        if bbox_detector is not None:
            logger.info("Running BBOX detection...")
            # Debug detector type
            detector_type = type(bbox_detector).__name__
            logger.info(f"Detector type: {detector_type}")
            
            # Use first image if batch > 1
            detect_image = image[0:1] if batch > 1 else image
            
            # Debug image shape before detection
            logger.info(f"Detection image shape: {detect_image.shape}")
            
            try:
                # Run detection
                segs = bbox_detector.detect(detect_image, detection_threshold, mask_dilation, crop_factor, drop_size)
                
                # Validate detection result format
                if not isinstance(segs, tuple) or len(segs) < 2:
                    logger.warning(f"Unexpected detection result format: {type(segs)}")
                    return processed_images, processed_masks, ((height, width), []), cropped_images, bypass_image, bbox_count
                    
                logger.info(f"Detection found {len(segs[1])} segments")
            except Exception as e:
                logger.error(f"Detection error: {e}")
                traceback.print_exc()
                return processed_images, processed_masks, ((height, width), []), cropped_images, bypass_image, bbox_count
                
            # Filter by labels
            if labels != '' and labels != 'all':
                label_list = [label.strip() for label in labels.split(',')]
                logger.info(f"Filtering for labels: {label_list}")
                
                filtered_segs = []
                for seg in segs[1]:
                    if seg.label in label_list:
                        filtered_segs.append(seg)
                        
                logger.info(f"Filtered from {len(segs[1])} to {len(filtered_segs)} segments")
                segs = (segs[0], filtered_segs)
        else:
            logger.warning("No detector provided!")
            return processed_images, processed_masks, ((height, width), []), cropped_images, bypass_image, bbox_count
        
        # Step 2: Process each detection
        if not segs[1] or len(segs[1]) == 0:
            logger.info("No detections found")
            return processed_images, processed_masks, ((height, width), []), cropped_images, bypass_image, bbox_count
        
        logger.info(f"Processing {len(segs[1])} detected regions...")
        bbox_count = len(segs[1])  # Store the number of detected bboxes
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if we have conditioning
        use_ksampler = model is not None and positive is not None
        if use_ksampler:
            if negative is None:
                # Create empty negative prompt
                negative = model.get_empty_conditioning()
            logger.info(f"Using KSampler with {steps} steps, CFG {cfg}, denoise {denoise}")
        
        for i, seg in enumerate(segs[1]):
            logger.info(f"Region {i+1}/{len(segs[1])}: {seg.label}")
            
            # Extract crop region
            x1, y1, x2, y2 = seg.crop_region
            crop_width = x2 - x1
            crop_height = y2 - y1
            logger.info(f"Crop region: ({x1},{y1}) to ({x2},{y2}) → {crop_width}x{crop_height}")
            
            # Extract cropped image
            if hasattr(seg, 'cropped_image') and seg.cropped_image is not None:
                # Convert PIL to tensor if needed
                try:
                    import numpy as np
                    from PIL import Image
                    if isinstance(seg.cropped_image, Image.Image):
                        cropped_np = np.array(seg.cropped_image).astype(np.float32) / 255.0
                        cropped_image = torch.from_numpy(cropped_np)
                        if len(cropped_image.shape) == 3:  # HWC
                            cropped_image = cropped_image.unsqueeze(0)  # BHWC
                    else:
                        cropped_image = seg.cropped_image
                        if len(cropped_image.shape) == 3:
                            cropped_image = cropped_image.unsqueeze(0)
                except:
                    # Fall back to direct extraction
                    cropped_image = image[:, y1:y2, x1:x2, :]
            else:
                cropped_image = image[:, y1:y2, x1:x2, :]
            
            # Store cropped image in the list
            cropped_images.append(cropped_image)
            
            # Get mask
            if hasattr(seg, 'cropped_mask') and seg.cropped_mask is not None:
                try:
                    # Handle different mask formats
                    if isinstance(seg.cropped_mask, np.ndarray):
                        cropped_mask = torch.from_numpy(seg.cropped_mask).float()
                    else:
                        cropped_mask = seg.cropped_mask
                        
                    # Add channel dimension if needed
                    if len(cropped_mask.shape) == 2:
                        cropped_mask = cropped_mask.unsqueeze(-1)
                    # Handle multiple channels
                    if len(cropped_mask.shape) == 3 and cropped_mask.shape[-1] != 1:
                        # Use first channel or average
                        if cropped_mask.shape[-1] <= 4:
                            cropped_mask = cropped_mask[..., 0:1]
                        else:
                            cropped_mask = cropped_mask.mean(dim=-1, keepdim=True)
                except:
                    # Create default mask
                    cropped_mask = torch.ones((crop_height, crop_width, 1), dtype=torch.float32)
            else:
                cropped_mask = torch.ones((crop_height, crop_width, 1), dtype=torch.float32)
            
            # Process mask if needed
            if mask_dilation != 0:
                # Extract mask data for OpenCV
                if len(cropped_mask.shape) == 4:  # BHWC
                    mask_np = cropped_mask[0, ..., 0].cpu().numpy()
                else:  # HWC
                    mask_np = cropped_mask[..., 0].cpu().numpy()
                    
                # Apply dilation
                dilated_np = dilate_mask_opencv(mask_np, mask_dilation)
                
                # Convert back to tensor
                if len(cropped_mask.shape) == 4:  # BHWC
                    cropped_mask = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(-1)
                else:  # HWC
                    cropped_mask = torch.from_numpy(dilated_np).unsqueeze(-1)
            
            # Apply mask blur
            if mask_blur > 0:
                cropped_mask = tensor_gaussian_blur_mask(cropped_mask, mask_blur)
            
            # Move to device for processing
            cropped_image = cropped_image.to(device)
            
            # Calculate target size using standard buckets
            long_edge = max(crop_width, crop_height)
            target_size = calculate_target_size(long_edge, scale_mode)
            
            # Calculate final target dimensions with scale factor
            if crop_width >= crop_height:
                ratio = crop_height / crop_width
                target_w = int(target_size * scale_factor)
                target_h = int(target_w * ratio)
            else:
                ratio = crop_width / crop_height
                target_h = int(target_size * scale_factor)
                target_w = int(target_h * ratio)
            
            # Ensure dimensions are divisible by required factor
            target_w = get_nearest_divisible_size(target_w, div_factor)
            target_h = get_nearest_divisible_size(target_h, div_factor)
            
            logger.info(f"Target dimensions: {target_w}x{target_h} (divisible by {div_factor})")
            
            # Process image with KSampler if available
            if use_ksampler and denoise > 0:
                try:
                    # Prepare latent from cropped image
                    logger.info("Encoding image with VAE...")
                    
                    # Convert image to proper format for VAE (BCHW)
                    img_for_vae = ensure_bchw_format(cropped_image, "img_for_vae")
                    
                    # Ensure dimensions are divisible by 8 for VAE
                    b, c, h, w = img_for_vae.shape
                    vae_h = ((h + 7) // 8) * 8
                    vae_w = ((w + 7) // 8) * 8
                    
                    # Resize if needed
                    if vae_h != h or vae_w != w:
                        logger.info(f"Resizing from {h}x{w} to {vae_h}x{vae_w} for VAE")
                        img_for_vae = F.interpolate(
                            img_for_vae,
                            size=(vae_h, vae_w),
                            mode='bicubic',
                            align_corners=False
                        )
                    
                    # Encode with VAE
                    vae_samples = apply_vae_encode(img_for_vae, vae, device)
                    if vae_samples is None:
                        raise ValueError("VAE encoding failed")
                        
                    logger.info(f"VAE encoding successful: {vae_samples.shape}")
                    
                    # Prepare latent for sampling
                    latent = {"samples": vae_samples}
                    
                    # Apply KSampler
                    sampled_latent = apply_ksampler(
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        latent=latent,
                        seed=seed,
                        steps=steps,
                        cfg=cfg,
                        denoise=denoise,
                        model=model,
                        positive=positive,
                        negative=negative,
                        upscaler_strength=scale_factor
                    )
                    
                    # Decode with VAE
                    logger.info("Decoding with VAE...")
                    # Ensure we have samples in the expected format
                    samples_to_decode = sampled_latent["samples"] if isinstance(sampled_latent, dict) else sampled_latent
                    
                    # Get device for decoding
                    decode_device = samples_to_decode.device
                    
                    try:
                        decoded = vae.decode(samples_to_decode)
                        
                        # Handle if decoded is a dictionary
                        if isinstance(decoded, dict) and "samples" in decoded:
                            decoded = decoded["samples"]
                            
                        logger.info(f"Decoded shape: {decoded.shape}")
                        
                        # Convert to BHWC
                        refined_image = ensure_bhwc_format(decoded, "vae_decode_output")
                        
                    except Exception as e:
                        logger.error(f"VAE decoding failed: {e}")
                        traceback.print_exc()
                        
                        # Fall back to original
                        refined_image = cropped_image
                        
                    # Now scale with model if available or with bicubic
                    if upscale_model is not None:
                        logger.info("Scaling with upscale model...")
                        scaled_image = process_with_model(
                            refined_image, 
                            upscale_model, 
                            target_size,
                            scale_factor=1.0,  # Already applied in target calculation
                            divisibility=div_factor,
                            tile_size=tile_size,
                            overlap=overlap
                        )
                    else:
                        # Convert to BCHW
                        img_bchw = ensure_bchw_format(refined_image, "bicubic_scale_input")
                            
                        # Apply scaling
                        scaled_bchw = F.interpolate(
                            img_bchw, 
                            size=(target_h, target_w),
                            mode='bicubic',
                            align_corners=False,
                            antialias=True
                        )
                            
                        # Convert back to BHWC
                        scaled_image = ensure_bhwc_format(scaled_bchw, "bicubic_scale_output")
                            
                except Exception as e:
                    logger.error(f"Error in VAE processing: {e}")
                    traceback.print_exc()
                        
                    # Fall back to direct scaling
                    logger.info("Falling back to direct scaling due to VAE error")
                    if upscale_model is not None:
                        scaled_image = process_with_model(
                            cropped_image, 
                            upscale_model, 
                            target_size,
                            scale_factor=1.0,
                            divisibility=div_factor,
                            tile_size=tile_size,
                            overlap=overlap
                        )
                    else:
                        # Simple bicubic scaling
                        img_bchw = ensure_bchw_format(cropped_image, "fallback_bicubic_input")
                        scaled_bchw = F.interpolate(
                            img_bchw, 
                            size=(target_h, target_w),
                            mode='bicubic',
                            align_corners=False,
                            antialias=True
                        )
                        scaled_image = ensure_bhwc_format(scaled_bchw, "fallback_bicubic_output")
            else:
                # Apply scaling with model or bicubic
                if upscale_model is not None:
                    scaled_image = process_with_model(
                        cropped_image, 
                        upscale_model, 
                        target_size,
                        scale_factor=1.0,  # Already applied in target calculation
                        divisibility=div_factor,
                        tile_size=tile_size,
                        overlap=overlap
                    )
                else:
                    # Convert to BCHW
                    img_bchw = ensure_bchw_format(cropped_image, "direct_bicubic_input")
                    
                    # Apply scaling
                    scaled_bchw = F.interpolate(
                        img_bchw, 
                        size=(target_h, target_w),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True
                    )
                    
                    # Convert back to BHWC
                    scaled_image = ensure_bhwc_format(scaled_bchw, "direct_bicubic_output")
            
            # Scale and match mask to target dimensions
            scaled_mask = match_mask_to_proportions(
                cropped_mask,
                target_w,
                target_h,
                div_factor
            )
            
            # Verify final dimensions match for mask and image
            if scaled_mask.shape[1:3] != scaled_image.shape[1:3]:
                logger.warning(f"Dimensions mismatch. Image: {scaled_image.shape}, Mask: {scaled_mask.shape}")
                # Fix mask dimensions to match image
                mask_bchw = ensure_bchw_format(scaled_mask, "fix_mask_mismatch")
                fixed_mask = F.interpolate(
                    mask_bchw,
                    size=(scaled_image.shape[1], scaled_image.shape[2]),
                    mode='nearest'
                )
                scaled_mask = ensure_bhwc_format(fixed_mask, "fixed_mask")
            
            # Move back to CPU for output
            scaled_image = scaled_image.cpu()
            scaled_mask = scaled_mask.cpu()
            
            # Store results
            processed_images.append(scaled_image)
            processed_masks.append(scaled_mask)
            
            # Free GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        logger.info(f"Processing complete - returning {len(processed_images)} images")
        
        # Make sure we're not returning empty lists
        if not processed_images:
            # Return input image as fallback
            logger.warning("No processed images found, using input image as fallback")
            
        return processed_images, processed_masks, segs, cropped_images, bypass_image, bbox_count