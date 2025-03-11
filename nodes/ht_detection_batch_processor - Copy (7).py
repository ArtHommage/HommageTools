"""
File: homage_tools/nodes/ht_detection_batch_processor.py
Version: 1.4.1
Description: Node for detecting regions, applying mask dilation, and processing with model-based upscaling/downscaling
             Using ComfyUI's built-in VAE nodes for better tensor handling
             Optimized for memory efficiency with large images
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
print(torch.__version__)
import torch.nn.functional as F
import numpy as np
import math
import logging
import time
import traceback
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2

# ComfyUI imports
import comfy
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy_extras.nodes_upscale_model as model_upscale

# Import ComfyUI's VAE nodes
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.4.1"
MAX_RESOLUTION = 8192
# Standard bucket sizes for auto-scaling
STANDARD_BUCKETS = [512, 768, 1024]
# Memory optimization constants
MAX_REGION_PIXELS = 1024 * 1024  # Max pixels in a region before scaling
MAX_PROCESSING_DIM = 1024  # Maximum dimension for processing
MEMORY_MONITORING = True  # Enable memory usage monitoring

#------------------------------------------------------------------------------
# Section 3: Helper Functions
#------------------------------------------------------------------------------
def create_placeholder():
    """Create minimal placeholder tensors for when no detections are found"""
    placeholder_image = torch.zeros((1, 10, 10, 3), dtype=torch.float32)
    placeholder_mask = torch.ones((1, 10, 10, 1), dtype=torch.float32)
    return placeholder_image, placeholder_mask
def log_memory():
    """Log GPU memory usage if available and enabled"""
    if MEMORY_MONITORING and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(f"GPU Memory: Current={allocated:.1f}MB, Peak={max_allocated:.1f}MB")

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

def dynamic_steps_for_size(input_h, input_w, base_steps):
    """Dynamically adjust steps based on image size to save memory"""
    pixels = input_h * input_w
    
    if pixels > 2 * MAX_REGION_PIXELS:
        return min(base_steps, 5)  # Very large regions
    elif pixels > MAX_REGION_PIXELS:
        return min(base_steps, 10)  # Large regions
    else:
        return base_steps  # Normal size regions

#------------------------------------------------------------------------------
# Section 4: Tensor Format Functions
#------------------------------------------------------------------------------
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
    
    # Safeguard against empty tensors
    if any(d == 0 for d in tensor.shape):
        raise ValueError(f"Tensor {name} has a zero dimension: {tensor.shape}")
    
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
    
    # Ensure the tensor is contiguous in memory for better performance
    if not result.is_contiguous():
        result = result.contiguous()
        print(f"  Made tensor contiguous in memory")
    
    return result

def process_with_model(
    image: torch.Tensor,
    upscale_model: Any,
    target_width: int,
    target_height: int,
    divisibility: int = 64,
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """Process image with model-based scaling to exact target dimensions"""
    import torchvision.transforms.functional as TF
    from PIL import Image
    # Ensure comfy_extras is available in this scope
    import comfy_extras.nodes_upscale_model
    
    # Ensure image is in BHWC format
    image = ensure_bhwc_format(image, "process_with_model input")

    # Get current dimensions
    input_h, input_w = image.shape[1:3]

    # Check if image is too large to process directly
    if input_h > MAX_PROCESSING_DIM or input_w > MAX_PROCESSING_DIM:
        print(f"Image too large ({input_w}x{input_h}), downscaling for processing")
        scale = MAX_PROCESSING_DIM / max(input_h, input_w)
        temp_h = int(input_h * scale)
        temp_w = int(input_w * scale)
        
        # Downsample using PIL
        batch_size = image.shape[0]
        temp_images = []
        
        for b in range(batch_size):
            # Extract single image and convert to numpy
            img_np = image[b].detach().cpu().numpy()
            # Ensure values are in 0-1 range
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype('uint8')
            else:
                img_np = img_np.astype('uint8')
            # Convert to PIL and resize
            pil_img = Image.fromarray(img_np)
            resized = pil_img.resize((temp_w, temp_h), Image.Resampling.LANCZOS)
            # Convert back to tensor
            tensor = TF.to_tensor(resized).unsqueeze(0)  # Add batch
            tensor = tensor.permute(0, 2, 3, 1)  # BCHW to BHWC
            temp_images.append(tensor)
        
        # Combine tensors
        if len(temp_images) > 1:
            image = torch.cat(temp_images, dim=0)
        else:
            image = temp_images[0]
    
    # Ensure dimensions are divisible by divisibility factor
    output_w = get_nearest_divisible_size(target_width, divisibility)
    output_h = get_nearest_divisible_size(target_height, divisibility)

    print(f"Scaling from {input_w}x{input_h} (BHWC) to {output_w}x{output_h} (divisible by {divisibility})")
    
    # Try to use model-based upscaling if available
    if upscale_model is not None:
        try:
            # CRITICAL FIX: Bypass model upscaling for ATD models which appear to have dimension issues
            model_type = getattr(upscale_model, 'model_type', None)
            if model_type and "ATD" in str(model_type):
                print(f"Detected ATD model type: {model_type}. Using PIL-based scaling for compatibility.")
                raise ValueError("ATD model detected - using fallback method")
                
            # Ensure we have 3 channels for RGB
            if image.shape[3] != 3:
                print(f"Adjusting channel count from {image.shape[3]} to 3")
                if image.shape[3] > 3:
                    image = image[:, :, :, :3]  # Take first three channels
                else:
                    # Duplicate channels to get 3
                    image = torch.cat([image] * 3, dim=3)[:, :, :, :3]
            
            # Convert BHWC to BCHW for model upscaling (standard PyTorch format)
            image_bchw = image.permute(0, 3, 1, 2).contiguous()
            print(f"Converted to BCHW: {image_bchw.shape}")
            
            # Get scale factor from model
            model_scale = getattr(upscale_model, 'scale', 4)
            print(f"Model scale factor: {model_scale}x")
            
            # Use smaller tile size for safety
            safe_tile_size = min(tile_size, 256)
            print(f"Using safe tile size: {safe_tile_size}x{safe_tile_size}")
            
            # Use Lanczos to get to intermediate size first if needed
            if abs(output_w/input_w - model_scale) > 0.5 or abs(output_h/input_h - model_scale) > 0.5:
                # We need to resize to an intermediate size first
                intermediate_w = int(output_w / model_scale)
                intermediate_h = int(output_h / model_scale)
                
                print(f"Using intermediate resize to {intermediate_w}x{intermediate_h} before model upscaling")
                
                # Resize to intermediate size using F.interpolate
                image_bchw = F.interpolate(
                    image_bchw, 
                    size=(intermediate_h, intermediate_w), 
                    mode='bicubic', 
                    align_corners=False
                ).contiguous()
                
                print(f"Intermediate resize complete: {image_bchw.shape}")
            
            # Ensure dimensions are divisible by 8 for most models
            c, h, w = image_bchw.shape[1], image_bchw.shape[2], image_bchw.shape[3]
            if h % 8 != 0 or w % 8 != 0:
                h_new = (h // 8) * 8
                w_new = (w // 8) * 8
                print(f"Adjusting dimensions to be divisible by 8: {w}x{h} -> {w_new}x{h_new}")
                image_bchw = F.interpolate(
                    image_bchw, 
                    size=(h_new, w_new), 
                    mode='bicubic', 
                    align_corners=False
                ).contiguous()
            
            # Get device safely without assuming parameters() method exists
            if hasattr(upscale_model, 'device'):
                device = upscale_model.device
            elif hasattr(upscale_model, 'model') and hasattr(upscale_model.model, 'parameters'):
                device = next(upscale_model.model.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Could not determine model device, using {device}")
            
            image_bchw = image_bchw.to(device)
            
            # For memory efficiency, process in tiles
            with torch.no_grad():
                # Use direct model upscaling instead of tiled to avoid dimension issues
                print(f"Input tensor shape before upscaling: {image_bchw.shape}")
                
                # Verify tensor is properly formed
                if image_bchw.shape[1] != 3:
                    raise ValueError(f"Expected 3 channels, got {image_bchw.shape[1]}")
                if not image_bchw.is_contiguous():
                    image_bchw = image_bchw.contiguous()
                    
                # Try direct upscaling first for more reliable results
                try:
                    # Process directly if image is small enough
                    if image_bchw.shape[2] * image_bchw.shape[3] <= 512 * 512:
                        print("Using direct model inference (small image)")
                        result_bchw = upscale_model(image_bchw)
                    else:
                        # Create upscaler instance
                        upscaler = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
                        
                        # Custom upscale without ComfyUI's tiled_scale which may have dimension issues
                        print("Using custom tiled upscaling")
                        
                        # Implement manual tiling
                        bs, c, h, w = image_bchw.shape
                        result_h = h * model_scale
                        result_w = w * model_scale
                        result_bchw = torch.zeros((bs, c, result_h, result_w), device=device)
                        
                        # Use smaller tiles
                        actual_tile = min(safe_tile_size, h, w)
                        actual_overlap = min(overlap, actual_tile // 4)
                        
                        # Manual tiling implementation
                        for y in range(0, h, actual_tile - actual_overlap):
                            for x in range(0, w, actual_tile - actual_overlap):
                                # Calculate tile boundaries
                                y_end = min(y + actual_tile, h)
                                x_end = min(x + actual_tile, w)
                                y_start = y
                                x_start = x
                                
                                # Extract tile
                                tile = image_bchw[:, :, y_start:y_end, x_start:x_end]
                                
                                # Upscale tile directly
                                upscaled_tile = upscale_model(tile)
                                
                                # Calculate output positions
                                out_y_start = y_start * model_scale
                                out_x_start = x_start * model_scale
                                out_y_end = y_end * model_scale
                                out_x_end = x_end * model_scale
                                
                                # Place upscaled tile
                                result_bchw[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = upscaled_tile
                                
                except Exception as e:
                    print(f"Direct model inference failed: {e}")
                    print("Falling back to ComfyUI upscaler...")
                    
                    # Fall back to ComfyUI upscaler
                    upscaler = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel()
                    result_bchw = upscaler.upscale(upscale_model, image_bchw)
                    
                print(f"Final BCHW shape after upscaling: {result_bchw.shape}")
                
                # Convert back to BHWC
                result = result_bchw.permute(0, 2, 3, 1)
                
                # Free memory
                torch.cuda.empty_cache()
                
                # Final resize to exact target size if needed
                if result.shape[1] != output_h or result.shape[2] != output_w:
                    print(f"Final resize to {output_w}x{output_h}")
                    final_bchw = result.permute(0, 3, 1, 2)
                    final_bchw = F.interpolate(
                        final_bchw, 
                        size=(output_h, output_w), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    result = final_bchw.permute(0, 2, 3, 1)
                
                return result
                
        except Exception as e:
            print(f"Error in model-based upscaling: {e}")
            traceback.print_exc()
            print("Falling back to PIL-based Lanczos scaling")
    
    # Fallback to PIL-based scaling which is reliable
    print("Using PIL-based Lanczos scaling")
    
    # Convert tensor to PIL images for resizing
    batch_size = image.shape[0]
    pil_images = []
    
    for b in range(batch_size):
        # Extract single image and convert to numpy
        img_np = image[b].detach().cpu().numpy()
        # Ensure the values are in 0-1 range
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype('uint8')
        else:
            img_np = img_np.astype('uint8')
        # Convert to PIL
        pil_img = Image.fromarray(img_np)
        pil_images.append(pil_img)
    
    # Resize images using Lanczos resampling
    resized_pil = []
    for pil_img in pil_images:
        resized = pil_img.resize((output_w, output_h), Image.Resampling.LANCZOS)
        resized_pil.append(resized)
    
    # Convert back to tensors
    result_tensors = []
    for pil_img in resized_pil:
        # Convert PIL to Tensor
        tensor = TF.to_tensor(pil_img).unsqueeze(0)  # Add batch dimension
        # Convert to BHWC
        tensor = tensor.permute(0, 2, 3, 1)
        result_tensors.append(tensor)
    
    # Combine tensors into a batch
    if len(result_tensors) > 1:
        return torch.cat(result_tensors, dim=0)
    else:
        return result_tensors[0]

def match_mask_to_proportions(mask: torch.Tensor, target_w: int, target_h: int, divisibility: int) -> torch.Tensor:
    """Dilate and adjust mask to match target proportions while ensuring divisibility"""
    from PIL import Image
    import numpy as np
    
    # Ensure we have BHWC format
    mask = ensure_bhwc_format(mask, "mask_input")
    
    # Ensure dimensions are divisible
    target_w = get_nearest_divisible_size(target_w, divisibility)
    target_h = get_nearest_divisible_size(target_h, divisibility)
    
    # Check for zero dimensions
    if target_h <= 0 or target_w <= 0:
        logger.warning(f"Invalid target dimensions for mask: {target_w}x{target_h}")
        # Return original mask
        return mask
    
    # Convert tensor mask to PIL
    batch_size = mask.shape[0]
    pil_masks = []
    
    for b in range(batch_size):
        # Extract single mask and convert to numpy
        mask_np = mask[b, ..., 0].detach().cpu().numpy()
        # Scale to 0-255 for PIL
        mask_np = (mask_np * 255).astype('uint8')
        # Convert to PIL
        pil_mask = Image.fromarray(mask_np)
        pil_masks.append(pil_mask)
    
    # Resize masks
    resized_pil = []
    for pil_mask in pil_masks:
        # Use nearest neighbor for mask resizing to preserve binary values
        resized = pil_mask.resize((target_w, target_h), Image.Resampling.NEAREST)
        resized_pil.append(resized)
    
    # Convert back to tensors
    result_tensors = []
    for pil_mask in resized_pil:
        # Convert PIL to numpy
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        # Convert numpy to tensor and add channel dimension
        tensor = torch.from_numpy(mask_np).unsqueeze(-1)  # Add channel dimension
        result_tensors.append(tensor)
    
    # Combine tensors into a batch
    if len(result_tensors) > 1:
        return torch.stack(result_tensors, dim=0)
    else:
        return result_tensors[0].unsqueeze(0)  # Add batch dimension

#------------------------------------------------------------------------------
# Section 5: Main Node Implementation
#------------------------------------------------------------------------------
class HTDetectionBatchProcessor:
    """
    Node for detecting regions, processing with mask dilation, and scaling detected regions.
    Combines BBOX detection, masking, and model-based upscaling in one workflow.
    Uses standard bucket sizes (512, 768, 1024) with additional scale factor.
    Memory-optimized for large images and regions.
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
    
    def apply_ksampler(self, sampler_name, scheduler, latent, seed, steps, cfg, denoise, model=None, positive=None, negative=None):
        """Apply KSampler to latent representation with memory optimization"""
        print(f"Applying KSampler with steps={steps}, cfg={cfg}, denoise={denoise}")
        
        try:
            from nodes import common_ksampler
            
            # Log initial memory usage
            if MEMORY_MONITORING:
                log_memory()
            
            # Use no_grad context to save memory
            with torch.no_grad():
                # Handle different return value formats from common_ksampler
                results = common_ksampler(
                    model=model,
                    seed=seed,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent=latent,
                    denoise=denoise
                )
                
                # Check if result is a tuple or a single value
                if isinstance(results, tuple) and len(results) > 0:
                    # Extract the first item if it's a tuple
                    result = results[0]
                else:
                    # Use result directly if it's not a tuple
                    result = results
            
            # Clear cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Log memory after processing
            if MEMORY_MONITORING:
                log_memory()
                
            return result
                
        except Exception as e:
            print(f"Error in KSampler: {e}")
            traceback.print_exc()
            
            # Try to clear memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return latent
    
    def process(self, image: torch.Tensor, detection_threshold: float, mask_dilation: int, crop_factor:float, drop_size: int, tile_size: int, scale_mode: str, scale_factor: float, divisibility: str, mask_blur: int, overlap: int, model, vae, steps: int, cfg: float, denoise: float, sampler_name: str, scheduler: str, seed: int, bbox_detector=None, upscale_model=None, positive=None, negative=None, labels="all") -> Tuple[List[torch.Tensor], List[torch.Tensor], Any, List[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Process the input image through detection, masking, and scaling pipeline.
        Uses standard bucket sizes with additional scale factor.
        Memory-optimized for large images and regions.
        """
        logger.info(f"\n===== HTDetectionBatchProcessor v{VERSION} - Starting =====")
        
        # Initialize ComfyUI's VAE nodes
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()
        
        # Convert divisibility from string to int
        div_factor = int(divisibility)
        
        # Initial memory usage
        if MEMORY_MONITORING:
            log_memory()
        
        # Ensure BHWC format
        try:
            image = ensure_bhwc_format(image, "input_image")
        except Exception as e:
            logger.error(f"Error converting input image to BHWC: {e}")
            return [], [], ((), []), [], image, 0
            
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
                
                # Clear cache after detection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Validate detection result format
                if not isinstance(segs, tuple) or len(segs) < 2:
                    logger.warning(f"Unexpected detection result format: {type(segs)}")
                        # Return tiny placeholder image for invalid detection format
                    placeholder_image, placeholder_mask = create_placeholder()
                    return [placeholder_image], [placeholder_mask], ((10, 10), []), [placeholder_image], image, 0
                    
                logger.info(f"Detection found {len(segs[1])} segments")
            except Exception as e:
                logger.error(f"Detection error: {e}")
                traceback.print_exc()
                # Return tiny placeholder image when detector errors occur
                placeholder_image, placeholder_mask = create_placeholder()
                return [placeholder_image], [placeholder_mask], ((10, 10), []), [placeholder_image], image, 0
                
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
            # Return tiny placeholder image when no detector is provided
            placeholder_image, placeholder_mask = create_placeholder()
            return [placeholder_image], [placeholder_mask], ((10, 10), []), [placeholder_image], image, 0
        
        # Step 2: Process each detection
        if not segs[1] or len(segs[1]) == 0:
            logger.info("No detections found")
            # Return tiny placeholder image when no detections found
            placeholder_image, placeholder_mask = create_placeholder()
            return [placeholder_image], [placeholder_mask], segs, [placeholder_image], image, 0
        
        logger.info(f"Processing {len(segs[1])} detected regions...")
        bbox_count = len(segs[1])  # Store the number of detected bboxes
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if we have conditioning
        use_ksampler = model is not None and positive is not None and denoise > 0
        if use_ksampler:
            if negative is None:
                # Create empty negative prompt
                negative = model.get_empty_conditioning()
            logger.info(f"Using KSampler with {steps} steps, CFG {cfg}, denoise {denoise}")
        else:
            logger.info("Skipping KSampler (either missing conditioning or denoise=0)")
        
        # Try to enable CPU offloading for VAE if possible 
        try:
            vae.cpu_offload = True
            logger.info("Enabled VAE CPU offloading")
        except:
            logger.info("VAE CPU offloading not available")
        
        for i, seg in enumerate(segs[1]):
            try:
                logger.info(f"Region {i+1}/{len(segs[1])}: {seg.label}")
                
                # Extract crop region
                x1, y1, x2, y2 = seg.crop_region
                crop_width = x2 - x1
                crop_height = y2 - y1
                logger.info(f"Crop region: ({x1},{y1}) to ({x2},{y2}) → {crop_width}x{crop_height}")
                
                # Check if region is too large and needs scaling
                region_pixels = crop_width * crop_height
                region_scale_factor = 1.0
                if region_pixels > MAX_REGION_PIXELS:
                    region_scale_factor = math.sqrt(MAX_REGION_PIXELS / region_pixels)
                    logger.info(f"Large region detected ({region_pixels} pixels), scaling by factor {region_scale_factor:.3f} for processing")
                
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
                    except Exception as e:
                        logger.error(f"Error processing cropped image from segment: {e}")
                        # Fall back to direct extraction
                        cropped_image = image[:, y1:y2, x1:x2, :]
                else:
                    cropped_image = image[:, y1:y2, x1:x2, :]
                
                # Double-check cropped image dimensions
                logger.info(f"Cropped image shape: {cropped_image.shape}")
                if any(d == 0 for d in cropped_image.shape):
                    logger.error(f"Cropped image has a zero dimension: {cropped_image.shape}")
                    continue  # Skip this region
                
                # Store cropped image in the list
                cropped_images.append(cropped_image)
                
                # Apply scaling if region is too large
                if region_scale_factor < 1.0:
                    import torchvision.transforms.functional as TF
                    temp_width = int(crop_width * region_scale_factor)
                    temp_height = int(crop_height * region_scale_factor)
                    
                    # Resize using PIL for memory efficiency
                    batch_size = cropped_image.shape[0]
                    temp_images = []
                    
                    for b in range(batch_size):
                        # Extract single image and convert to numpy
                        img_np = cropped_image[b].detach().cpu().numpy()
                        # Ensure values are in 0-1 range
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype('uint8')
                        else:
                            img_np = img_np.astype('uint8')
                        # Convert to PIL and resize
                        pil_img = Image.fromarray(img_np)
                        resized = pil_img.resize((temp_width, temp_height), Image.Resampling.LANCZOS)
                        # Convert back to tensor
                        tensor = TF.to_tensor(resized).unsqueeze(0)  # Add batch
                        tensor = tensor.permute(0, 2, 3, 1)  # BCHW to BHWC
                        temp_images.append(tensor)
                    
                    # Combine tensors
                    if len(temp_images) > 1:
                        cropped_image = torch.cat(temp_images, dim=0)
                    else:
                        cropped_image = temp_images[0]
                    
                    logger.info(f"Scaled to {temp_width}x{temp_height} for processing")
                
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
                    except Exception as e:
                        logger.error(f"Error processing mask from segment: {e}")
                        # Create default mask
                        cropped_mask = torch.ones((crop_height, crop_width, 1), dtype=torch.float32)
                else:
                    cropped_mask = torch.ones((crop_height, crop_width, 1), dtype=torch.float32)
                
                # Apply region scaling to mask if needed
                if region_scale_factor < 1.0:
                    # Scale mask using same method as above
                    temp_width = int(crop_width * region_scale_factor)
                    temp_height = int(crop_height * region_scale_factor)
                    
                    # Convert to PIL and resize
                    mask_np = cropped_mask[..., 0].cpu().numpy()
                    mask_np = (mask_np * 255).astype('uint8')
                    mask_pil = Image.fromarray(mask_np)
                    mask_resized = mask_pil.resize((temp_width, temp_height), Image.Resampling.NEAREST)
                    
                    # Convert back to tensor
                    mask_np_resized = np.array(mask_resized).astype(np.float32) / 255.0
                    cropped_mask = torch.from_numpy(mask_np_resized).unsqueeze(-1)
                
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
                if use_ksampler:
                    try:
                        # Dynamically adjust steps based on region size
                        effective_steps = dynamic_steps_for_size(
                            cropped_image.shape[1], 
                            cropped_image.shape[2], 
                            steps
                        )
                        
                        if effective_steps < steps:
                            logger.info(f"Reduced steps from {steps} to {effective_steps} for memory efficiency")
                        
                        # ===== VAE ENCODING using ComfyUI's VAEEncode node =====
                        logger.info("Using ComfyUI's VAEEncode node")
                        # Note: VAEEncode expects BCHW, but should handle conversion internally
                        with torch.amp.autocast('cuda'):  # Use mixed precision to save memory
                            latent = vae_encoder.encode(vae, cropped_image)[0]
                            logger.info(f"VAE encoding successful: {latent['samples'].shape}")
                        
                            # Apply KSampler
                            sampled_latent = self.apply_ksampler(
                                sampler_name=sampler_name,
                                scheduler=scheduler,
                                latent=latent,
                                seed=seed,
                                steps=effective_steps,
                                cfg=cfg,
                                denoise=denoise,
                                model=model,
                                positive=positive,
                                negative=negative
                            )
                        
                            # Decode using ComfyUI's VAEDecode node
                            logger.info("Decoding with VAEDecode")
                            if tile_size >= 512:  # Use tiled decode for large images
                                logger.info(f"Using tiled decode with tile size {tile_size}")
                                decoded = vae_decoder_tiled.decode(vae, sampled_latent, tile_size)[0]
                            else:
                                decoded = vae_decoder.decode(vae, sampled_latent)[0]
                        
                        logger.info(f"Decoded shape: {decoded.shape}")
                        
                        # VAEDecode should return BHWC format
                        refined_image = decoded
                        use_direct_scaling = False
                        
                    except Exception as e:
                        logger.error(f"Error in VAE processing: {e}")
                        traceback.print_exc()
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Fall back to direct scaling
                        use_direct_scaling = True
                else:
                    # No KSampler, just use direct scaling
                    use_direct_scaling = True
                
                # Apply scaling (either with refined image from VAE or original cropped image)
                if use_direct_scaling:
                    logger.info("Using direct scaling of original image")
                    source_image = cropped_image
                else:
                    logger.info("Scaling refined image from VAE")
                    source_image = refined_image
                
                # Scale to target size
                logger.info(f"Scaling to {target_w}x{target_h}")
                
                # Apply scaling with memory efficiency
                try:
                    scaled_image = process_with_model(
                        source_image, 
                        upscale_model, 
                        target_w,
                        target_h,
                        divisibility=div_factor,
                        tile_size=min(tile_size, 256),  # Limit tile size for memory efficiency
                        overlap=overlap
                    )
                except Exception as e:
                    logger.error(f"Error in image scaling: {e}")
                    # Fall back to PIL-based scaling in case of error
                    import torchvision.transforms.functional as TF
                    from PIL import Image
                    
                    # Convert to PIL and scale
                    batch_size = source_image.shape[0]
                    scaled_batch = []
                    
                    for b in range(batch_size):
                        img_np = source_image[b].detach().cpu().numpy()
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype('uint8')
                        else:
                            img_np = img_np.astype('uint8')
                        pil_img = Image.fromarray(img_np)
                        scaled_pil = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        tensor = TF.to_tensor(scaled_pil).unsqueeze(0)
                        tensor = tensor.permute(0, 2, 3, 1)
                        scaled_batch.append(tensor)
                    
                    if len(scaled_batch) > 1:
                        scaled_image = torch.cat(scaled_batch, dim=0)
                    else:
                        scaled_image = scaled_batch[0]
                
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
                    # Try to fix mask dimensions to match image
                    try:
                        mask_bchw = scaled_mask.permute(0, 3, 1, 2)
                        fixed_mask = F.interpolate(
                            mask_bchw,
                            size=(scaled_image.shape[1], scaled_image.shape[2]),
                            mode='nearest'
                        )
                        scaled_mask = fixed_mask.permute(0, 2, 3, 1)
                    except Exception as e:
                        logger.error(f"Error fixing mask dimensions: {e}")
                        # Create new mask with correct dimensions
                        scaled_mask = torch.ones((scaled_image.shape[0], scaled_image.shape[1], scaled_image.shape[2], 1), 
                                               dtype=torch.float32, device=scaled_image.device)
                
                # Move back to CPU for output
                scaled_image = scaled_image.cpu()
                scaled_mask = scaled_mask.cpu()
                
                # Store results
                processed_images.append(scaled_image)
                processed_masks.append(scaled_mask)
                
                # Free GPU memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing region {i+1}: {e}")
                traceback.print_exc()
                # Free memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Continue with the next region
                continue
        
        logger.info(f"Processing complete - returning {len(processed_images)} images")
        
        # Make sure we're not returning empty lists
        if not processed_images:
            # Return tiny placeholder image when processing failed for all regions
            logger.warning("No processed images found, using placeholder image")
            placeholder_image, placeholder_mask = create_placeholder()
            processed_images.append(placeholder_image)
            processed_masks.append(placeholder_mask)
            
        return processed_images, processed_masks, segs, cropped_images, bypass_image, bbox_count