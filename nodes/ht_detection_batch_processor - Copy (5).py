"""
File: homage_tools/nodes/ht_detection_batch_processor.py
Version: 1.4.0
Description: Node for detecting regions, applying mask dilation, and processing with model-based upscaling/downscaling
             Using ComfyUI's built-in VAE nodes for better tensor handling
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
VERSION = "1.4.0"
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
    target_size: int,
    scale_factor: float = 1.0,
    divisibility: int = 64,
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """Process image with model-based scaling to target size multiplied by scale factor"""
    # NOTE: This function currently uses bicubic scaling instead of model-based upscaling
    # We'll need to restore model-based upscaling once the core functionality is working
    
    # Ensure image is in BHWC format
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

    # TEMPORARY SOLUTION: Just use bicubic scaling
    # This should be replaced with model-based upscaling once the core functionality works
    logger.info("Using bicubic scaling (model-based upscaling will be re-enabled in future versions)")
    
    # Convert to BCHW for F.interpolate
    if len(image.shape) == 4 and image.shape[3] <= 4:  # BHWC
        image_bchw = image.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected tensor shape for upscaling: {image.shape}")
    
    # Apply scaling
    scaled_bchw = F.interpolate(
        image_bchw, 
        size=(output_h, output_w),
        mode='bicubic',
        align_corners=False,
        antialias=True
    )
    
    # Convert back to BHWC
    return scaled_bchw.permute(0, 2, 3, 1)

def match_mask_to_proportions(mask: torch.Tensor, target_w: int, target_h: int, divisibility: int) -> torch.Tensor:
    """Dilate and adjust mask to match target proportions while ensuring divisibility"""
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
    
    # Convert to BCHW for F.interpolate
    mask_bchw = mask.permute(0, 3, 1, 2)
    
    # Resize mask
    resized_mask = F.interpolate(
        mask_bchw,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to BHWC
    resized_mask = resized_mask.permute(0, 2, 3, 1)
    
    # Maintain mask integrity after resizing
    resized_mask = (resized_mask > 0.5).float()
    
    return resized_mask

#------------------------------------------------------------------------------
# Section 5: Main Node Implementation
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
    
    def apply_ksampler(self, sampler_name, scheduler, latent, seed, steps, cfg, denoise, model=None, positive=None, negative=None):
        """Apply KSampler to latent representation"""
        print(f"Applying KSampler with steps={steps}, cfg={cfg}, denoise={denoise}")
        
        try:
            from nodes import common_ksampler
            
            # Handle different return value formats from common_ksampler
            result = common_ksampler(
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
            if isinstance(result, tuple):
                # Extract the first item if it's a tuple
                return result[0]
            else:
                # Use result directly if it's not a tuple
                return result
                
        except Exception as e:
            print(f"Error in KSampler: {e}")
            traceback.print_exc()
            return latent
    
    def process(self, image: torch.Tensor, detection_threshold: float, mask_dilation: int, crop_factor:float, drop_size: int, tile_size: int, scale_mode: str, scale_factor: float, divisibility: str, mask_blur: int, overlap: int, model, vae, steps: int, cfg: float, denoise: float, sampler_name: str, scheduler: str, seed: int, bbox_detector=None, upscale_model=None, positive=None, negative=None, labels="all") -> Tuple[List[torch.Tensor], List[torch.Tensor], Any, List[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Process the input image through detection, masking, and scaling pipeline.
        Uses standard bucket sizes with additional scale factor.
        """
        logger.info(f"\n===== HTDetectionBatchProcessor v{VERSION} - Starting =====")
        
        # Initialize ComfyUI's VAE nodes
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()
        
        # Convert divisibility from string to int
        div_factor = int(divisibility)
        
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
        use_ksampler = model is not None and positive is not None and denoise > 0
        if use_ksampler:
            if negative is None:
                # Create empty negative prompt
                negative = model.get_empty_conditioning()
            logger.info(f"Using KSampler with {steps} steps, CFG {cfg}, denoise {denoise}")
        else:
            logger.info("Skipping KSampler (either missing conditioning or denoise=0)")
        
        for i, seg in enumerate(segs[1]):
            try:
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
                        # ===== VAE ENCODING using ComfyUI's VAEEncode node =====
                        logger.info("Using ComfyUI's VAEEncode node")
                        # Note: VAEEncode expects BCHW, but should handle conversion internally
                        latent = vae_encoder.encode(vae, cropped_image)[0]
                        logger.info(f"VAE encoding successful: {latent['samples'].shape}")
                        
                        # Apply KSampler
                        logger.info(f"Applying KSampler with steps={steps}, cfg={cfg}, denoise={denoise}")
                        sampled_latent = self.apply_ksampler(
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            latent=latent,
                            seed=seed,
                            steps=steps,
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
                
                # Apply bicubic scaling (model-based upscaling to be reimplemented later)
                scaled_image = process_with_model(
                    source_image, 
                    upscale_model, 
                    target_size,
                    scale_factor=1.0,  # Already applied in target calculation
                    divisibility=div_factor,
                    tile_size=tile_size,
                    overlap=overlap
                )
                
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
                    mask_bchw = scaled_mask.permute(0, 3, 1, 2)
                    fixed_mask = F.interpolate(
                        mask_bchw,
                        size=(scaled_image.shape[1], scaled_image.shape[2]),
                        mode='nearest'
                    )
                    scaled_mask = fixed_mask.permute(0, 2, 3, 1)
                
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
                # Continue with the next region
                continue
        
        logger.info(f"Processing complete - returning {len(processed_images)} images")
        
        # Make sure we're not returning empty lists
        if not processed_images:
            # Return input image as fallback
            logger.warning("No processed images found, using input image as fallback")
            processed_images.append(image)
            processed_masks.append(torch.ones((batch, height, width, 1), dtype=torch.float32))
            
        return processed_images, processed_masks, segs, cropped_images, bypass_image, bbox_count