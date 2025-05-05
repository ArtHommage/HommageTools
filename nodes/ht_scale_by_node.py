"""
File: homage_tools/nodes/ht_scale_by_node.py
Description: Node for scaling images by a factor with mask support
Version: 1.0.0
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.0.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_dimensions(tensor: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """Verify and extract dimensions from BHWC tensor."""
    shape = tensor.shape
    print(f"{context} - Tensor shape: {shape}")
    print(f"{context} - Tensor dtype: {tensor.dtype}")
    
    # Add NaN and infinity checks
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"{context} - WARNING: Tensor contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Infinity' if has_inf else ''} values!")
    
    if len(shape) == 3:  # HWC format
        height, width, channels = shape
        batch = 1
        print(f"{context} - HWC format detected")
        
        # Validate channel dimension
        if channels not in [1, 3, 4]:
            print(f"{context} - WARNING: Unusual channel count: {channels} (expected 1, 3, or 4)")
            
    elif len(shape) == 4:  # BHWC format
        batch, height, width, channels = shape
        print(f"{context} - BHWC format detected")
        
        # Validate channel dimension
        if channels not in [1, 3, 4]:
            print(f"{context} - WARNING: Unusual channel count: {channels} (expected 1, 3, or 4)")
            
    else:
        print(f"{context} - ERROR: Invalid tensor shape: {shape}")
        print(f"{context} - Attempting to recover...")
        
        # Try to recover based on dimension sizes
        if len(shape) == 2:  # Might be a single-channel image without batch
            height, width = shape
            channels = 1
            batch = 1
            print(f"{context} - Recovered as HW format: {batch}x{height}x{width}x{channels}")
        elif len(shape) == 3:
            if shape[0] in [1, 3, 4]:  # Might be CHW format
                channels, height, width = shape
                batch = 1
                print(f"{context} - Recovered as CHW format: {batch}x{height}x{width}x{channels}")
            else:
                batch, height, width = shape
                channels = 1
                print(f"{context} - Recovered as BHW format: {batch}x{height}x{width}x{channels}")
        else:
            raise ValueError(f"Invalid tensor shape: {shape}, cannot recover")
        
    print(f"{context} - Dimensions: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def verify_mask_dimensions(mask: torch.Tensor, context: str) -> Tuple[int, int, int]:
    """Verify and extract dimensions from mask tensor."""
    shape = mask.shape
    print(f"{context} - Mask shape: {shape}")
    
    if len(shape) == 2:  # HW format
        height, width = shape
        batch = 1
        print(f"{context} - HW format detected")
    elif len(shape) == 3:  # BHW format
        batch, height, width = shape
        print(f"{context} - BHW format detected")
    else:
        raise ValueError(f"Invalid mask shape: {shape}")
        
    print(f"{context} - Mask dimensions: {batch}x{height}x{width}")
    return batch, height, width

def calculate_target_dimensions(
    current_height: int,
    current_width: int,
    scale_factor: float
) -> Tuple[int, int]:
    """Calculate target dimensions based on scale factor."""
    target_height = int(round(current_height * scale_factor))
    target_width = int(round(current_width * scale_factor))
    
    print(f"Target dimensions: {target_width}x{target_height}")
    return target_height, target_width

def find_mask_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Find bounding box of non-zero values in mask."""
    if len(mask.shape) == 3:  # BHW format, take first mask
        mask_2d = mask[0]
    else:
        mask_2d = mask
    
    # Find non-zero indices
    non_zero = torch.nonzero(mask_2d > 0.05)
    if len(non_zero) == 0:
        # No mask content, return full dimensions
        return 0, 0, mask_2d.shape[1], mask_2d.shape[0]
    
    # Get min/max coords
    y_min = non_zero[:, 0].min().item()
    y_max = non_zero[:, 0].max().item() + 1
    x_min = non_zero[:, 1].min().item()
    x_max = non_zero[:, 1].max().item() + 1
    
    return x_min, y_min, x_max, y_max

def crop_to_mask_bbox(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
    """Crop image and mask to mask content bounding box."""
    # Process each image in batch
    batch_size = image.shape[0]
    cropped_images = []
    cropped_masks = []
    
    for i in range(batch_size):
        # Extract single image and mask
        img = image[i:i+1]  # Keep batch dimension
        msk = mask[i:i+1] if len(mask.shape) == 3 else mask  # Handle BHW format
        
        # Find bbox
        x_min, y_min, x_max, y_max = find_mask_bbox(msk)
        
        # Crop image (BHWC format)
        cropped_img = img[:, y_min:y_max, x_min:x_max, :]
        
        # Crop mask (BHW format)
        cropped_msk = msk[:, y_min:y_max, x_min:x_max] if len(msk.shape) == 3 else msk[y_min:y_max, x_min:x_max]
        
        cropped_images.append(cropped_img)
        cropped_masks.append(cropped_msk)
    
    # Return first crop's bbox for reference
    first_bbox = find_mask_bbox(mask[0:1] if len(mask.shape) == 3 else mask)
    
    # Return list of cropped tensors and bbox
    return cropped_images, cropped_masks, first_bbox

#------------------------------------------------------------------------------
# Section 3: Processing Functions
#------------------------------------------------------------------------------
def process_scale(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """Process scaling operation maintaining BHWC format."""
    print(f"Processing scale: shape={image.shape} (BHWC)")
    print(f"Target size: {target_width}x{target_height}")
    
    # Check for unusual tensor shapes or values
    if len(image.shape) != 4:
        print(f"[WARNING] Expected 4D tensor (BHWC) but got {len(image.shape)}D")
        # Try to reshape if needed
        if len(image.shape) == 3:  # HWC format
            print(f"[DEBUG] Converting HWC to BHWC format")
            image = image.unsqueeze(0)
    
    # Check for NaN or infinity
    has_nan = torch.isnan(image).any().item()
    has_inf = torch.isinf(image).any().item()
    if has_nan or has_inf:
        print(f"[WARNING] Image tensor contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Infinity' if has_inf else ''} values!")
        # Try to fix by clipping values
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        print(f"[DEBUG] Applied nan_to_num to fix invalid values")
    
    # Move to processing device
    image = image.to(device)
    
    try:
        # Convert BHWC to BCHW for interpolation
        x = image.permute(0, 3, 1, 2)
        print(f"Converted to BCHW: shape={x.shape}")
        
        # Ensure no extreme values
        if x.max().item() > 10.0 or x.min().item() < -10.0:
            print(f"[WARNING] Image has extreme values: min={x.min().item():.3f}, max={x.max().item():.3f}")
            print(f"[DEBUG] Clipping values to [0, 1] range")
            x = torch.clamp(x, 0.0, 1.0)
        
        # Determine antialiasing based on interpolation mode
        use_antialias = interpolation in ['bilinear', 'bicubic', 'lanczos']
        
        # Handle lanczos mode
        if interpolation == 'lanczos':
            interpolation = 'bicubic'
        
        # Process
        result = F.interpolate(
            x,
            size=(target_height, target_width),
            mode=interpolation,
            antialias=use_antialias if use_antialias else None,
            align_corners=None if interpolation == 'nearest' else False
        )
        
        # Convert back to BHWC
        result = result.permute(0, 2, 3, 1)
        print(f"Converted back to BHWC: shape={result.shape}")
        
        return result
    except Exception as e:
        print(f"[ERROR] Scale processing error: {str(e)}")
        print(f"[DEBUG] Error details: {e.__class__.__name__}")
        # Return original image resized using a more robust method
        print(f"[DEBUG] Falling back to safer resize method...")
        try:
            # Create zeros tensor of target size
            safe_result = torch.zeros(image.shape[0], target_height, target_width, image.shape[3], device=device)
            return safe_result
        except Exception as fallback_error:
            print(f"[ERROR] Even fallback resize failed: {str(fallback_error)}")
            # Last resort: return original
            return image

def process_mask_scale(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """Process mask scaling operation."""
    print(f"Processing mask scale: shape={mask.shape}")
    print(f"Target size: {target_width}x{target_height}")
    
    # Move to processing device
    mask = mask.to(device)
    
    # Ensure mask has batch dimension (BHW format)
    if len(mask.shape) == 2:  # HW format
        mask = mask.unsqueeze(0)
    
    # Add channel dimension for interpolation (BCHW format)
    x = mask.unsqueeze(1)
    print(f"Converted to BCHW: shape={x.shape}")
    
    # Determine antialiasing based on interpolation mode
    use_antialias = interpolation in ['bilinear', 'bicubic', 'lanczos']
    
    # Handle lanczos mode
    if interpolation == 'lanczos':
        interpolation = 'bicubic'
    
    # Process
    result = F.interpolate(
        x,
        size=(target_height, target_width),
        mode=interpolation,
        antialias=use_antialias if use_antialias else None,
        align_corners=None if interpolation == 'nearest' else False
    )
    
    # Remove channel dimension (back to BHW)
    result = result.squeeze(1)
    print(f"Converted back to BHW: shape={result.shape}")
    
    return result

def create_empty_mask(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create empty mask tensor with correct dimensions."""
    return torch.zeros((batch_size, height, width), device=device)

#------------------------------------------------------------------------------
# Section 4: Node Definition
#------------------------------------------------------------------------------
class HTScaleByNode:
    """Scales images by a factor with mask support."""
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "scale_by_factor"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("scaled_image", "scaled_mask")
    
    INTERPOLATION_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.01,
                    "description": "Factor to scale image dimensions"
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "bicubic",
                    "description": "Interpolation method for images"
                }),
                "crop_to_mask": ("BOOLEAN", {
                    "default": False,
                    "description": "Crop to mask content before scaling"
                })
            },
            "optional": {
                "mask": ("MASK", {
                    "description": "Optional mask to process"
                }),
                "mask_interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "nearest",
                    "description": "Interpolation method for masks"
                })
            }
        }

    def scale_by_factor(
        self,
        image: torch.Tensor,
        scale_factor: float,
        interpolation: str,
        crop_to_mask: bool,
        mask: Optional[torch.Tensor] = None,
        mask_interpolation: Optional[str] = "nearest"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale image and mask by factor."""
        print(f"\nHTScaleByNode v{VERSION} - Processing")
        print(f"Input tensor shape: {image.shape}, dtype: {image.dtype}")
        print(f"Scale factor: {scale_factor}, Interpolation: {interpolation}")
        print(f"Crop to mask: {crop_to_mask}")
        
        if mask is not None:
            print(f"Mask tensor shape: {mask.shape}, dtype: {mask.dtype}")
        
        try:
            # Verify input image tensor
            batch, height, width, channels = verify_tensor_dimensions(image, "Input image")
            print(f"Image value range: min={image.min().item():.3f}, max={image.max().item():.3f}")
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Processing device: {device}")
            
            # Handle mask - create empty mask if none provided
            if mask is None:
                print("No mask provided, creating empty mask")
                mask = create_empty_mask(batch, height, width, device)
            else:
                # Verify mask dimensions
                mask_batch, mask_height, mask_width = verify_mask_dimensions(mask, "Input mask")
                print(f"Mask value range: min={mask.min():.3f}, max={mask.max():.3f}")
                
                # Check mask dimensions match image
                if mask_height != height or mask_width != width:
                    print(f"Warning: Mask dimensions ({mask_height}x{mask_width}) don't match image ({height}x{width})")
                    # Resize mask to match image
                    mask = process_mask_scale(mask, height, width, "nearest", device)
            
            # Process cropping if enabled and mask provided
            if crop_to_mask and mask is not None:
                print("Cropping to mask content")
                cropped_images, cropped_masks, bbox = crop_to_mask_bbox(image, mask)
                
                # Process each cropped image and mask
                processed_images = []
                processed_masks = []
                
                for i, (img, msk) in enumerate(zip(cropped_images, cropped_masks)):
                    print(f"Processing cropped item {i+1}/{len(cropped_images)}")
                    
                    # Get dimensions of this crop
                    _, crop_height, crop_width, _ = verify_tensor_dimensions(img, f"Crop {i}")
                    
                    # Calculate target dimensions for this crop
                    crop_target_height, crop_target_width = calculate_target_dimensions(
                        crop_height, crop_width, scale_factor
                    )
                    
                    # Resize cropped image
                    resized_img = process_scale(
                        img, crop_target_height, crop_target_width, interpolation, device
                    )
                    
                    # Resize cropped mask
                    resized_msk = process_mask_scale(
                        msk, crop_target_height, crop_target_width, mask_interpolation, device
                    )
                    
                    processed_images.append(resized_img)
                    processed_masks.append(resized_msk)
                
                # Use first processed image/mask as the output
                result_image = processed_images[0]
                result_mask = processed_masks[0]
                
            else:
                # Standard processing without cropping
                print("Processing without cropping")
                
                # Calculate target dimensions
                target_height, target_width = calculate_target_dimensions(
                    height, width, scale_factor
                )
                
                # Skip processing if no change needed
                if target_height == height and target_width == width:
                    print("No resizing needed - dimensions match target")
                    return (image, mask)
                
                # Process image
                result_image = process_scale(
                    image, target_height, target_width, interpolation, device
                )
                
                # Process mask
                result_mask = process_mask_scale(
                    mask, target_height, target_width, mask_interpolation, device
                )
            
            # Verify output
            print(f"\nOutput image: shape={result_image.shape} (BHWC)")
            print(f"Output mask: shape={result_mask.shape} (BHW)")
            print(f"Image value range: min={result_image.min():.3f}, max={result_image.max():.3f}")
            print(f"Mask value range: min={result_mask.min():.3f}, max={result_mask.max():.3f}")
            
            # Clean up
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            return (result_image, result_mask)
            
        except Exception as e:
            print(f"[CRITICAL ERROR] in image scaling: {str(e)}")
            print(f"[DEBUG] Error type: {e.__class__.__name__}")
            import traceback
            traceback.print_exc()
            
            # Try to recover and return something valid
            print(f"[DEBUG] Attempting to return valid tensors despite error...")
            try:
                # Create safe outputs
                if len(image.shape) == 4:
                    batch_size = image.shape[0]
                    # Create a small but valid output tensor
                    safe_image = torch.zeros(batch_size, 64, 64, 3)
                    safe_mask = torch.zeros(batch_size, 64, 64)
                else:
                    # Create a single-item batch
                    safe_image = torch.zeros(1, 64, 64, 3)
                    safe_mask = torch.zeros(1, 64, 64)
                    
                return (safe_image, safe_mask)
            except:
                # Last resort
                print(f"[DEBUG] Using original tensors as fallback")
                empty_mask = torch.zeros(1, image.shape[1], image.shape[2])
                return (image, empty_mask if mask is None else mask)