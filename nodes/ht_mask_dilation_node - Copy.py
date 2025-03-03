"""
File: homage_tools/nodes/ht_mask_dilation_node.py
Version: 2.0.0
Description: Node for cropping images to mask content and calculating scaling factors
"""

import torch
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "2.0.0"
STANDARD_BUCKETS = [512, 768, 1024]

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def find_mask_bounds(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Find the bounding box of non-zero mask content."""
    # Debug mask information
    print(f"DEBUG: Mask shape: {mask.shape}")
    print(f"DEBUG: Mask min/max values: {mask.min().item():.6f}/{mask.max().item():.6f}")
    
    # Extract first batch and channel for calculation
    mask_2d = mask[0, ..., 0]
    print(f"DEBUG: Extracted 2D mask shape: {mask_2d.shape}")
    
    # Find non-zero indices
    indices = torch.nonzero(mask_2d > 0)
    print(f"DEBUG: Non-zero indices count: {len(indices)}")
    
    if len(indices) == 0:
        print("DEBUG: Empty mask detected")
        return 0, mask.shape[1], 0, mask.shape[2]
    
    # Get bounds
    min_y = indices[:, 0].min().item()
    max_y = indices[:, 0].max().item() + 1
    min_x = indices[:, 1].min().item()
    max_x = indices[:, 1].max().item() + 1
    
    print(f"DEBUG: Bounds - Y: {min_y} to {max_y}, X: {min_x} to {max_x}")
    return min_y, max_y, min_x, max_x

def calculate_target_size(bbox_size: int, scale_mode: str) -> int:
    """Calculate target size based on scale mode."""
    print(f"DEBUG: Original size: {bbox_size}, Mode: {scale_mode}")
    
    if scale_mode == "Scale Closest":
        target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
    elif scale_mode == "Scale Up":
        target = next((x for x in STANDARD_BUCKETS if x >= bbox_size), STANDARD_BUCKETS[-1])
    elif scale_mode == "Scale Down":
        target = next((x for x in reversed(STANDARD_BUCKETS) if x <= bbox_size), STANDARD_BUCKETS[0])
    elif scale_mode == "Scale Max":
        target = STANDARD_BUCKETS[-1]
    else:
        target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
    
    print(f"DEBUG: Target size: {target}")
    return target

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTMaskDilationNode:
    """Node for cropping images to mask content and calculating scaling factors."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_mask"
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("dilated_mask", "cropped_image", "width", "height", "scale_factor")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale_mode": (["Scale Closest", "Scale Up", "Scale Down", "Scale Max"], {
                    "default": "Scale Closest"
                }),
                "padding": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8
                })
            }
        }

    def process_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        scale_mode: str,
        padding: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]:
        """Process mask and image based on mask content."""
        print(f"\nHTMaskDilationNode v{VERSION} - Processing")
        
        try:
            # Verify input shapes
            print(f"DEBUG: Original image shape: {image.shape}")
            print(f"DEBUG: Original mask shape: {mask.shape}")
            
            # Ensure BHWC format
            if len(image.shape) == 3:  # HWC format
                image = image.unsqueeze(0)  # Add batch -> BHWC
                print(f"DEBUG: Converted image to BHWC: {image.shape}")
                
            if len(mask.shape) == 3:  # HWC format
                mask = mask.unsqueeze(0)  # Add batch -> BHWC
                print(f"DEBUG: Converted mask to BHWC: {mask.shape}")
            
            # Print detailed tensor info
            print(f"DEBUG: Image: type={type(image)}, dtype={image.dtype}")
            print(f"DEBUG: Mask: type={type(mask)}, dtype={mask.dtype}")
            
            # Force expected format if needed
            if mask.shape[-1] != 1 and len(mask.shape) == 4:
                print(f"DEBUG: Fixing unexpected mask format from {mask.shape}")
                mask = mask.permute(0, 2, 3, 1)  # BCHW → BHWC
                print(f"DEBUG: New mask shape: {mask.shape}")
            
            # Find mask bounds
            min_y, max_y, min_x, max_x = find_mask_bounds(mask)
            
            # Verify bounds are sensible
            print(f"DEBUG: Image dimensions: {image.shape[1]}x{image.shape[2]}")
            print(f"DEBUG: Bounds check - X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
            
            # Calculate bounding box dimensions
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            long_edge = max(bbox_width, bbox_height)
            
            print(f"DEBUG: Bounding box size: {bbox_width}x{bbox_height}")
            print(f"DEBUG: Long edge: {long_edge}")
            
            # Empty mask check
            if bbox_width <= 1 or bbox_height <= 1:
                print("WARNING: Nearly empty mask detected. Using full image.")
                return (mask, image, image.shape[2], image.shape[1], 1.0)
            
            # Calculate target size and scale factor
            target_size = calculate_target_size(long_edge, scale_mode)
            scale_factor = target_size / long_edge
            
            print(f"DEBUG: Scale factor: {scale_factor:.4f}")
            
            # Crop image and mask to the mask content bounds
            cropped_image = image[:, min_y:max_y, min_x:max_x, :]
            cropped_mask = mask[:, min_y:max_y, min_x:max_x, :]
            
            print(f"DEBUG: Cropped image shape: {cropped_image.shape}")
            print(f"DEBUG: Cropped mask shape: {cropped_mask.shape}")
            
            # Check dimensions match expectations
            if cropped_image.shape[1] != bbox_height or cropped_image.shape[2] != bbox_width:
                print(f"WARNING: Dimension mismatch. Expected {bbox_height}x{bbox_width}, got {cropped_image.shape[1]}x{cropped_image.shape[2]}")
            
            # Return with the correct dimensions
            print(f"DEBUG: Target dimensions would be: {int(bbox_width * scale_factor)}x{int(bbox_height * scale_factor)}")
            # Important: return the actual cropped dimensions, not tensor shape indices
            return (cropped_mask, cropped_image, bbox_width, bbox_height, scale_factor)
            
        except Exception as e:
            logger.error(f"Error in mask dilation: {str(e)}")
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return (mask, image, image.shape[2], image.shape[1], 1.0)