"""
File: homage_tools/nodes/ht_resolution_downsample_node.py
Description: Node for downsampling images to a target resolution with BHWC tensor handling
Version: 1.1.0
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
VERSION = "1.1.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_dimensions(tensor: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """Verify and extract dimensions from BHWC tensor."""
    shape = tensor.shape
    print(f"{context} - Tensor shape: {shape}")
    
    if len(shape) == 3:  # HWC format
        height, width, channels = shape
        batch = 1
        print(f"{context} - HWC format detected")
    elif len(shape) == 4:  # BHWC format
        batch, height, width, channels = shape
        print(f"{context} - BHWC format detected")
    else:
        raise ValueError(f"Invalid tensor shape: {shape}")
        
    print(f"{context} - Dimensions: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def calculate_target_dimensions(
    current_height: int,
    current_width: int,
    target_long_edge: int
) -> Tuple[int, int, float]:
    """Calculate target dimensions maintaining aspect ratio."""
    current_long_edge = max(current_height, current_width)
    aspect_ratio = current_width / current_height
    
    scale_factor = target_long_edge / current_long_edge
    print(f"Scale factor: {scale_factor:.3f}")
    
    if current_width >= current_height:
        new_width = target_long_edge
        new_height = int(round(new_width / aspect_ratio))
    else:
        new_height = target_long_edge
        new_width = int(round(new_height * aspect_ratio))
        
    print(f"Target dimensions: {new_width}x{new_height}")
    return new_height, new_width, scale_factor

#------------------------------------------------------------------------------
# Section 3: Processing Functions
#------------------------------------------------------------------------------
def process_downsample(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """Process downsampling operation maintaining BHWC format."""
    print(f"Processing downsample: shape={image.shape} (BHWC)")
    print(f"Target size: {target_width}x{target_height}")
    
    # Move to processing device
    image = image.to(device)
    
    # Convert BHWC to BCHW for interpolation
    x = image.permute(0, 3, 1, 2)
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
    
    # Convert back to BHWC
    result = result.permute(0, 2, 3, 1)
    print(f"Converted back to BHWC: shape={result.shape}")
    
    return result

#------------------------------------------------------------------------------
# Section 4: Node Definition
#------------------------------------------------------------------------------
class HTResolutionDownsampleNode:
    """Downsamples images to a target resolution while maintaining aspect ratio."""
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "downsample_to_resolution"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("downsampled_image",)
    
    INTERPOLATION_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "target_long_edge": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "description": "Target size for longest edge"
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "bicubic",
                    "description": "Interpolation method"
                })
            }
        }

    def downsample_to_resolution(
        self,
        image: torch.Tensor,
        target_long_edge: int,
        interpolation: str
    ) -> Tuple[torch.Tensor]:
        """Downsample image to target resolution maintaining BHWC format."""
        print(f"\nHTResolutionDownsampleNode v{VERSION} - Processing")
        
        try:
            # Verify input tensor
            batch, height, width, channels = verify_tensor_dimensions(image, "Input")
            print(f"Value range: min={image.min():.3f}, max={image.max():.3f}")
            
            # Calculate target dimensions
            target_height, target_width, scale_factor = calculate_target_dimensions(
                height, width, target_long_edge
            )
            
            # Skip processing if no change needed
            if target_height == height and target_width == width:
                print("No resizing needed - dimensions match target")
                return (image,)
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Processing device: {device}")
            
            # Process image
            result = process_downsample(
                image, target_height, target_width, interpolation, device
            )
            
            # Verify output
            print(f"\nOutput tensor: shape={result.shape} (BHWC)")
            print(f"Value range: min={result.min():.3f}, max={result.max():.3f}")
            
            # Clean up
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in resolution downsample: {str(e)}")
            print(f"Error details: {str(e)}")
            return (image,)