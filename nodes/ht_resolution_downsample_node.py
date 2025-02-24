"""
File: homage_tools/nodes/ht_resolution_downsample_node.py
Version: 1.0.1
Description: Node for downsampling images to a target resolution with BHWC tensor handling

Sections:
1. Imports and Type Definitions
2. Memory Management Functions
3. Resolution Calculation
4. Processing Implementation
5. Node Class Definition
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Memory Management Functions
#------------------------------------------------------------------------------
def estimate_memory_requirement(
    height: int,
    width: int,
    channels: int,
    target_height: int,
    target_width: int,
    dtype: torch.dtype
) -> int:
    """
    Estimate memory requirement for processing.
    
    Args:
        height: Original image height
        width: Original image width
        channels: Number of channels
        target_height: Target height after downsampling
        target_width: Target width after downsampling
        dtype: Tensor data type
        
    Returns:
        int: Estimated memory requirement in bytes
    """
    # Calculate sizes
    input_size = height * width * channels
    output_size = target_height * target_width * channels
    
    # Account for intermediate tensors (2x for safe measure)
    total_elements = (input_size + output_size) * 2
    
    # Calculate bytes per element
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    return total_elements * bytes_per_element

#------------------------------------------------------------------------------
# Section 3: Resolution Calculation
#------------------------------------------------------------------------------
def calculate_target_dimensions(
    current_height: int,
    current_width: int,
    target_long_edge: int
) -> Tuple[int, int, float]:
    """
    Calculate target dimensions maintaining aspect ratio.
    
    Args:
        current_height: Current image height
        current_width: Current image width
        target_long_edge: Desired length of longest edge
        
    Returns:
        Tuple[int, int, float]: (target_height, target_width, scale_factor)
    """
    # Determine current long edge and aspect ratio
    current_long_edge = max(current_height, current_width)
    aspect_ratio = current_width / current_height
    
    # Calculate scale factor
    scale_factor = target_long_edge / current_long_edge
    
    # Calculate new dimensions
    if current_width >= current_height:
        new_width = target_long_edge
        new_height = int(round(new_width / aspect_ratio))
    else:
        new_height = target_long_edge
        new_width = int(round(new_height * aspect_ratio))
        
    return new_height, new_width, scale_factor

#------------------------------------------------------------------------------
# Section 4: Processing Implementation
#------------------------------------------------------------------------------
def process_downsample(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """
    Process downsampling operation.
    
    Args:
        image: Input image tensor (BHWC)
        target_height: Target height
        target_width: Target width
        interpolation: Interpolation method
        device: Processing device
        
    Returns:
        torch.Tensor: Downsampled image
    """
    # Move to processing device
    image = image.to(device)
    
    # Convert BHWC to BCHW for processing
    image = image.permute(0, 3, 1, 2)
    
    # Determine antialiasing based on interpolation mode
    use_antialias = interpolation in ['bilinear', 'bicubic', 'lanczos']
    
    # Handle lanczos mode (PyTorch doesn't have native lanczos)
    if interpolation == 'lanczos':
        interpolation = 'bicubic'  # Use bicubic as closest approximation
    
    # Perform downsampling
    result = F.interpolate(
        image,
        size=(target_height, target_width),
        mode=interpolation,
        antialias=use_antialias if use_antialias else None,
        align_corners=None if interpolation == 'nearest' else False
    )
    
    # Convert back to BHWC
    result = result.permute(0, 2, 3, 1)
    
    return result

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
#------------------------------------------------------------------------------
class HTResolutionDownsampleNode:
    """
    Downsamples images to a target resolution while maintaining aspect ratio.
    Uses the longest edge as the target dimension reference.
    """
    
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
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu"
                })
            }
        }

    def downsample_to_resolution(
        self,
        image: torch.Tensor,
        target_long_edge: int,
        interpolation: str,
        device: str
    ) -> Tuple[torch.Tensor]:
        """
        Downsample image to target resolution.
        
        Args:
            image: Input image tensor (BHWC format)
            target_long_edge: Desired length of longest edge
            interpolation: Interpolation method
            device: Processing device
            
        Returns:
            Tuple[torch.Tensor]: Downsampled image
        """
        try:
            # Ensure BHWC format
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                
            # Get current dimensions (BHWC format)
            _, current_height, current_width, channels = image.shape
            logger.debug(f"Input image shape (BHWC): {image.shape}")
            
            # Calculate target dimensions
            target_height, target_width, scale_factor = calculate_target_dimensions(
                current_height, current_width, target_long_edge
            )
            logger.debug(f"Target dimensions: {target_width}x{target_height} (scale: {scale_factor:.3f})")
            
            # Skip processing if no change needed
            if target_height == current_height and target_width == current_width:
                return (image,)
                
            # Set processing device
            proc_device = torch.device(
                "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            
            # Process downsampling
            result = process_downsample(
                image, target_height, target_width, interpolation, proc_device
            )
            
            # Cleanup
            if proc_device.type == "cuda":
                torch.cuda.empty_cache()
                
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in resolution downsample: {str(e)}")
            return (image,)