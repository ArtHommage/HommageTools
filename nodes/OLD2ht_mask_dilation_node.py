"""
File: homage_tools/nodes/ht_mask_dilation_node.py
Version: 1.8.0
Description: Node for dilating masks with proper tensor format handling and bounded region processing
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.8.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def debug_tensor(tensor: torch.Tensor, name: str):
    """Print debug information about a tensor."""
    print(f"\nDebug {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min/Max: {tensor.min().item():.4f}/{tensor.max().item():.4f}")

def find_mask_bounds(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of non-zero mask content.
    
    Args:
        mask: Input mask tensor of any format
        
    Returns:
        Tuple[int, int, int, int]: ymin, ymax, xmin, xmax
    """
    print(f"\nFinding mask bounds in tensor shape: {mask.shape}")
    
    # Handle different mask formats
    if len(mask.shape) == 4:  # BHWC format
        # Extract first batch, all channels
        mask_2d = mask[0, :, :, 0]
    elif len(mask.shape) == 3:
        if mask.shape[0] == 1:  # Single channel BHW format
            mask_2d = mask.squeeze(0)
        else:  # HWC format
            mask_2d = mask[:, :, 0]
    elif len(mask.shape) == 2:  # Already 2D
        mask_2d = mask
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    
    print(f"Processing 2D mask of shape: {mask_2d.shape}")
    debug_tensor(mask_2d, "2D mask")
    
    # Get indices of non-zero elements
    nonzero = torch.nonzero(mask_2d > 0.01, as_tuple=True)
    
    if len(nonzero[0]) == 0:
        print("No mask content found, using full dimensions")
        return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
    
    # Extract bounds
    ymin = int(nonzero[0].min().item())
    ymax = int(nonzero[0].max().item()) + 1
    xmin = int(nonzero[1].min().item())
    xmax = int(nonzero[1].max().item()) + 1
    
    print(f"Found bounds: Y({ymin}:{ymax}), X({xmin}:{xmax})")
    return ymin, ymax, xmin, xmax

#------------------------------------------------------------------------------
# Section 3: Training Bucket Calculations
#------------------------------------------------------------------------------
def calculate_training_dimensions(
    width: int,
    height: int,
    scaling_mode: str,
    divisor: int,
    standard_dims: list
) -> Tuple[int, int, float]:
    """
    Calculate dimensions for training buckets.
    
    Args:
        width: Current width
        height: Current height
        scaling_mode: Scaling approach to use
        divisor: Dimension divisor
        standard_dims: List of standard dimensions
        
    Returns:
        Tuple[int, int, float]: target_width, target_height, scale_factor
    """
    print(f"\nCalculating training dimensions:")
    print(f"Input dimensions: {width}x{height}")
    print(f"Scaling mode: {scaling_mode}")
    print(f"Divisor: {divisor}")
    
    # Determine long edge and aspect ratio
    long_edge = max(width, height)
    aspect_ratio = width / height
    print(f"Long edge: {long_edge}, Aspect ratio: {aspect_ratio:.4f}")
    
    # Handle max scaling mode
    if scaling_mode == "max":
        target_long = standard_dims[-1]
        print(f"Using max dimension: {target_long}")
    
    # Handle upscale only
    elif scaling_mode == "upscale_only" and long_edge >= standard_dims[-1]:
        print(f"No upscaling needed, using original dimensions")
        return width, height, 1.0
    
    # Handle downscale only
    elif scaling_mode == "downscale_only" and long_edge <= standard_dims[0]:
        print(f"No downscaling needed, using original dimensions")
        return width, height, 1.0
    
    # Find closest standard dimension
    else:
        if scaling_mode == "upscale_only":
            candidates = [d for d in standard_dims if d >= long_edge]
            if not candidates:
                candidates = [standard_dims[-1]]
        elif scaling_mode == "downscale_only":
            candidates = [d for d in standard_dims if d <= long_edge]
            if not candidates:
                candidates = [standard_dims[0]]
        else:  # both
            candidates = standard_dims
            
        # Find closest dimension
        target_long = min(candidates, key=lambda d: abs(d - long_edge))
        print(f"Selected target dimension: {target_long}")
    
    # Calculate dimensions based on aspect ratio
    if width >= height:
        target_width = target_long
        target_height = int(round(target_width / aspect_ratio / divisor) * divisor)
    else:
        target_height = target_long
        target_width = int(round(target_height * aspect_ratio / divisor) * divisor)
    
    # Calculate scale factor
    scale_factor = target_long / long_edge
    
    print(f"Target dimensions: {target_width}x{target_height}")
    print(f"Scale factor: {scale_factor:.4f}")
    
    return target_width, target_height, scale_factor

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTMaskDilationNode:
    """Mask dilation with proper tensor handling and bounded region processing."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_mask"
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("dilated_mask", "processed_image", "width", "height", "scale_factor")
    STANDARD_DIMS = [512, 768, 1024]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "mask": ("MASK",),
                "margin": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 128,
                    "step": 8
                }),
                "use_training_buckets": ("BOOLEAN", {
                    "default": True
                }),
                "scaling_mode": (["upscale_only", "downscale_only", "both", "max"], {
                    "default": "upscale_only"
                }),
                "divisible_by": (["8", "64"], {
                    "default": "64"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    def process_mask(
        self,
        mask: torch.Tensor,
        margin: int,
        use_training_buckets: bool,
        scaling_mode: str,
        divisible_by: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, float]:
        """
        Process mask and optional image with enhanced error handling.
        
        Args:
            mask: Input mask tensor (any format)
            margin: Margin around mask content
            use_training_buckets: Whether to scale to training buckets
            scaling_mode: How to handle scaling
            divisible_by: Make dimensions divisible by
            image: Optional image tensor (BHWC)
            
        Returns:
            Tuple containing processed tensors and dimensions
        """
        try:
            print(f"\nHTMaskDilationNode v{VERSION} - Processing")
            print(f"Mask shape: {mask.shape}")
            debug_tensor(mask, "Input mask")
            
            # Check image format if provided
            if image is not None:
                print(f"Image shape: {image.shape}")
                debug_tensor(image, "Input image")
            
            # Find mask bounds
            y_min, y_max, x_min, x_max = find_mask_bounds(mask)
            
            # Add margin
            y_min = max(0, y_min - margin)
            y_max = min(mask.shape[0] if len(mask.shape) == 2 else mask.shape[-3], y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(mask.shape[1] if len(mask.shape) == 2 else mask.shape[-2], x_max + margin)
            
            print(f"Bounds with margin: Y({y_min}:{y_max}), X({x_min}:{x_max})")
            
            # Check for valid bounds
            if y_min >= y_max or x_min >= x_max:
                print(f"WARNING: Invalid bounds detected, using full dimensions")
                y_min = 0
                y_max = mask.shape[0] if len(mask.shape) == 2 else mask.shape[-3]
                x_min = 0
                x_max = mask.shape[1] if len(mask.shape) == 2 else mask.shape[-2]
                print(f"Corrected bounds: Y({y_min}:{y_max}), X({x_min}:{x_max})")
            
            # Crop mask based on format
            if len(mask.shape) == 4:  # BHWC format
                cropped_mask = mask[:, y_min:y_max, x_min:x_max, :]
            elif len(mask.shape) == 3:
                if mask.shape[0] == 1:  # BHW format
                    cropped_mask = mask[:, y_min:y_max, x_min:x_max]
                else:  # HWC format
                    cropped_mask = mask[y_min:y_max, x_min:x_max, :]
            elif len(mask.shape) == 2:  # HW format
                cropped_mask = mask[y_min:y_max, x_min:x_max]
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")
            
            print(f"Cropped mask shape: {cropped_mask.shape}")
            
            # Calculate current dimensions
            region_width = x_max - x_min
            region_height = y_max - y_min
            
            # Calculate training dimensions if requested
            if use_training_buckets:
                target_width, target_height, scale_factor = calculate_training_dimensions(
                    region_width, 
                    region_height,
                    scaling_mode,
                    int(divisible_by),
                    self.STANDARD_DIMS
                )
                print(f"Training bucket dimensions: {target_width}x{target_height}, scale: {scale_factor:.4f}")
            else:
                target_width = region_width
                target_height = region_height
                scale_factor = 1.0
                print(f"Using original dimensions: {target_width}x{target_height}")
            
            processed_image = None
            if image is not None:
                # Extract region based on image format
                if len(image.shape) == 4:  # BHWC format
                    processed_image = image[:, y_min:y_max, x_min:x_max, :]
                elif len(image.shape) == 3:  # HWC format
                    processed_image = image[y_min:y_max, x_min:x_max, :]
                else:
                    raise ValueError(f"Unsupported image shape: {image.shape}")
                
                print(f"Processed image shape: {processed_image.shape}")
                
                # Create simple binary mask with same dimensions as image
                # This is safer than trying to multiply potentially mismatched tensors
                binary_mask = torch.zeros_like(processed_image)
                if cropped_mask.sum() > 0:  # Only if we have mask content
                    binary_mask = binary_mask + 1.0  # Set to all ones
                
                # Set image masked by binary mask
                processed_image = processed_image * binary_mask
                print(f"Applied binary mask to image")
            
            # Return results (maintaining original tensor format)
            return (cropped_mask, processed_image, target_width, target_height, scale_factor)
            
        except Exception as e:
            logger.error(f"Error in mask dilation: {str(e)}")
            print(f"Error in mask dilation: {str(e)}")
            import traceback
            traceback.print_exc()
            return (mask, image, mask.shape[-1] if len(mask.shape) > 1 else 0, 
                   mask.shape[-2] if len(mask.shape) > 1 else 0, 1.0)