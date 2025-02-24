"""
File: homage_tools/nodes/ht_mask_dilation_node.py
Version: 1.6.3
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
VERSION = "1.6.3"

#------------------------------------------------------------------------------
# Section 2: Dimension Handling
#------------------------------------------------------------------------------
def verify_mask_dimensions(mask: torch.Tensor, context: str) -> Tuple[int, int, int, int, torch.Tensor]:
    """
    Verify and normalize tensor dimensions.
    Ensures output is in BHWC format for processing.
    
    Args:
        mask: Input tensor
        context: Context string for logging
        
    Returns:
        Tuple[int, int, int, int, torch.Tensor]: batch, height, width, channels, normalized tensor
    """
    shape = mask.shape
    print(f"DEBUG {context} - Initial shape: {shape}")
    
    # Ensure CPU tensor
    mask = mask.cpu()
    
    if len(shape) == 3:  # HWC format
        height, width, channels = shape
        batch = 1
        mask = mask.unsqueeze(0)  # Add batch dimension -> BHWC
    elif len(shape) == 4:  # BHWC format
        batch, height, width, channels = shape
    else:
        raise ValueError(f"Invalid tensor shape: {shape}")
        
    print(f"DEBUG {context} - Final BHWC: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels, mask

#------------------------------------------------------------------------------
# Section 3: Mask Processing Functions
#------------------------------------------------------------------------------
def find_mask_bounds(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of non-zero mask content.
    
    Args:
        mask: Input mask in BHWC format
        
    Returns:
        Tuple[int, int, int, int]: ymin, ymax, xmin, xmax
    """
    # Get first batch and channel
    mask_2d = mask[0, ..., 0]
    
    # Get indices of non-zero elements
    indices = torch.nonzero(mask_2d)
    
    if len(indices) == 0:  # No content found
        return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
    
    # Extract min/max for each dimension
    ymin = indices[:, 0].min().item()
    ymax = indices[:, 0].max().item() + 1
    xmin = indices[:, 1].min().item()
    xmax = indices[:, 1].max().item() + 1
    
    print(f"Found active mask region: Y({ymin}:{ymax}), X({xmin}:{xmax})")
    return ymin, ymax, xmin, xmax

def apply_dilation(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    Apply dilation to mask in BHWC format.
    
    Args:
        mask: Input mask [B,H,W,C]
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations
        
    Returns:
        torch.Tensor: Dilated mask [B,H,W,C]
    """
    # Convert to BCHW for processing
    mask = mask.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
    # Create kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    
    # Process each channel
    channels = []
    for c in range(mask.shape[1]):
        channel = mask[:, c:c+1]  # Keep 4D: [B,1,H,W]
        
        result = channel
        for _ in range(iterations):
            # Apply padding and convolution
            padded = F.pad(result, (kernel_size//2,)*4, mode='reflect')
            dilated = F.conv2d(padded, kernel)
            result = (dilated > 0).float()
            
        channels.append(result)
    
    # Combine channels
    result = torch.cat(channels, dim=1)
    
    # Convert back to BHWC
    return result.permute(0, 2, 3, 1)

def apply_smoothing(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Apply smoothing to mask in BHWC format.
    
    Args:
        mask: Input mask [B,H,W,C]
        kernel_size: Size of smoothing kernel
        
    Returns:
        torch.Tensor: Smoothed mask [B,H,W,C]
    """
    # Convert to BCHW for processing
    mask = mask.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
    # Create smoothing kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel / kernel.sum()
    
    # Process each channel
    channels = []
    for c in range(mask.shape[1]):
        channel = mask[:, c:c+1]  # Keep 4D: [B,1,H,W]
        
        # Apply smoothing
        padded = F.pad(channel, (kernel_size//2,)*4, mode='reflect')
        smoothed = F.conv2d(padded, kernel)
        channels.append(smoothed)
    
    # Combine channels and convert back to BHWC
    result = torch.cat(channels, dim=1)
    return result.permute(0, 2, 3, 1)

#------------------------------------------------------------------------------
# Section 4: Training Bucket Calculations
#------------------------------------------------------------------------------
def calculate_training_dimensions(
    width: int,
    height: int,
    scaling_mode: str,
    divisor: int,
    standard_dims: list
) -> Tuple[int, int, float]:
    """Calculate dimensions for training buckets."""
    long_edge = max(width, height)
    aspect_ratio = width / height
    
    if scaling_mode == "upscale_only" and long_edge >= standard_dims[-1]:
        return width, height, 1.0

    target_long = min([d for d in standard_dims if d >= long_edge], 
                     default=min(1024, standard_dims[-1]))
                     
    if width >= height:
        target_width = target_long
        target_height = int(round(target_width / aspect_ratio / divisor) * divisor)
    else:
        target_height = target_long
        target_width = int(round(target_height * aspect_ratio / divisor) * divisor)
        
    scale_factor = target_long / long_edge
    print(f"Target dimensions: {target_width}x{target_height} (scale: {scale_factor:.3f})")
    return target_width, target_height, scale_factor

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
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
                "dilation_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "use_training_buckets": ("BOOLEAN", {
                    "default": True
                }),
                "scaling_mode": (["upscale_only", "both"], {
                    "default": "upscale_only"
                }),
                "divisible_by": (["8", "64"], {
                    "default": "64"
                }),
                "smoothing_passes": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1
                }),
                "margin": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 64,
                    "step": 8
                }),
                "dilation_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    def process_mask(
        self,
        mask: torch.Tensor,
        dilation_factor: float,
        use_training_buckets: bool,
        scaling_mode: str,
        divisible_by: str,
        smoothing_passes: int,
        margin: int,
        dilation_iterations: int,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, float]:
        """
        Process mask with dilation and optional image cropping.
        
        Args:
            mask: Input mask in BHWC format
            dilation_factor: Factor for dilation strength
            use_training_buckets: Whether to use standard training dimensions
            scaling_mode: How to handle scaling
            divisible_by: Dimension divisibility requirement
            smoothing_passes: Number of smoothing iterations
            margin: Margin around mask bounds
            dilation_iterations: Number of dilation iterations
            image: Optional input image to crop
            
        Returns:
            Tuple containing:
            - Processed mask
            - Cropped and processed image (if provided)
            - Output width
            - Output height
            - Scale factor
        """
        try:
            print(f"\nHTMaskDilationNode v{VERSION} - Processing")
            
            # Verify and normalize mask dimensions
            batch, height, width, channels, mask = verify_mask_dimensions(mask, "Input Mask")
            print(f"Mask range: min={mask.min():.3f}, max={mask.max():.3f}")
            
            # Find mask bounds
            y_min, y_max, x_min, x_max = find_mask_bounds(mask)
            print(f"Initial bounds: Y({y_min}:{y_max}), X({x_min}:{x_max})")
            
            # Add margin
            y_min = max(0, y_min - margin)
            y_max = min(height, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(width, x_max + margin)
            print(f"Bounds with margin: Y({y_min}:{y_max}), X({x_min}:{x_max})")
            
            # Crop mask to bounds
            mask_cropped = mask[:, y_min:y_max, x_min:x_max, :]
            print(f"Cropped mask shape: {mask_cropped.shape}")
            
            # Apply processing to cropped mask
            processed_mask = mask_cropped.clone()
            
            if smoothing_passes > 0:
                processed_mask = apply_smoothing(processed_mask, kernel_size=3)
                
            if dilation_iterations > 0:
                processed_mask = apply_dilation(
                    processed_mask,
                    kernel_size=3,
                    iterations=dilation_iterations
                )
            
            # Process image if provided
            processed_image = None
            if image is not None:
                # Verify and normalize image dimensions
                i_batch, i_height, i_width, i_channels, image = verify_mask_dimensions(
                    image, "Input Image"
                )
                
                # Verify dimensions match
                if i_height != height or i_width != width:
                    raise ValueError(
                        f"Image dimensions ({i_width}x{i_height}) don't match "
                        f"mask dimensions ({width}x{height})"
                    )
                
                # Crop image to same bounds
                processed_image = image[:, y_min:y_max, x_min:x_max, :]
                print(f"Cropped image shape: {processed_image.shape}")
                
                # Apply mask
                processed_image = processed_image * processed_mask
            
            # Get output dimensions
            out_height = y_max - y_min
            out_width = x_max - x_min
            print(f"Output dimensions: {out_width}x{out_height}")
            
            # Remove batch dimension if added
            if batch == 1 and len(mask.shape) == 3:
                processed_mask = processed_mask.squeeze(0)
                if processed_image is not None:
                    processed_image = processed_image.squeeze(0)
            
            return (processed_mask, processed_image, out_width, out_height, dilation_factor)
            
        except Exception as e:
            logger.error(f"Error in mask dilation: {str(e)}")
            print(f"Error details: {str(e)}")
            return (mask, image, width, height, 1.0)