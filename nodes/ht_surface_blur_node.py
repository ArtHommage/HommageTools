"""
File: homage_tools/nodes/ht_surface_blur_node.py
Version: 1.0.1
Description: Node for applying surface blur with edge preservation

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Node Class Definition
4. Blur Processing Logic
5. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def calculate_color_distance(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Calculate color distance weights between center pixel and neighbors.
    
    Args:
        center: Center pixel values
        neighbors: Neighboring pixel values
        threshold: Color difference threshold
        
    Returns:
        torch.Tensor: Distance weights
    """
    diff = torch.abs(neighbors - center.unsqueeze(2).unsqueeze(3))
    diff = torch.mean(diff, dim=1)  # Average across color channels
    weights = torch.exp(-diff / threshold)
    return weights

def normalize_kernel(kernel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize kernel weights to sum to 1.
    
    Args:
        kernel: Input kernel weights
        eps: Small value to prevent division by zero
        
    Returns:
        torch.Tensor: Normalized kernel
    """
    return kernel / (torch.sum(kernel, dim=(-1, -2), keepdim=True) + eps)

def normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor to BCHW format.
    
    Args:
        image: Input image tensor (BCHW or BHWC format)
        
    Returns:
        torch.Tensor: Normalized tensor in BCHW format
    """
    # Add batch dimension if needed
    if image.ndim == 3:
        # Check if it's HWC format
        if image.shape[-1] in [1, 3, 4]:
            # Convert HWC to BCHW
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            # Already in CHW format
            image = image.unsqueeze(0)
    elif image.ndim == 4:
        # Check if it's BHWC format
        if image.shape[-1] in [1, 3, 4]:
            # Convert BHWC to BCHW
            image = image.permute(0, 3, 1, 2)
            
    return image

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTSurfaceBlurNode:
    """
    Applies surface blur with edge preservation.
    Similar to Photoshop's Surface Blur filter.
    """
    
    CATEGORY = "HommageTools/Filters"
    FUNCTION = "apply_surface_blur"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blurred_image",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "description": "Blur radius in pixels"
                }),
                "threshold": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 0.1,
                    "description": "Color difference threshold"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 4: Blur Processing Logic
#------------------------------------------------------------------------------
    def process_surface_blur(
        self,
        image: torch.Tensor,
        radius: int,
        threshold: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Apply surface blur to image.
        
        Args:
            image: Input image tensor [B,C,H,W]
            radius: Blur radius
            threshold: Color difference threshold
            device: Processing device
            
        Returns:
            torch.Tensor: Blurred image
        """
        # Normalize threshold to 0-1 range
        threshold = threshold / 255.0
        
        # Print diagnostic information
        print(f"Processing tensor with shape: {image.shape}")
        print(f"Radius: {radius}, Threshold: {threshold}")
        
        # Create sampling grid for neighbors
        kernel_size = 2 * radius + 1
        x = torch.linspace(-radius, radius, kernel_size, device=device)
        y = torch.linspace(-radius, radius, kernel_size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Prepare padding
        pad = radius
        padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
        
        # Process image
        b, c, h, w = image.shape
        result = torch.zeros_like(image)
        
        # Diagnostic print
        print(f"Processing with dimensions: B={b}, C={c}, H={h}, W={w}")
        print(f"Kernel size: {kernel_size}x{kernel_size}")
        print(f"Padding: {pad}")
        
        import time
        start_time = time.time()
        total_pixels = h * w
        processed_pixels = 0
        last_update = time.time()
        update_interval = 2.0  # Update every 2 seconds

        print(f"\nStarting surface blur processing...")
        print(f"Total pixels to process: {total_pixels:,}")

        for i in range(h):
            for j in range(w):
                # Update progress
                processed_pixels += 1
                current_time = time.time()
                if current_time - last_update > update_interval:
                    progress = (processed_pixels / total_pixels) * 100
                    elapsed_time = current_time - start_time
                    pixels_per_second = processed_pixels / elapsed_time
                    remaining_pixels = total_pixels - processed_pixels
                    estimated_remaining = remaining_pixels / pixels_per_second if pixels_per_second > 0 else 0
                    
                    print(f"Progress: {progress:.1f}% | "
                          f"Processed: {processed_pixels:,}/{total_pixels:,} pixels | "
                          f"Elapsed: {elapsed_time:.1f}s | "
                          f"Remaining: {estimated_remaining:.1f}s")
                    last_update = current_time

                # Extract local region
                patch = padded[
                    :, :,
                    i:i + kernel_size,
                    j:j + kernel_size
                ]
                
                # Get center pixel
                center = image[:, :, i:i+1, j:j+1]
                
                # Calculate weights based on color difference
                weights = calculate_color_distance(
                    center, patch, threshold
                )
                
                # Apply spatial weighting
                spatial_weight = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * radius * radius))
                weights = weights * spatial_weight.to(device)
                
                # Normalize and apply weights
                weights = normalize_kernel(weights)
                weighted_sum = torch.sum(
                    patch * weights.unsqueeze(1),
                    dim=(-1, -2)
                )
                
                result[:, :, i, j] = weighted_sum
        
        # Final progress update
        total_time = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average processing speed: {total_pixels/total_time:.0f} pixels/second")

        return result

#------------------------------------------------------------------------------
# Section 5: Error Handling
#------------------------------------------------------------------------------
    def apply_surface_blur(
        self,
        image: torch.Tensor,
        radius: int,
        threshold: float
    ) -> Tuple[torch.Tensor]:
        """
        Main processing function with error handling.
        
        Args:
            image: Input image tensor
            radius: Blur radius
            threshold: Color difference threshold
            
        Returns:
            Tuple[torch.Tensor]: Processed image
        """
        try:
            device = image.device
            print(f"\nInput tensor shape: {image.shape}")
            
            # Normalize tensor format
            normalized_image = normalize_image_tensor(image)
            print(f"Normalized tensor shape: {normalized_image.shape}")
            
            # Store original format info
            was_bhwc = image.shape[-1] in [1, 3, 4] and image.ndim == 4
            was_3d = image.ndim == 3
            
            # Process image
            result = self.process_surface_blur(
                normalized_image, radius, threshold, device
            )
            
            # Restore original format
            if was_bhwc:
                result = result.permute(0, 2, 3, 1)
            if was_3d and result.shape[0] == 1:
                if was_bhwc:
                    result = result.squeeze(0)
                else:
                    result = result.squeeze(0)
            
            print(f"Output tensor shape: {result.shape}")
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in surface blur: {str(e)}")
            print(f"Error details: {str(e)}")
            print(f"Input tensor shape: {image.shape}")
            return (image,)  # Return original image on error