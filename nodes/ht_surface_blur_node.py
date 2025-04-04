"""
File: homage_tools/nodes/ht_surface_blur_node.py
Description: Memory-efficient surface blur with tiled processing and CUDA optimization
Version: 1.4.2

Sections:
1. Imports and Configuration
2. Memory Management Functions
3. Tensor Validation and Debug
4. Processing Implementation
5. Node Class Definition
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
import logging

logger = logging.getLogger('HommageTools')
VERSION = "1.4.2"

#------------------------------------------------------------------------------
# Section 2: Memory Management Functions
#------------------------------------------------------------------------------
def calculate_memory_requirements(
    height: int,
    width: int,
    channels: int,
    radius: int,
    dtype: torch.dtype
) -> int:
    """Calculate memory requirements for processing."""
    kernel_size = 2 * radius + 1
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    # Calculate memory for main tensors (BHWC format)
    input_memory = height * width * channels * bytes_per_element
    unfold_memory = height * width * kernel_size * kernel_size * channels * bytes_per_element
    weights_memory = kernel_size * kernel_size * bytes_per_element
    
    # Add buffer for intermediate calculations (2x for safety)
    total_memory = (input_memory + unfold_memory + weights_memory) * 2
    
    return total_memory

def get_optimal_tile_size(
    height: int,
    width: int,
    channels: int,
    radius: int,
    dtype: torch.dtype
) -> int:
    """Calculate optimal tile size based on available memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(total_memory * 0.7)  # Use 70% of available memory
    else:
        available_memory = 8 * (1024 ** 3)  # Assume 8GB for CPU
    
    # Target memory per tile
    target_memory = available_memory // 4  # Use 25% of available memory per tile
    
    # Calculate base tile size
    kernel_size = 2 * radius + 1
    bytes_per_pixel = channels * {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    # Account for unfolding operation memory
    memory_factor = kernel_size * kernel_size * 2  # 2x for safety
    pixels_per_tile = target_memory / (bytes_per_pixel * memory_factor)
    tile_size = int(math.sqrt(pixels_per_tile))
    
    # Ensure tile size is divisible by 8 for GPU efficiency
    tile_size = max(256, (tile_size // 8) * 8)
    
    return tile_size

#------------------------------------------------------------------------------
# Section 3: Tensor Validation and Debug
#------------------------------------------------------------------------------
def verify_tensor_format(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Verify and normalize tensor to BHWC format."""
    if len(tensor.shape) == 3:  # HWC
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        logger.debug("Converted HWC to BHWC format")
        
    if len(tensor.shape) != 4:
        raise ValueError(f"Invalid tensor shape: {tensor.shape}")
        
    b, h, w, c = tensor.shape
    return tensor, {
        "batch_size": b,
        "height": h,
        "width": w,
        "channels": c
    }

#------------------------------------------------------------------------------
# Section 4: Processing Implementation
#------------------------------------------------------------------------------
def process_tile(
    tile: torch.Tensor,
    radius: int,
    threshold: float,
    device: torch.device
) -> torch.Tensor:
    """Process a single tile with surface blur."""
    # Ensure BHWC format
    tile, dims = verify_tensor_format(tile)
    tile = tile.to(device)
    
    try:
        kernel_size = 2 * radius + 1
        
        # Create position weights
        y_coords = torch.arange(-radius, radius + 1, device=device)
        x_coords = torch.arange(-radius, radius + 1, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        position_weights = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * radius * radius))
        
        # Process each channel separately
        channels = []
        for c in range(dims['channels']):
            channel = tile[..., c:c+1]
            
            # Convert to BCHW for unfold operation
            x = channel.permute(0, 3, 1, 2)
            
            # Extract patches
            patches = F.unfold(
                x,
                kernel_size=(kernel_size, kernel_size),
                padding=radius
            )
            
            # Reshape patches
            patches = patches.view(1, -1, kernel_size * kernel_size, channel.shape[1] * channel.shape[2])
            
            # Calculate color differences
            center = patches[:, :, kernel_size * kernel_size // 2:kernel_size * kernel_size // 2 + 1]
            color_weights = torch.exp(-torch.abs(patches - center) / threshold)
            
            # Apply weights
            weights = color_weights * position_weights.view(1, 1, -1, 1)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
            
            # Calculate weighted sum
            result = (patches * weights).sum(dim=2)
            
            # Reshape back to image format
            result = result.view(1, channel.shape[1], channel.shape[2], 1)
            channels.append(result)
            
            # Clear GPU cache after each channel
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Combine channels
        result = torch.cat(channels, dim=-1)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in tile processing: {str(e)}")
        return tile

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
#------------------------------------------------------------------------------
class HTSurfaceBlurNode:
    """Surface blur with tiled processing and memory optimization."""
    
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
                    "step": 1
                }),
                "threshold": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 0.1
                })
            }
        }

    def apply_surface_blur(
        self,
        image: torch.Tensor,
        radius: int,
        threshold: float
    ) -> Tuple[torch.Tensor]:
        """Apply surface blur with tiled processing."""
        logger.info(f"HTSurfaceBlurNode v{VERSION} - Processing")
        
        try:
            # Ensure BHWC format
            image, dims = verify_tensor_format(image)
            
            # Calculate optimal tile size
            tile_size = get_optimal_tile_size(
                dims['height'],
                dims['width'],
                dims['channels'],
                radius,
                image.dtype
            )
            logger.info(f"Using tile size: {tile_size}x{tile_size}")
            
            # Normalize threshold to 0-1 range
            threshold = threshold / 255.0
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Processing device: {device}")
            
            # Initialize output tensor
            result = torch.zeros_like(image)
            
            # Process tiles
            total_tiles = ((dims['height'] + tile_size - 1) // tile_size) * ((dims['width'] + tile_size - 1) // tile_size)
            current_tile = 0
            
            for y in range(0, dims['height'], tile_size):
                for x in range(0, dims['width'], tile_size):
                    current_tile += 1
                    if current_tile % 5 == 0 or current_tile == total_tiles:
                        logger.info(f"Processing tile {current_tile}/{total_tiles}")
                    
                    # Calculate tile bounds
                    y_end = min(y + tile_size + radius, dims['height'])
                    x_end = min(x + tile_size + radius, dims['width'])
                    y_start = max(0, y - radius)
                    x_start = max(0, x - radius)
                    
                    # Extract and process tile
                    tile = image[:, y_start:y_end, x_start:x_end, :]
                    processed = process_tile(tile, radius, threshold, device)
                    
                    # Calculate output region
                    out_y_start = y
                    out_x_start = x
                    out_y_end = min(y + tile_size, dims['height'])
                    out_x_end = min(x + tile_size, dims['width'])
                    
                    # Store result
                    result[:, out_y_start:out_y_end, out_x_start:out_x_end, :] = \
                        processed[:, radius:radius+out_y_end-out_y_start, 
                                radius:radius+out_x_end-out_x_start, :]
                    
                    # Clear GPU cache after each tile
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            logger.info("Surface blur processing complete")
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in surface blur: {str(e)}")
            return (image,)