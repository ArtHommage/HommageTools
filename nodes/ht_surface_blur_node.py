"""
File: homage_tools/nodes/ht_surface_blur_node.py
Description: Memory-efficient surface blur with tiled processing and CUDA optimization
Version: 1.6.0
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
import logging

logger = logging.getLogger('HommageTools')
VERSION = "1.6.0"

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------

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
    dtype: torch.dtype,
    memory_usage: str = "conservative"
) -> int:
    """Calculate optimal tile size based on available memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Adjust memory usage based on selected mode
        if memory_usage == "super_aggressive":
            available_memory = int(total_memory * 0.95)  # Use 95% of available memory
        elif memory_usage == "aggressive":
            available_memory = int(total_memory * 0.9)   # Use 90% of available memory
        else:  # conservative (default)
            available_memory = int(total_memory * 0.7)   # Use 70% of available memory
    else:
        available_memory = 8 * (1024 ** 3)  # Assume 8GB for CPU
    
    # Target memory per tile
    if memory_usage == "super_aggressive":
        target_memory = available_memory * 0.7  # Use 70% of available memory per tile
    elif memory_usage == "aggressive":
        target_memory = available_memory // 2   # Use 50% of available memory per tile
    else:  # conservative
        target_memory = available_memory // 4   # Use 25% of available memory per tile
    
    # Calculate base tile size
    kernel_size = 2 * radius + 1
    bytes_per_pixel = channels * {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    # Account for unfolding operation memory
    if memory_usage == "super_aggressive":
        memory_factor = kernel_size * kernel_size * 1.1  # 1.1x for super aggressive
    elif memory_usage == "aggressive":
        memory_factor = kernel_size * kernel_size * 1.5  # 1.5x for aggressive
    else:  # conservative
        memory_factor = kernel_size * kernel_size * 2    # 2x for conservative
        
    pixels_per_tile = target_memory / (bytes_per_pixel * memory_factor)
    tile_size = int(math.sqrt(pixels_per_tile))
    
    # Ensure tile size is divisible by 8 for GPU efficiency
    tile_size = max(512, (tile_size // 8) * 8)
    
    return tile_size

#------------------------------------------------------------------------------
# Section 3: Tensor Validation and Debug
#------------------------------------------------------------------------------
def debug_tensor_stats(tensor: torch.Tensor, name: str) -> None:
    """Print debug statistics for tensor values and format."""
    shape = tensor.shape
    print(f"\nDebug {name}:")
    print(f"Shape: {shape}")
    
    if len(shape) == 3:  # HWC
        print(f"Format: HWC")
        print(f"Dimensions: {shape[0]}x{shape[1]}")
        print(f"Channels: {shape[2]}")
    elif len(shape) == 4:  # BHWC
        print(f"Format: BHWC")
        print(f"Dimensions: {shape[1]}x{shape[2]}")
        print(f"Batch: {shape[0]}, Channels: {shape[3]}")
        
    print(f"Value range: min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")
    if torch.isnan(tensor).any():
        print("WARNING: Contains NaN values")
    if torch.isinf(tensor).any():
        print("WARNING: Contains Inf values")

def verify_tensor_format(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Verify and normalize tensor to BHWC format."""
    if len(tensor.shape) == 3:  # HWC
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        # print("Converted HWC to BHWC format")
        
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
    debug_tensor_stats(tile, "Input Tile")
    
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
            # print(f"Processing channel {c}: shape={channel.shape}")
            
            # Convert to BCHW for unfold operation
            x = channel.permute(0, 3, 1, 2)
            # print(f"Converted to BCHW: shape={x.shape}")
            
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
        debug_tensor_stats(result, "Final Tile Output")
        
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
                }),
                "memory_usage": (["conservative", "aggressive", "super_aggressive"], {
                    "default": "conservative",
                    "description": "Memory usage strategy"
                })
            }
        }

    def apply_surface_blur(
        self,
        image: torch.Tensor,
        radius: int,
        threshold: float,
        memory_usage: str = "conservative"
    ) -> Tuple[torch.Tensor]:
        """Apply surface blur with tiled processing."""
        print(f"\nHTSurfaceBlurNode v{VERSION} - Processing")
        debug_tensor_stats(image, "Input Image")
        
        try:
            # Ensure BHWC format
            image, dims = verify_tensor_format(image)
            
            # Calculate optimal tile size
            tile_size = get_optimal_tile_size(
                dims['height'],
                dims['width'],
                dims['channels'],
                radius,
                image.dtype,
                memory_usage
            )
            print(f"Using tile size: {tile_size}x{tile_size} (Memory usage: {memory_usage})")
            
            # Normalize threshold to 0-1 range
            threshold = threshold / 255.0
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Processing device: {device}")
            
            # Initialize output tensor
            result = torch.zeros_like(image)
            
            # Process tiles
            total_tiles = ((dims['height'] + tile_size - 1) // tile_size) * ((dims['width'] + tile_size - 1) // tile_size)
            current_tile = 0
            
            for y in range(0, dims['height'], tile_size):
                for x in range(0, dims['width'], tile_size):
                    current_tile += 1
                    print(f"\nProcessing tile {current_tile}/{total_tiles}")
                    
                    # Calculate actual tile area (without padding)
                    out_y_start = y
                    out_x_start = x
                    out_y_end = min(y + tile_size, dims['height'])
                    out_x_end = min(x + tile_size, dims['width'])
                    
                    # Calculate padding for this tile
                    pad_top = min(radius, y)
                    pad_left = min(radius, x)
                    pad_bottom = min(radius, dims['height'] - out_y_end)
                    pad_right = min(radius, dims['width'] - out_x_end)
                    
                    # Calculate extraction bounds with actual available padding
                    y_start = out_y_start - pad_top
                    x_start = out_x_start - pad_left
                    y_end = out_y_end + pad_bottom
                    x_end = out_x_end + pad_right
                    
                    print(f"Tile area: ({out_x_start}, {out_y_start}) to ({out_x_end}, {out_y_end})")
                    print(f"Padded extraction: ({x_start}, {y_start}) to ({x_end}, {y_end})")
                    print(f"Applied padding: top={pad_top}, left={pad_left}, bottom={pad_bottom}, right={pad_right}")
                    
                    # Extract tile with available padding
                    tile = image[:, y_start:y_end, x_start:x_end, :]
                    processed = process_tile(tile, radius, threshold, device)
                    
                    # Calculate correct slicing from processed tile
                    # The processed area we need is offset by the applied padding
                    result[:, out_y_start:out_y_end, out_x_start:out_x_end, :] = \
                        processed[:, pad_top:pad_top+(out_y_end-out_y_start), 
                                 pad_left:pad_left+(out_x_end-out_x_start), :]
                    
                    # Clear GPU cache after each tile
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Final memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in surface blur: {str(e)}")
            print(f"Error details: {str(e)}")
            
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return (image,)