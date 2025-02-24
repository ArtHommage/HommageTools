"""
File: homage_tools/nodes/ht_oidn_node.py
Version: 2.1.1
Description: Enhanced OIDN denoising node with proper BHWC tensor handling and value range fixes
"""

import torch
import torch.nn.functional as F
import numpy as np
import oidn
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import math
import gc
import logging

logger = logging.getLogger('HommageTools')

@dataclass
class TileConfig:
    """Configuration for tiling and processing."""
    size: int
    overlap: int
    batch_size: int
    device: torch.device

def debug_tensor_stats(tensor: torch.Tensor, name: str):
    """Print debug statistics for tensor values."""
    print(f"\nDebug {name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min().item():.6f}")
    print(f"Max: {tensor.max().item():.6f}")
    print(f"Mean: {tensor.mean().item():.6f}")
    if torch.isnan(tensor).any():
        print("WARNING: Contains NaN values")
    if torch.isinf(tensor).any():
        print("WARNING: Contains Inf values")

def process_tile(
    tile: torch.Tensor,
    strength: float,
    oidn_filter
) -> torch.Tensor:
    """
    Process a single tile with OIDN.
    
    Args:
        tile: Input tile (BHWC format, values in [0, 1])
        strength: Denoising strength
        oidn_filter: OIDN filter object
        
    Returns:
        torch.Tensor: Processed tile (BHWC format, values in [0, 1])
    """
    # Debug input tile
    debug_tensor_stats(tile, "Input Tile")
    
    # Convert from BHWC to NHWC (required by OIDN)
    if len(tile.shape) == 3:  # HWC
        tile = tile.unsqueeze(0)  # Add batch dimension -> BHWC
        
    # Ensure values are in [0, 1] range
    tile = torch.clamp(tile, 0, 1)
    
    # Move to CPU and convert to numpy
    tile_cpu = tile.cpu().numpy()
    result = np.zeros_like(tile_cpu)
    
    # Process with OIDN (expects HWC format for each image)
    for b in range(tile_cpu.shape[0]):
        # Get single image in HWC format
        img = tile_cpu[b]
        
        # Ensure correct shape and type
        if img.shape[-1] > 3:  # Handle alpha channel
            img = img[..., :3]
            
        debug_tensor_stats(torch.from_numpy(img), "OIDN Input")
            
        oidn.SetSharedFilterImage(
            oidn_filter, "color",
            img.copy(), oidn.FORMAT_FLOAT3,  # Use copy to ensure memory is contiguous
            img.shape[1], img.shape[0]  # width, height
        )
        oidn.SetSharedFilterImage(
            oidn_filter, "output",
            result[b], oidn.FORMAT_FLOAT3,
            img.shape[1], img.shape[0]  # width, height
        )
        oidn.ExecuteFilter(oidn_filter)
        
        debug_tensor_stats(torch.from_numpy(result[b]), "OIDN Output")
    
    # Ensure result is in valid range
    result = np.clip(result, 0, 1)
    
    # Apply strength adjustment
    if strength != 1.0:
        if strength > 1.0:
            result = result + (result - tile_cpu) * (strength - 1.0)
        else:
            result = result * strength + tile_cpu * (1.0 - strength)
    
    # Convert back to tensor and ensure valid range
    result_tensor = torch.from_numpy(result).to(tile.device)
    result_tensor = torch.clamp(result_tensor, 0, 1)
    
    debug_tensor_stats(result_tensor, "Final Tile Output")
    
    return result_tensor

def process_image_tiled(
    image: torch.Tensor,
    tile_size: int,
    overlap: int,
    strength: float,
    oidn_filter
) -> torch.Tensor:
    """
    Process image using tiled approach for memory efficiency.
    
    Args:
        image: Input image (BHWC format, values in [0, 1])
        tile_size: Size of processing tiles
        overlap: Overlap between tiles
        strength: Denoising strength
        oidn_filter: OIDN filter object
        
    Returns:
        torch.Tensor: Processed image (BHWC format, values in [0, 1])
    """
    # Debug input image
    debug_tensor_stats(image, "Input Image")
    
    # Ensure BHWC format
    if len(image.shape) == 3:  # HWC
        image = image.unsqueeze(0)  # Add batch dimension -> BHWC
        
    B, H, W, C = image.shape
    device = image.device
    
    print(f"\nProcessing image of shape {image.shape} (BHWC)")
    print(f"Using tile size: {tile_size}x{tile_size} with {overlap}px overlap")
    
    # Initialize output tensor
    result = torch.zeros_like(image)
    weights = torch.zeros((B, H, W, 1), device=device)
    
    # Calculate tile positions
    y_positions = range(0, H - overlap, tile_size - overlap)
    x_positions = range(0, W - overlap, tile_size - overlap)
    
    # Create blending mask
    blend_mask = torch.ones((tile_size, tile_size), device=device)
    blend_mask[:overlap] *= torch.linspace(0, 1, overlap, device=device).view(-1, 1)
    blend_mask[-overlap:] *= torch.linspace(1, 0, overlap, device=device).view(-1, 1)
    blend_mask[:, :overlap] *= torch.linspace(0, 1, overlap, device=device)
    blend_mask[:, -overlap:] *= torch.linspace(1, 0, overlap, device=device)
    
    # Process tiles
    total_tiles = len(y_positions) * len(x_positions)
    current_tile = 0
    
    for y in y_positions:
        for x in x_positions:
            current_tile += 1
            print(f"\nProcessing tile {current_tile}/{total_tiles}")
            
            # Calculate tile bounds
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)
            
            print(f"Tile coordinates: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Extract and process tile
            tile = image[:, y1:y2, x1:x2]
            processed = process_tile(tile, strength, oidn_filter)
            
            # Apply blending mask
            mask = blend_mask[:y2-y1, :x2-x1].view(1, y2-y1, x2-x1, 1)
            
            # Accumulate result
            result[:, y1:y2, x1:x2] += processed * mask
            weights[:, y1:y2, x1:x2] += mask
    
    # Normalize by weights
    result = result / (weights + 1e-8)
    
    # Ensure final result is in valid range
    result = torch.clamp(result, 0, 1)
    
    debug_tensor_stats(result, "Final Output Image")
    
    return result

class HTOIDNNode:
    """
    Enhanced OIDN node with proper BHWC tensor handling and tiling support.
    """
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "denoise_image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64
                })
            }
        }
    
    def __init__(self):
        self.device = None
        self.filter = None
        
    def initialize_oidn(self):
        """Initialize OIDN device and filter."""
        if self.device is None:
            print("\nInitializing OIDN...")
            self.device = oidn.NewDevice()
            oidn.CommitDevice(self.device)
            self.filter = oidn.NewFilter(self.device, "RT")
            oidn.CommitFilter(self.filter)
            print("OIDN initialization complete")
    
    def denoise_image(
        self,
        image: torch.Tensor,
        strength: float,
        tile_size: int
    ) -> Tuple[torch.Tensor]:
        """
        Denoise image using OIDN with tiling.
        
        Args:
            image: Input image (BHWC format, values in [0, 1])
            strength: Denoising strength
            tile_size: Processing tile size
            
        Returns:
            Tuple[torch.Tensor]: Processed image (BHWC format, values in [0, 1])
        """
        try:
            # Initialize OIDN
            self.initialize_oidn()
            
            # Process with tiling
            overlap = tile_size // 4  # 25% overlap
            result = process_image_tiled(
                image,
                tile_size,
                overlap,
                strength,
                self.filter
            )
            
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in OIDN processing: {str(e)}")
            print(f"\nError details: {str(e)}")
            return (image,)