"""
File: homage_tools/nodes/ht_surface_blur_node.py
Version: 1.2.1
Description: Surface blur with fixed imports
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
import logging
import time

logger = logging.getLogger('HommageTools')

def calculate_color_distance(center: torch.Tensor, neighbors: torch.Tensor, threshold: float) -> torch.Tensor:
    diff = torch.abs(neighbors - center)
    diff = torch.mean(diff, dim=-1, keepdim=True)
    return torch.exp(-diff / threshold)

def normalize_kernel(kernel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return kernel / (torch.sum(kernel, dim=(1,2), keepdim=True) + eps)

def process_surface_blur(image: torch.Tensor, radius: int, threshold: float, device: torch.device) -> torch.Tensor:
    batch_size, height, width, channels = image.shape
    kernel_size = 2 * radius + 1
    
    y_coords = torch.arange(-radius, radius + 1, device=device)
    x_coords = torch.arange(-radius, radius + 1, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    position_weights = torch.exp(-(x_grid.pow(2) + y_grid.pow(2)) / (2 * radius * radius))
    
    tile_size = min(32, height, width)
    result = torch.zeros_like(image)
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            pad_y1, pad_y2 = radius, radius
            pad_x1, pad_x2 = radius, radius
            
            if y == 0: pad_y1 = 0
            if y_end == height: pad_y2 = 0
            if x == 0: pad_x1 = 0
            if x_end == width: pad_x2 = 0
            
            tile = image[:, max(0, y - radius):min(height, y_end + radius),
                           max(0, x - radius):min(width, x_end + radius), :]
            
            tile_bchw = tile.permute(0, 3, 1, 2)
            unfold = F.unfold(tile_bchw, kernel_size=(kernel_size, kernel_size), 
                            padding=(pad_y1, pad_x1))
            
            patches = unfold.view(batch_size, channels, kernel_size * kernel_size, -1)
            patches = patches.permute(0, 3, 2, 1)
            
            center_idx = kernel_size * kernel_size // 2
            center = patches[:, :, center_idx:center_idx+1, :]
            
            weights = calculate_color_distance(center, patches, threshold)
            weights = weights * position_weights.view(1, 1, -1, 1)
            weights = normalize_kernel(weights)
            
            weighted_sum = torch.sum(patches * weights, dim=2)
            weighted_sum = weighted_sum.view(batch_size, y_end - y, x_end - x, channels)
            
            result[:, y:y_end, x:x_end, :] = weighted_sum
            
    return result

class HTSurfaceBlurNode:
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
                "device": (["cpu", "gpu"], {
                    "default": "gpu" if torch.cuda.is_available() else "cpu"
                })
            }
        }

    def apply_surface_blur(self, image: torch.Tensor, radius: int, threshold: float, device: str) -> Tuple[torch.Tensor]:
        try:
            start_time = time.time()
            
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            proc_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
            result = process_surface_blur(image.to(proc_device), radius, threshold / 255.0, proc_device)
            
            if proc_device.type == "cuda":
                result = result.cpu()
            
            print(f"Processing completed in {time.time() - start_time:.2f}s")
            return (result,)
            
        except Exception as e:
            logger.error(f"Surface blur failed: {str(e)}")
            return (image,)