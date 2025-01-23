"""
File: homage_tools/nodes/ht_training_size_node.py
Version: 1.2.2
Description: Node for calculating optimal training dimensions with BHWC tensor handling
"""

import torch
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger('HommageTools')

def validate_dimensions(width: int, height: int) -> Tuple[bool, str]:
    """Validate input dimensions."""
    if width <= 0 or height <= 0:
        return False, f"Invalid dimensions: {width}x{height}"
    if width > 16384 or height > 16384:
        return False, f"Dimensions too large: {width}x{height}"
    return True, ""

def round_to_multiple(value: float, multiple: int) -> int:
    """Round value to nearest multiple."""
    return int(round(value / multiple) * multiple)

class HTTrainingSizeNode:
    """Calculates optimal dimensions for training images."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_dimensions"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "scale_factor", "requires_clipping")
    
    STANDARD_DIMS: List[int] = [512, 768, 1024]
    MIN_DIM: int = 64
    MAX_DIM: int = 8192
    ROUND_TO: int = 64
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": cls.MIN_DIM,
                    "max": cls.MAX_DIM,
                    "step": cls.ROUND_TO
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": cls.MIN_DIM,
                    "max": cls.MAX_DIM,
                    "step": cls.ROUND_TO
                }),
                "scaling_mode": (["both", "upscale_only", "downscale_only"], {
                    "default": "both"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    def _find_optimal_dimensions(
        self,
        current_width: int,
        current_height: int,
        scaling_mode: str
    ) -> Tuple[int, int, float, int]:
        """Find optimal training dimensions."""
        logger.debug(f"Finding optimal dimensions for {current_width}x{current_height}")
        
        current_long_edge = max(current_width, current_height)
        is_width_longer = current_width >= current_height
        aspect_ratio = current_width / current_height
        
        best_long_edge = None
        min_quality_loss = float('inf')
        best_scale = 1.0
        
        for target in self.STANDARD_DIMS:
            if scaling_mode == "upscale_only" and target < current_long_edge:
                continue
            if scaling_mode == "downscale_only" and target > current_long_edge:
                continue
                
            scale = target / current_long_edge
            quality_loss = abs(1 - scale)
            
            if quality_loss < min_quality_loss:
                min_quality_loss = quality_loss
                best_long_edge = target
                best_scale = scale
                
        if best_long_edge is None:
            best_long_edge = self.STANDARD_DIMS[0]
            best_scale = best_long_edge / current_long_edge
        
        if is_width_longer:
            new_width = best_long_edge
            new_height = round_to_multiple(best_long_edge / aspect_ratio, self.ROUND_TO)
        else:
            new_height = best_long_edge
            new_width = round_to_multiple(best_long_edge * aspect_ratio, self.ROUND_TO)
        
        scaled_width = int(current_width * best_scale)
        scaled_height = int(current_height * best_scale)
        requires_clipping = 1 if (scaled_width > new_width or scaled_height > new_height) else 0
        
        return new_width, new_height, best_scale, requires_clipping

    def calculate_dimensions(
        self,
        width: int,
        height: int,
        scaling_mode: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, float, int]:
        """Calculate optimal training dimensions with BHWC handling."""
        try:
            if image is not None:
                print(f"Processing tensor shape: {image.shape}")
                if len(image.shape) == 3:  # HWC
                    height, width = image.shape[0:2]
                elif len(image.shape) == 4:  # BHWC
                    height, width = image.shape[1:3]
                else:
                    raise ValueError(f"Invalid tensor shape: {image.shape}")
                print(f"Extracted dimensions: {width}x{height}")
            
            # Validate dimensions
            is_valid, error_msg = validate_dimensions(width, height)
            if not is_valid:
                logger.error(error_msg)
                return (0, 0, 0.0, 0)
            
            return self._find_optimal_dimensions(width, height, scaling_mode)
            
        except Exception as e:
            logger.error(f"Error calculating dimensions: {str(e)}")
            return (0, 0, 0.0, 0)