"""
File: homage_tools/nodes/ht_training_size_node.py

HommageTools Training Size Calculator Node
Version: 1.0.0
Description: A specialized node for calculating optimal image dimensions for training,
ensuring dimensions align with common training resolutions (512, 768, 1024) for the
long edge while maintaining aspect ratios and 64-pixel divisibility constraints.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Dimension Calculation Methods
4. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
from typing import Dict, Any, Tuple, Optional, Literal

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTTrainingSizeNode:
    """
    A ComfyUI node that calculates optimal dimensions for training images.
    
    Features:
    - Forces long edge to standard training resolutions (512, 768, 1024)
    - Ensures dimensions are divisible by 64
    - Maintains aspect ratios
    - Provides scaling options (up/down/both)
    - Detects if clipping will be required
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_dimensions"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "scale_factor", "requires_clipping")
    
    # Standard training dimensions
    STANDARD_DIMS = [512, 768, 1024]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define input types and their default values."""
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "description": "Current image width"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "description": "Current image height"
                }),
                "scaling_mode": (["both", "upscale_only", "downscale_only"], {
                    "default": "both",
                    "description": "Restrict scaling direction"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

#------------------------------------------------------------------------------
# Section 3: Dimension Calculation Methods
#------------------------------------------------------------------------------
    def _round_to_64(self, dimension: float) -> int:
        """Round dimension to nearest multiple of 64."""
        return round(dimension / 64) * 64

    def _calculate_quality_loss(
        self,
        current_size: int,
        target_size: int,
        current_area: int,
        target_area: int
    ) -> float:
        """
        Calculate quality loss for a potential resize.
        
        Args:
            current_size: Current long edge size
            target_size: Potential target size
            current_area: Current image area
            target_area: Potential target area
            
        Returns:
            float: Quality loss score (lower is better)
        """
        # Calculate dimension change (weighted 1.5x)
        dimension_change = abs(1 - (target_size / current_size)) * 1.5
        
        # Calculate area change (weighted 0.5x)
        area_change = abs(1 - (target_area / current_area)) * 0.5
        
        return dimension_change + area_change

    def _find_optimal_dimensions(
        self,
        current_width: int,
        current_height: int,
        scaling_mode: str
    ) -> Tuple[int, int, float, int]:
        """
        Find optimal dimensions ensuring long edge matches a standard dimension.
        
        Args:
            current_width: Original width
            current_height: Original height
            scaling_mode: Scaling direction restriction
            
        Returns:
            Tuple[int, int, float, int]: (width, height, scale_factor, requires_clipping)
        """
        # Get longer edge and aspect ratio
        current_long_edge = max(current_width, current_height)
        is_width_longer = current_width >= current_height
        aspect_ratio = current_width / current_height
        current_area = current_width * current_height
        
        # Find best standard dimension for long edge based on quality loss
        best_long_edge = None
        min_quality_loss = float('inf')
        best_scale = 1.0
        
        for target in self.STANDARD_DIMS:
            # Skip if scaling mode restricts this direction
            if scaling_mode == "upscale_only" and target < current_long_edge:
                continue
            if scaling_mode == "downscale_only" and target > current_long_edge:
                continue
            
            # Calculate potential dimensions and area
            scale = target / current_long_edge
            if is_width_longer:
                test_height = self._round_to_64(target / aspect_ratio)
                test_area = target * test_height
            else:
                test_width = self._round_to_64(target * aspect_ratio)
                test_area = test_width * target
            
            # Calculate quality loss
            quality_loss = self._calculate_quality_loss(
                current_long_edge, target, current_area, test_area
            )
            
            if quality_loss < min_quality_loss:
                min_quality_loss = quality_loss
                best_long_edge = target
                best_scale = scale
        
        # If no valid target was found (due to scaling restrictions), use closest allowed
        if best_long_edge is None:
            best_long_edge = self.STANDARD_DIMS[0]
            best_scale = best_long_edge / current_long_edge
        
        # Calculate final dimensions
        if is_width_longer:
            new_width = best_long_edge
            new_height = self._round_to_64(best_long_edge / aspect_ratio)
        else:
            new_height = best_long_edge
            new_width = self._round_to_64(best_long_edge * aspect_ratio)
            
        # Check if clipping will be required (using integer dimensions)
        scaled_width = int(current_width * best_scale)
        scaled_height = int(current_height * best_scale)
        requires_clipping = 1 if (scaled_width > new_width or scaled_height > new_height) else 0
        
        return new_width, new_height, best_scale, requires_clipping

#------------------------------------------------------------------------------
# Section 4: Main Processing Logic
#------------------------------------------------------------------------------
    def calculate_dimensions(
        self,
        width: int,
        height: int,
        scaling_mode: Literal["both", "upscale_only", "downscale_only"],
        image: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, float, int]:
        """
        Calculate optimal dimensions for training.
        
        Args:
            width: Widget input width
            height: Widget input height
            scaling_mode: Scaling direction restriction
            image: Optional input image
            
        Returns:
            Tuple[int, int, float, int]: (width, height, scale_factor, requires_clipping)
        """
        try:
            # Use image dimensions if provided
            if image is not None:
                height, width = image.shape[-2:]
            
            # Find optimal dimensions
            return self._find_optimal_dimensions(width, height, scaling_mode)
            
        except Exception as e:
            print(f"Error in HTTrainingSizeNode: {str(e)}")
            return (width, height, 1.0, 0)  # Return original dimensions on error