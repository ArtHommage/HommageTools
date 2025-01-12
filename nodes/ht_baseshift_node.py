"""
File: homage_tools/nodes/ht_baseshift_node.py
Version: 1.0.1
Description: Node for calculating base shift values for image processing

Sections:
1. Imports and Type Definitions
2. Node Class Definition and Configuration
3. Calculation Methods
4. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional
import torch

#------------------------------------------------------------------------------
# Section 2: Node Class Definition and Configuration
#------------------------------------------------------------------------------
class HTBaseShiftNode:
    """
    Calculates base shift values for images.
    Provides configurable shift parameters for image processing adjustments.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_shift"
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("max_shift", "base_shift")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define input parameters and their configurations.
        Includes image dimensions and shift parameters with specific ranges and steps.
        """
        return {
            "required": {
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.50,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    #--------------------------------------------------------------------------
    # Section 3: Calculation Methods
    #--------------------------------------------------------------------------
    def _calculate_max_shift(
        self,
        width: int,
        height: int,
        base_shift: float,
        max_shift: float
    ) -> float:
        """
        Calculate max shift value based on image dimensions and shift parameters.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            base_shift: Base shift value
            max_shift: Maximum shift value
            
        Returns:
            float: Calculated shift value
        """
        try:
            calculated_shift = (
                (base_shift - max_shift) / 
                (256 - ((width * height) / 256)) * 
                3840 + base_shift
            )
            return float(calculated_shift)
        except ZeroDivisionError:
            return float(base_shift)

    #--------------------------------------------------------------------------
    # Section 4: Main Processing Logic
    #--------------------------------------------------------------------------
    def calculate_shift(
        self,
        image_width: int,
        image_height: int,
        max_shift: float,
        base_shift: float,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[float, float]:
        """Calculate shift values for image processing."""

        print(f"Input image_width: {image_width}")
        print(f"Input image_height: {image_height}")

        if image is not None:
            print(f"Image tensor shape: {image.shape}")
            if len(image.shape) == 3:  # (C, H, W)
                _, image_height, image_width = image.shape
            elif len(image.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
                if image.shape[1] < image.shape[2]: #check if channels are second or third dim
                    _, channels, image_height, image_width = image.shape
                else:
                    _, image_height, image_width, channels = image.shape
            else:
                raise ValueError(
                    f"Image tensor has unsupported shape: {image.shape}. Expected 3 or 4 dimensions (C, H, W) or (B, H, W, C) or (B, C, H, W)"
                )

            print(f"Extracted image_width from tensor: {image_width}")
            print(f"Extracted image_height from tensor: {image_height}")

        calculated_max_shift = self._calculate_max_shift(
            image_width, image_height, base_shift, max_shift
        )

        return (calculated_max_shift, base_shift)