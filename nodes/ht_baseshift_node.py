"""
File: homage_tools/nodes/ht_baseshift_node.py
Version: 1.0.2
Description: Node for calculating base shift values with BHWC tensor handling
"""

from typing import Dict, Any, Tuple, Optional
import torch

class HTBaseShiftNode:
    """Calculates base shift values for images."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_shift"
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("max_shift", "base_shift")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
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

    def calculate_shift(
        self,
        image_width: int,
        image_height: int,
        max_shift: float,
        base_shift: float,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[float, float]:
        """Calculate shift values using BHWC format."""
        print(f"Input dimensions: {image_width}x{image_height}")

        if image is not None:
            print(f"Image tensor shape: {image.shape}")
            if len(image.shape) == 3:  # HWC
                image_height, image_width = image.shape[0:2]
            elif len(image.shape) == 4:  # BHWC
                image_height, image_width = image.shape[1:3]
            else:
                raise ValueError(f"Unexpected tensor shape: {image.shape}")

            print(f"Extracted dimensions: {image_width}x{image_height}")

        try:
            calculated_shift = (
                (base_shift - max_shift) / 
                (256 - ((image_width * image_height) / 256)) * 
                3840 + base_shift
            )
            return (calculated_shift, base_shift)
        except ZeroDivisionError:
            return (base_shift, base_shift)