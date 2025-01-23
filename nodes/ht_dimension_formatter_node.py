"""
File: homage_tools/nodes/ht_dimension_formatter_node.py
Version: 1.2.1
Description: Node for formatting image dimensions with BHWC tensor handling
"""

from typing import Dict, Any, Tuple, Optional
import torch
import logging

logger = logging.getLogger('HommageTools')

class HTDimensionFormatterNode:
    """Formats image dimensions into standardized strings."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "format_dimensions"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_dimensions",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8
                }),
                "spacing": ("STRING", {
                    "default": " "
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    def _extract_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """Extract dimensions from BHWC format tensor."""
        try:
            if len(image.shape) == 3:  # HWC format
                height, width = image.shape[0:2]
            elif len(image.shape) == 4:  # BHWC format
                height, width = image.shape[1:3]
            else:
                raise ValueError(f"Invalid tensor dimensions: {image.shape}")
            
            print(f"Extracted dimensions: {width}x{height}")
            return width, height
            
        except Exception as e:
            logger.error(f"Dimension extraction error: {str(e)}")
            raise

    def format_dimensions(
        self,
        width: int,
        height: int,
        spacing: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """Format dimensions with BHWC handling."""
        try:
            if image is not None:
                try:
                    width, height = self._extract_dimensions(image)
                except Exception as e:
                    print(f"Using widget dimensions due to error: {str(e)}")

            result = f"{width}{spacing}x{spacing}{height}"
            return (result,)
            
        except Exception as e:
            logger.error(f"Formatting error: {str(e)}")
            return ("Error formatting dimensions",)