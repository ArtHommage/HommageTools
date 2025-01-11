"""
File: homage_tools/nodes/ht_dimension_formatter_node.py

HommageTools Dimension Formatter Node
Version: 1.0.0
Description: A node that takes image dimensions (either from an image input or widget values)
and formats them into a standardized dimension string with configurable spacing.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. String Formatting Methods
4. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional
import torch

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTDimensionFormatterNode:
    """
    A ComfyUI node that formats image dimensions into a standardized string.
    
    Features:
    - Accepts either direct dimension inputs or an image
    - Image input overrides widget values when provided
    - Configurable spacing in output format
    - Consistent dimension string output
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "format_dimensions"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_dimensions",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "description": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "description": "Image height in pixels"
                }),
                "spacing": ("STRING", {
                    "default": " ",
                    "multiline": False,
                    "description": "Spacing around the 'x' (empty for none)"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

#------------------------------------------------------------------------------
# Section 3: String Formatting Methods
#------------------------------------------------------------------------------
    def _format_dimension_string(
        self,
        width: int,
        height: int,
        spacing: str
    ) -> str:
        """
        Format dimensions into a string with specified spacing.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            spacing: Spacing to use around the 'x' character
            
        Returns:
            str: Formatted dimension string
        """
        return f"{width}{spacing}x{spacing}{height}"

#------------------------------------------------------------------------------
# Section 4: Main Processing Logic
#------------------------------------------------------------------------------
    def format_dimensions(
        self,
        width: int,
        height: int,
        spacing: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """
        Format image dimensions into a string with configurable spacing.
        
        Args:
            width: Widget input width
            height: Widget input height
            spacing: Spacing to use around the 'x'
            image: Optional input image tensor
            
        Returns:
            Tuple[str]: Single-element tuple containing formatted dimension string
        """
        try:
            # Use image dimensions if provided
            if image is not None:
                height, width = image.shape[-2:]
            
            # Format the dimension string
            result = self._format_dimension_string(width, height, spacing)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in HTDimensionFormatterNode: {str(e)}")
            return (f"{width}x{height}",)  # Return basic format on error