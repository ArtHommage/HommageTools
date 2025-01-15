"""
File: homage_tools/nodes/ht_dimension_formatter_node.py
Version: 1.2.0
Description: Node for formatting image dimensions into standardized strings with improved tensor handling

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Node Class Definition
4. Dimension Processing Logic
5. Error Handling and Validation
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional
import torch
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def validate_dimensions(width: int, height: int) -> Tuple[bool, str]:
    """
    Validate input dimensions.
    
    Args:
        width: Input width
        height: Input height
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if width <= 0 or height <= 0:
        return False, f"Invalid dimensions: {width}x{height}. Must be positive."
    if width > 16384 or height > 16384:
        return False, f"Dimensions too large: {width}x{height}. Max is 16384."
    return True, ""

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTDimensionFormatterNode:
    """
    Formats image dimensions into standardized strings.
    Supports both direct dimension inputs and image tensors with improved handling.
    """
    
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

#------------------------------------------------------------------------------
# Section 4: Dimension Processing Logic
#------------------------------------------------------------------------------
    def _extract_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """
        Extract dimensions from image tensor with proper handling of different formats.
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple[int, int]: (width, height)
        """
        print(f"\nDEBUG: Image tensor processing:")
        print(f"- Raw tensor shape: {image.shape}")
        print(f"- Tensor type: {type(image)}")
        print(f"- Number of dimensions: {image.ndim}")
        print(f"- Shape values: {[dim for dim in image.shape]}")

        try:
            if image.ndim == 3:  # (C, H, W)
                print("- Processing 3D tensor (C, H, W)")
                height = int(image.shape[1])
                width = int(image.shape[2])
                print(f"- Extracted from 3D: h={height}, w={width}")
            elif image.ndim == 4:  # (B, C, H, W) or (B, H, W, C)
                print("- Processing 4D tensor")
                # Check if it's BCHW or BHWC format
                if image.shape[-1] in [1, 3, 4]:  # Likely BHWC
                    print("- Detected BHWC format")
                    height = int(image.shape[1])
                    width = int(image.shape[2])
                else:  # Likely BCHW
                    print("- Detected BCHW format")
                    height = int(image.shape[2])
                    width = int(image.shape[3])
                print(f"- Extracted from 4D: h={height}, w={width}")
            else:
                raise ValueError(f"Invalid image tensor dimensions: {image.ndim}")
            
            print(f"\nFinal extracted dimensions:")
            print(f"- Width: {width} (type: {type(width)})")
            print(f"- Height: {height} (type: {type(height)})")
            
            return width, height
            
        except Exception as e:
            error_msg = f"Error extracting dimensions: {str(e)}"
            logger.error(error_msg)
            print(f"\nERROR TRACE:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            print(f"- Image shape: {image.shape}")
            raise

#------------------------------------------------------------------------------
# Section 5: Error Handling and Validation
#------------------------------------------------------------------------------
    def format_dimensions(
        self,
        width: int,
        height: int,
        spacing: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """
        Format dimensions to string with improved error handling.
        
        Args:
            width: Widget input width
            height: Widget input height
            spacing: Spacing string for formatting
            image: Optional input image tensor
            
        Returns:
            Tuple[str]: Formatted dimension string
        """
        try:
            # Use image dimensions if provided
            if image is not None:
                try:
                    width, height = self._extract_dimensions(image)
                except Exception as e:
                    print(f"Failed to extract dimensions from image: {str(e)}")
                    # Fall back to widget dimensions
                    pass

            # Validate dimensions
            is_valid, error_msg = validate_dimensions(width, height)
            if not is_valid:
                logger.error(error_msg)
                print(f"Validation error: {error_msg}")
                return ("Invalid dimensions",)
            
            # Format dimension string
            result = f"{width}{spacing}x{spacing}{height}"
            
            return (result,)
            
        except Exception as e:
            error_msg = f"Error formatting dimensions: {str(e)}"
            logger.error(error_msg)
            print(f"\nERROR TRACE:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            if image is not None:
                print(f"- Image was provided with shape: {image.shape}")
            print(f"- Input width: {width} ({type(width)})")
            print(f"- Input height: {height} ({type(height)})")
            print(f"- Spacing: {spacing}")
            return ("Error formatting dimensions",)