"""
File: homage_tools/nodes/ht_baseshift_node.py

HommageTools Base Shift Calculator Node
Version: 1.0.0
Description: A node that calculates base shift values for images based on 
dimensions and shift parameters.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Input/Output Configuration
4. Calculation Methods
5. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional
import torch

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTBaseShiftNode:
    """
    A ComfyUI node that calculates base shift values for images.
    
    Features:
    - Calculate max shift based on image dimensions and base shift
    - Accept optional image input for automatic dimension calculation
    - Widget-based dimension input when no image is provided
    - Configurable base and max shift parameters
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_shift"
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("max_shift", "base_shift")

#------------------------------------------------------------------------------
# Section 3: Input/Output Configuration
#------------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "description": "Width of the image"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "description": "Height of the image"
                }),
                "base_shift": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "description": "Base shift value"
                }),
                "max_shift": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "description": "Maximum shift value"
                })
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

#------------------------------------------------------------------------------
# Section 4: Calculation Methods
#------------------------------------------------------------------------------
    def _calculate_max_shift(
        self,
        width: int,
        height: int,
        base_shift: float,
        max_shift: float
    ) -> float:
        """
        Calculate the max shift value based on image dimensions and shift parameters.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            base_shift: Base shift value
            max_shift: Maximum shift value
            
        Returns:
            float: Calculated max shift value
        """
        try:
            # Calculate according to the provided formula
            calculated_shift = (
                (base_shift - max_shift) / 
                (256 - ((width * height) / 256)) * 
                3840 + base_shift
            )
            return float(calculated_shift)
            
        except ZeroDivisionError:
            print("Warning: Division by zero in shift calculation. Using base_shift.")
            return float(base_shift)
        except Exception as e:
            print(f"Error in shift calculation: {str(e)}. Using base_shift.")
            return float(base_shift)

#------------------------------------------------------------------------------
# Section 5: Main Processing Logic
#------------------------------------------------------------------------------
    def calculate_shift(
        self,
        image_width: int,
        image_height: int,
        base_shift: float,
        max_shift: float,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[float, float]:
        """
        Main processing function to calculate shift values.
        
        Args:
            image_width: Width from widget input
            image_height: Height from widget input
            base_shift: Base shift value
            max_shift: Maximum shift value
            image: Optional input image tensor
            
        Returns:
            Tuple[float, float]: (max_shift, base_shift) values
        """
        try:
            # If image is provided, use its dimensions
            if image is not None:
                image_height, image_width = image.shape[-2:]
            
            # Calculate max shift using the formula
            calculated_max_shift = self._calculate_max_shift(
                image_width,
                image_height,
                base_shift,
                max_shift
            )
            
            # Return both calculated max shift and original base shift
            return (calculated_max_shift, base_shift)
            
        except Exception as e:
            print(f"Error in HTBaseShiftNode: {str(e)}")
            return (float(max_shift), float(base_shift))  # Return inputs on error