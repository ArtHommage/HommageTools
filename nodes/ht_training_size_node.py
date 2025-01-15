"""
File: homage_tools/nodes/ht_training_size_node.py
Version: 1.2.1
Description: Node for calculating optimal training dimensions with improved tensor handling

Sections:
1. Imports and Constants
2. Helper Functions
3. Main Node Class
4. Dimension Calculation Logic
5. Error Handling and Validation
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Constants
#------------------------------------------------------------------------------
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
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

def round_to_multiple(value: float, multiple: int) -> int:
    """
    Round a value to the nearest multiple.
    
    Args:
        value: Value to round
        multiple: Multiple to round to
        
    Returns:
        int: Rounded value
    """
    return int(round(value / multiple) * multiple)

#------------------------------------------------------------------------------
# Section 3: Main Node Class
#------------------------------------------------------------------------------
class HTTrainingSizeNode:
    """
    Calculates optimal dimensions for training images with improved error handling.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "calculate_dimensions"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "scale_factor", "requires_clipping")
    
    # Standard dimensions for training
    STANDARD_DIMS: List[int] = [512, 768, 1024]
    
    # Minimum and maximum dimensions
    MIN_DIM: int = 64
    MAX_DIM: int = 8192
    ROUND_TO: int = 64
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define input parameters and their configurations."""
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

    #--------------------------------------------------------------------------
    # Section 4: Dimension Calculation Logic
    #--------------------------------------------------------------------------
    def _find_optimal_dimensions(
        self,
        current_width: int,
        current_height: int,
        scaling_mode: str
    ) -> Tuple[int, int, float, int]:
        """
        Find optimal dimensions for training.
        
        Args:
            current_width: Current image width
            current_height: Current image height
            scaling_mode: Scaling mode selection
            
        Returns:
            Tuple[int, int, float, int]: (new_width, new_height, scale_factor, requires_clipping)
        """
        # Log input parameters
        logger.debug(f"Finding optimal dimensions for {current_width}x{current_height}")
        print(f"Processing dimensions - Width: {current_width}, Height: {current_height}")
        
        current_long_edge = max(current_width, current_height)
        is_width_longer = current_width >= current_height
        aspect_ratio = current_width / current_height
        
        # Find best target dimension
        best_long_edge = None
        min_quality_loss = float('inf')
        best_scale = 1.0
        
        for target in self.STANDARD_DIMS:
            # Skip targets based on scaling mode
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
                
        # Default to smallest standard dimension if no match found
        if best_long_edge is None:
            best_long_edge = self.STANDARD_DIMS[0]
            best_scale = best_long_edge / current_long_edge
            logger.warning("No optimal dimension found, using smallest standard size")
            print("Warning: No optimal dimension found, defaulting to smallest standard size")
        
        # Calculate new dimensions
        if is_width_longer:
            new_width = best_long_edge
            new_height = round_to_multiple(best_long_edge / aspect_ratio, self.ROUND_TO)
        else:
            new_height = best_long_edge
            new_width = round_to_multiple(best_long_edge * aspect_ratio, self.ROUND_TO)
        
        # Calculate if clipping is needed
        scaled_width = int(current_width * best_scale)
        scaled_height = int(current_height * best_scale)
        requires_clipping = 1 if (scaled_width > new_width or scaled_height > new_height) else 0
        
        # Log results
        print(f"Calculated dimensions - New Width: {new_width}, New Height: {new_height}")
        print(f"Scale factor: {best_scale:.3f}, Requires clipping: {requires_clipping}")
        
        return new_width, new_height, best_scale, requires_clipping

    #--------------------------------------------------------------------------
    # Section 5: Error Handling and Validation
    #--------------------------------------------------------------------------
    def calculate_dimensions(
        self,
        width: int,
        height: int,
        scaling_mode: str,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, float, int]:
        """
        Calculate optimal dimensions for training with error handling.
        
        Args:
            width: Widget input width
            height: Widget input height
            scaling_mode: Scaling mode selection
            image: Optional input image tensor
            
        Returns:
            Tuple[int, int, float, int]: (width, height, scale_factor, requires_clipping)
        """
        try:
            # Use image dimensions if provided
            if image is not None:
                print(f"\nDEBUG: Image tensor processing:")
                print(f"- Raw tensor shape: {image.shape}")
                print(f"- Tensor type: {type(image)}")
                print(f"- Number of dimensions: {image.ndim}")
                print(f"- Shape values: {[dim for dim in image.shape]}")

                # Extract dimensions based on tensor format
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
                    
                    # Ensure we have valid numbers
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid dimensions extracted: {width}x{height}")
                        
                except Exception as e:
                    print(f"ERROR during dimension extraction: {str(e)}")
                    raise
            
            # Validate dimensions
            is_valid, error_msg = validate_dimensions(width, height)
            if not is_valid:
                logger.error(error_msg)
                print(f"Validation error: {error_msg}")
                return (0, 0, 0.0, 0)
            
            # Calculate optimal dimensions
            return self._find_optimal_dimensions(width, height, scaling_mode)
            
        except Exception as e:
            error_msg = f"Error calculating dimensions: {str(e)}"
            logger.error(error_msg)
            print(f"\nERROR TRACE:")
            print(f"- Error type: {type(e).__name__}")
            print(f"- Error message: {str(e)}")
            if image is not None:
                print(f"- Image was provided with shape: {image.shape}")
            print(f"- Input width: {width} ({type(width)})")
            print(f"- Input height: {height} ({type(height)})")
            print(f"- Scaling mode: {scaling_mode}")
            return (0, 0, 0.0, 0)