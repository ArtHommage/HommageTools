"""
File: homage_tools/nodes/ht_dimension_analyzer_node.py 
Version: 1.0.0
Description: Node for analyzing image dimensions and returning long/short edge values

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Image Processing Logic
4. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
from typing import Dict, Any, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTDimensionAnalyzerNode:
    """
    Analyzes image dimensions and returns long and short edge values.
    Handles various tensor formats with robust dimension extraction.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "analyze_dimensions"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("long_edge", "short_edge")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

#------------------------------------------------------------------------------
# Section 3: Image Processing Logic
#------------------------------------------------------------------------------
    def _extract_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """
        Extract dimensions from image tensor with proper format handling.
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple[int, int]: (height, width)
        """
        logger.debug(f"Processing tensor with shape: {image.shape}")
        
        try:
            if image.ndim == 3:  # (C, H, W)
                height = int(image.shape[1])
                width = int(image.shape[2])
            elif image.ndim == 4:  # (B, C, H, W) or (B, H, W, C)
                if image.shape[-1] in [1, 3, 4]:  # BHWC format
                    height = int(image.shape[1])
                    width = int(image.shape[2])
                else:  # BCHW format
                    height = int(image.shape[2])
                    width = int(image.shape[3])
            else:
                raise ValueError(f"Invalid image tensor dimensions: {image.ndim}")
                
            return height, width
            
        except Exception as e:
            logger.error(f"Dimension extraction error: {str(e)}")
            raise

#------------------------------------------------------------------------------
# Section 4: Error Handling
#------------------------------------------------------------------------------
    def analyze_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """
        Analyze image dimensions and return long/short edge values.
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple[int, int]: (long_edge, short_edge)
        """
        try:
            height, width = self._extract_dimensions(image)
            
            # Determine long and short edges
            long_edge = max(width, height)
            short_edge = min(width, height)
            
            logger.debug(f"Analyzed dimensions - Long: {long_edge}, Short: {short_edge}")
            return (long_edge, short_edge)
            
        except Exception as e:
            logger.error(f"Dimension analysis error: {str(e)}")
            return (0, 0)  # Safe fallback values