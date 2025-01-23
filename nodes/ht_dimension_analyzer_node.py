"""
File: homage_tools/nodes/ht_dimension_analyzer_node.py 
Version: 1.1.0
Description: Node for analyzing image dimensions and returning long/short edge values
"""

import torch
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

class HTDimensionAnalyzerNode:
    """Analyzes image dimensions and returns long/short edge values."""
    
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

    def _extract_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """Extract dimensions from image tensor in BHWC format."""
        logger.debug(f"Processing tensor with shape: {image.shape}")
        
        try:
            if len(image.shape) == 3:  # HWC format
                height, width = image.shape[0:2]
            elif len(image.shape) == 4:  # BHWC format
                height, width = image.shape[1:3]
            else:
                raise ValueError(f"Invalid image tensor dimensions: {len(image.shape)}")
                
            return height, width
            
        except Exception as e:
            logger.error(f"Dimension extraction error: {str(e)}")
            raise

    def analyze_dimensions(self, image: torch.Tensor) -> Tuple[int, int]:
        """Analyze image dimensions and return long/short edge values."""
        try:
            height, width = self._extract_dimensions(image)
            
            # Determine long and short edges
            long_edge = max(width, height)
            short_edge = min(width, height)
            
            logger.debug(f"Analyzed dimensions - Long: {long_edge}, Short: {short_edge}")
            return (long_edge, short_edge)
            
        except Exception as e:
            logger.error(f"Dimension analysis error: {str(e)}")
            return (0, 0)