"""
File: homage_tools/nodes/ht_layer_nodes.py
Version: 1.4.2
Description: Layer collection and export nodes for ComfyUI with BHWC tensor handling
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tifffile
import logging
import os
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import ColorMode

logger = logging.getLogger('HommageTools')

def process_tensor_dimensions(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Convert tensor to BHWC format."""
    print(f"\nDEBUG: Tensor processing:")
    print(f"- Input shape: {tensor.shape}")
    
    # Handle different dimension formats
    if len(tensor.shape) == 3:  # HWC format
        tensor = tensor.unsqueeze(0)
        
    # Ensure BHWC format
    if len(tensor.shape) == 4 and tensor.shape[-1] not in [1, 3, 4]:
        # Convert from BCHW to BHWC
        tensor = tensor.permute(0, 2, 3, 1)
    
    b, h, w, c = tensor.shape
    print(f"- Normalized shape: {tensor.shape} (BHWC)")
    
    return tensor, {
        "batch_size": b,
        "height": h,
        "width": w,
        "channels": c
    }

@dataclass
class LayerData:
    """Layer data container with metadata."""
    image: torch.Tensor  # Stored in BHWC format
    name: str
    
    def __post_init__(self):
        """Validate layer creation."""
        logger.debug(f"Created layer '{self.name}' with shape {self.image.shape}")

class HTLayerCollectorNode:
    """Collects images into a layer stack with BHWC format handling."""
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "add_layer"
    RETURN_TYPES = ("LAYER_STACK",)
    RETURN_NAMES = ("layer_stack",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "layer_name": ("STRING", {"default": "Layer"})
            },
            "optional": {
                "mask": ("MASK",),
                "input_stack": ("LAYER_STACK",)
            }
        }
        
    def process_image_with_mask(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process image and handle alpha channel/mask in BHWC format."""
        # Normalize dimensions
        image, image_info = process_tensor_dimensions(image)
        print(f"Processing image with dimensions: {image_info}")
        
        # Handle mask if provided
        if mask is not None:
            mask, mask_info = process_tensor_dimensions(mask)
            print(f"Processing mask with dimensions: {mask_info}")
            
            # Ensure mask matches image dimensions
            if mask_info['height'] != image_info['height'] or mask_info['width'] != image_info['width']:
                print(f"Warning: Mask dimensions don't match image")
                mask = torch.nn.functional.interpolate(
                    mask.permute(0, 3, 1, 2),  # Convert to BCHW for interpolate
                    size=(image_info['height'], image_info['width']),
                    mode='bilinear'
                ).permute(0, 2, 3, 1)  # Back to BHWC
        
        # Convert grayscale to RGB
        if image_info['channels'] == 1:
            print("Converting grayscale to RGB")
            image = image.repeat(1, 1, 1, 3)
            image_info['channels'] = 3
        
        # Handle mask/alpha
        if mask is not None:
            print("Using provided mask as alpha channel")
            if image_info['channels'] == 4:
                print("Replacing existing alpha with provided mask")
                image = image[..., :3]
            return torch.cat([image, mask], dim=-1)
        
        # No mask provided - keep existing alpha if present
        if image_info['channels'] == 4:
            print("Preserving existing alpha channel")
        elif image_info['channels'] == 3:
            print("Keeping as RGB (no alpha)")
            
        return image
        
    def add_layer(
        self,
        image: torch.Tensor,
        layer_name: str,
        mask: Optional[torch.Tensor] = None,
        input_stack: Optional[List[LayerData]] = None
    ) -> Tuple[List[LayerData]]:
        """Add new layer to stack with BHWC format handling."""
        try:
            stack = input_stack if input_stack is not None else []
            
            processed_image = self.process_image_with_mask(image, mask)
            
            new_layer = LayerData(image=processed_image, name=layer_name)
            print(f"Created layer '{layer_name}' with shape {processed_image.shape}")
            
            stack.append(new_layer)
            return (stack,)
            
        except Exception as e:
            logger.error(f"Error in add_layer: {str(e)}")
            print(f"\nERROR in add_layer: {str(e)}")
            if input_stack:
                print(f"Current stack size: {len(input_stack)} layers")
            print(f"Attempting to add layer '{layer_name}' with shape {image.shape}")
            raise