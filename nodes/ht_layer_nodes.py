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

#------------------------------------------------------------------------------
# Section 1: Helper Functions and Data Classes
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Section 2: Layer Collection Node
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Section 3: Layer Export Node
#------------------------------------------------------------------------------
class HTLayerExportNode:
    """Exports layer stack to PSD or TIFF format."""
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "export_layers"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "output_path": ("STRING", {
                    "default": "output.psd",
                    "multiline": False
                }),
                "format": (["psd", "tiff"], {
                    "default": "psd"
                })
            }
        }

    def prepare_layer_data(
        self,
        layer: LayerData
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare layer data for export."""
        # Convert to numpy and ensure correct format
        image_np = layer.image.cpu().numpy()
        
        # Handle alpha channel
        if image_np.shape[-1] == 4:
            return image_np[..., :3], image_np[..., 3]
        return image_np, None

    def export_as_psd(
        self,
        layer_stack: List[LayerData],
        output_path: str
    ) -> None:
        """Export layers as PSD file."""
        if not layer_stack:
            raise ValueError("No layers to export")
            
        # Get dimensions from first layer
        height, width = layer_stack[0].image.shape[1:3]
        
        # Create new PSD
        psd = PSDImage.new(width, height, color_mode=ColorMode.RGB)
        
        # Add layers in reverse order (PSD layers are bottom-up)
        for layer in reversed(layer_stack):
            rgb_data, alpha_data = self.prepare_layer_data(layer)
            
            # Create layer
            pil_image = Image.fromarray((rgb_data * 255).astype(np.uint8))
            if alpha_data is not None:
                alpha = Image.fromarray((alpha_data * 255).astype(np.uint8))
                pil_image.putalpha(alpha)
                
            layer_specs = {"name": layer.name}
            psd.compose([(pil_image, layer_specs)])
            
        # Save PSD
        psd.save(output_path)

    def export_as_tiff(
        self,
        layer_stack: List[LayerData],
        output_path: str
    ) -> None:
        """Export layers as multi-page TIFF."""
        if not layer_stack:
            raise ValueError("No layers to export")
            
        # Prepare data for TIFF
        pages = []
        for layer in layer_stack:
            rgb_data, alpha_data = self.prepare_layer_data(layer)
            
            # Convert to 8-bit format
            rgb_data = (rgb_data * 255).astype(np.uint8)
            if alpha_data is not None:
                alpha_data = (alpha_data * 255).astype(np.uint8)
                
            # Combine RGB and alpha if present
            if alpha_data is not None:
                page_data = np.dstack([rgb_data, alpha_data])
            else:
                page_data = rgb_data
                
            pages.append(page_data)
            
        # Save multi-page TIFF
        tifffile.imwrite(
            output_path,
            pages,
            photometric='rgb',
            planarconfig='contig'
        )

    def export_layers(
        self,
        layer_stack: List[LayerData],
        output_path: str,
        format: str = "psd"
    ) -> Tuple:
        """Export layer stack to specified format."""
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Export based on format
            if format == "psd":
                self.export_as_psd(layer_stack, output_path)
            else:  # tiff
                self.export_as_tiff(layer_stack, output_path)
                
            print(f"Successfully exported {len(layer_stack)} layers to {output_path}")
            return tuple()
            
        except Exception as e:
            logger.error(f"Error exporting layers: {str(e)}")
            print(f"\nERROR exporting layers: {str(e)}")
            if layer_stack:
                print(f"Layer stack size: {len(layer_stack)}")
            raise