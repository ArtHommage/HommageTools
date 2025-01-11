"""
File: homage_tools/nodes/ht_layer_nodes.py

HommageTools Layer Handling Nodes
Version: 1.0.0
Description: Nodes for creating and exporting layered images in PSD and TIFF formats.
Each layer remains distinct and separate in the output file.
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from dataclasses import dataclass
from psd_tools import PSDImage
import tifffile

@dataclass
class LayerData:
    """Container for layer information"""
    image: torch.Tensor
    name: str
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get layer dimensions (height, width)"""
        return self.image.shape[-2:]

#------------------------------------------------------------------------------
# Section 2: Layer Collection Node
#------------------------------------------------------------------------------
class HTLayerCollectorNode:
    """
    Collects images into a layer stack for multi-layer file export.
    Each layer remains distinct and separate - no blending is performed.
    """
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "add_layer"
    RETURN_TYPES = ("LAYER_STACK",)
    RETURN_NAMES = ("layer_stack",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "layer_name": ("STRING", {
                    "default": "Layer",
                    "multiline": False
                })
            },
            "optional": {
                "input_stack": ("LAYER_STACK",)
            }
        }
        
    def add_layer(
        self,
        layer_image: torch.Tensor,
        layer_name: str,
        input_stack: Optional[List[LayerData]] = None
    ) -> Tuple[List[LayerData]]:
        """Add a new layer to the stack"""
        try:
            # Initialize or use existing stack
            layer_stack = input_stack if input_stack is not None else []
            
            # Create new layer
            new_layer = LayerData(
                image=layer_image,
                name=layer_name
            )
            
            # Add to stack
            layer_stack.append(new_layer)
            
            return (layer_stack,)
            
        except Exception as e:
            print(f"Error in HTLayerCollectorNode: {str(e)}")
            return ([],)

#------------------------------------------------------------------------------
# Section 3: Layer Export Node
#------------------------------------------------------------------------------
class HTLayerExportNode:
    """
    Exports collected layers to PSD or TIFF format.
    Maintains separate layers in the output file.
    """
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "export_layers"
    OUTPUT_NODE = True
    
    FORMATS = ["PSD", "TIFF"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "format": (cls.FORMATS, {
                    "default": "PSD"
                }),
                "output_path": ("STRING", {
                    "default": "output",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ()
    
    def verify_layer_dimensions(self, layer_stack: List[LayerData]) -> bool:
        """Verify all layers have the same dimensions"""
        if not layer_stack:
            raise ValueError("Empty layer stack")
            
        base_dims = layer_stack[0].dimensions
        
        for i, layer in enumerate(layer_stack[1:], 1):
            if layer.dimensions != base_dims:
                raise ValueError(
                    f"Layer dimension mismatch: Base layer is {base_dims}, "
                    f"but layer '{layer.name}' is {layer.dimensions}"
                )
        
        return True

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor to PIL Image"""
        # Convert to numpy and scale to 0-255
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # Handle single image vs batch
        if len(array.shape) == 4:
            array = array[0]
        # Convert to PIL (assuming RGB)
        return Image.fromarray(array.transpose(1, 2, 0), 'RGB')

    def _export_psd(
        self,
        layer_stack: List[LayerData],
        output_path: str
    ) -> None:
        """Export layers to PSD format"""
        # Create new PSD file
        base_layer = layer_stack[0]
        height, width = base_layer.dimensions
        
        # Create base PSD
        psd = PSDImage.new(width, height)
        
        # Add layers in reverse order (PSD layers are top-to-bottom)
        for layer_data in reversed(layer_stack):
            # Convert tensor to PIL image
            pil_image = self._tensor_to_pil(layer_data.image)
            
            # Create PSD layer
            psd.add_layer(
                name=layer_data.name,
                image=pil_image
            )
        
        # Save the file
        psd.save(output_path)

    def _export_tiff(
        self,
        layer_stack: List[LayerData],
        output_path: str
    ) -> None:
        """Export layers to TIFF format"""
        # Convert all layers to numpy arrays
        images = []
        for layer in layer_stack:
            # Convert tensor to numpy and scale to 0-255
            array = (layer.image.cpu().numpy() * 255).astype(np.uint8)
            if len(array.shape) == 4:
                array = array[0]
            array = array.transpose(1, 2, 0)  # CHW to HWC
            images.append(array)
            
        # Stack images along new axis for TIFF pages
        image_stack = np.stack(images)
            
        # Save as multi-page TIFF
        tifffile.imwrite(
            output_path,
            image_stack,
            metadata={
                'ImageDescription': '\n'.join(
                    f"Layer {i}: {layer.name}" 
                    for i, layer in enumerate(layer_stack)
                )
            },
            photometric='rgb'
        )

    def export_layers(
        self,
        layer_stack: List[LayerData],
        format: str,
        output_path: str
    ) -> Tuple:
        """Export layers to selected format"""
        try:
            # Verify dimensions
            self.verify_layer_dimensions(layer_stack)
            
            # Ensure proper file extension
            if format == "PSD" and not output_path.lower().endswith('.psd'):
                output_path += '.psd'
            elif format == "TIFF" and not output_path.lower().endswith(('.tif', '.tiff')):
                output_path += '.tiff'
            
            # Export based on format
            if format == "PSD":
                self._export_psd(layer_stack, output_path)
            else:  # TIFF
                self._export_tiff(layer_stack, output_path)
            
            print(f"Successfully saved {format} to {output_path}")
            return ()
            
        except Exception as e:
            print(f"Error in HTLayerExportNode: {str(e)}")
            return ()