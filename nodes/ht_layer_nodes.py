"""
File: homage_tools/nodes/ht_layer_nodes.py
Version: 1.0.0
Description: Simple nodes for layer collection and export
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tifffile

@dataclass
class LayerData:
    """Layer data container."""
    image: torch.Tensor
    name: str

class HTLayerCollectorNode:
    """Collects images into a layer stack."""
    
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
                "input_stack": ("LAYER_STACK",)
            }
        }
        
    def add_layer(
        self,
        image: torch.Tensor,
        layer_name: str,
        input_stack: Optional[List[LayerData]] = None
    ) -> Tuple[List[LayerData]]:
        """Add new layer to stack."""
        stack = input_stack if input_stack is not None else []
        stack.append(LayerData(image=image, name=layer_name))
        return (stack,)

class HTLayerExportNode:
    """Exports layer stack to TIFF."""
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "export_layers"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "output_path": ("STRING", {"default": "output.tiff"})
            }
        }
    
    def export_layers(
        self,
        layer_stack: List[LayerData],
        output_path: str
    ) -> Tuple:
        """Export layers to TIFF."""
        if not layer_stack:
            return ()
            
        # Ensure .tiff extension
        if not output_path.lower().endswith('.tiff'):
            output_path += '.tiff'
            
        # Convert all layers to numpy arrays
        images = []
        for layer in layer_stack:
            array = (layer.image.cpu().numpy() * 255).astype(np.uint8)
            if len(array.shape) == 4:
                array = array[0]
            array = array.transpose(1, 2, 0)  # CHW to HWC
            images.append(array)
            
        # Stack and save
        image_stack = np.stack(images)
        tifffile.imwrite(
            output_path,
            image_stack,
            photometric='rgb'
        )
        
        return ()