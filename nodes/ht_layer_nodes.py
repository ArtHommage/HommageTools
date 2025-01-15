"""
File: homage_tools/nodes/ht_layer_nodes.py
Version: 1.4.1
Description: Layer collection and export nodes for ComfyUI with improved tensor handling

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Data Classes
4. Layer Collector Node
5. Layer Export Node
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
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

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def normalize_tensor_dimensions(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Normalize input tensor to standard BCHW format with proper dimension detection.
    
    Args:
        tensor: Input tensor in various possible formats
        
    Returns:
        Tuple[torch.Tensor, Dict[str, int]]: Normalized tensor and dimension info
    """
    print(f"\nDEBUG: Tensor normalization:")
    print(f"- Input shape: {tensor.shape}")
    
    # Handle different dimension formats
    if len(tensor.shape) == 3:  # CHW format
        print("- Converting CHW to BCHW")
        tensor = tensor.unsqueeze(0)
    
    if len(tensor.shape) == 4:
        if tensor.shape[-1] in [1, 3, 4]:  # BHWC format
            print("- Converting BHWC to BCHW")
            tensor = tensor.permute(0, 3, 1, 2)
    
    b, c, h, w = tensor.shape
    print(f"- Normalized shape: {tensor.shape}")
    
    return tensor, {
        "batch_size": b,
        "channels": c,
        "height": h,
        "width": w
    }

#------------------------------------------------------------------------------
# Section 3: Data Classes
#------------------------------------------------------------------------------
@dataclass
class LayerData:
    """Layer data container with metadata."""
    image: torch.Tensor
    name: str
    
    def __post_init__(self):
        """Validate and log layer creation."""
        logger.debug(f"Created layer '{self.name}' with shape {self.image.shape}")

#------------------------------------------------------------------------------
# Section 4: Layer Collector Node
#------------------------------------------------------------------------------
class HTLayerCollectorNode:
    """
    Collects images into a layer stack with proper alpha handling.
    Converts grayscale to RGB, preserves existing alpha channels,
    and allows mask override.
    """
    
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
        """
        Process image and handle alpha channel/mask appropriately.
        
        Args:
            image: Input image tensor
            mask: Optional mask tensor - overrides any existing alpha
            
        Returns:
            torch.Tensor: Processed image with preserved format
        """
        # Normalize dimensions
        image, image_info = normalize_tensor_dimensions(image)
        print(f"Processing image with dimensions: {image_info}")
        
        # Handle mask if provided
        if mask is not None:
            mask, mask_info = normalize_tensor_dimensions(mask)
            print(f"Processing mask with dimensions: {mask_info}")
            
            # Ensure mask matches image dimensions
            if mask_info['height'] != image_info['height'] or mask_info['width'] != image_info['width']:
                print(f"Warning: Mask dimensions don't match image")
                mask = torch.nn.functional.interpolate(
                    mask,
                    size=(image_info['height'], image_info['width']),
                    mode='bilinear'
                )
        
        # Convert grayscale to RGB
        if image_info['channels'] == 1:
            print("Converting grayscale to RGB")
            image = image.repeat(1, 3, 1, 1)
            image_info['channels'] = 3
        
        # Handle mask/alpha
        if mask is not None:
            print("Using provided mask as alpha channel")
            if image_info['channels'] == 4:
                print("Replacing existing alpha with provided mask")
                image = image[:, :3, :, :]
            return torch.cat([image, mask], dim=1)
        
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
        """
        Add new layer to stack with mask/alpha handling.
        
        Args:
            image: Image tensor
            layer_name: Name for the layer
            mask: Optional mask tensor (overrides any existing alpha)
            input_stack: Optional existing stack
            
        Returns:
            Tuple[List[LayerData]]: Updated layer stack
        """
        try:
            # Initialize or use existing stack
            stack = input_stack if input_stack is not None else []
            
            # Process image with alpha/mask
            processed_image = self.process_image_with_mask(image, mask)
            
            # Create new layer
            new_layer = LayerData(image=processed_image, name=layer_name)
            print(f"Created layer '{layer_name}' with shape {processed_image.shape}")
            
            # Add layer
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
# Section 5: Layer Export Node
#------------------------------------------------------------------------------
class HTLayerExportNode:
    """Exports layer stack to TIFF or PSD."""
    
    CATEGORY = "HommageTools/Layers"
    FUNCTION = "export_layers"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    
    SUPPORTED_FORMATS = {
        '.tiff': 'TIFF',
        '.tif': 'TIFF',
        '.psd': 'PSD'
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "save_path": ("STRING", {
                    "default": "outputs/layer_exports",
                    "description": "Directory to save files"
                }),
                "filename": ("STRING", {
                    "default": "output.tiff",
                    "description": "Filename with extension (.tiff, .tif, or .psd)"
                }),
                "layer_stack": ("LAYER_STACK",)
            }
        }
    
    def _resolve_save_path(self, base_path: str, filename: str) -> Tuple[bool, str, Optional[str]]:
        """Resolve and validate save path."""
        try:
            # Clean up paths
            base_path = os.path.normpath(base_path.strip())
            filename = os.path.basename(filename.strip())
            
            # If relative path, make relative to ComfyUI root
            if not os.path.isabs(base_path):
                comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                base_path = os.path.join(comfy_root, base_path)
            
            # Get extension
            ext = os.path.splitext(filename)[1].lower()
            if not ext or ext not in self.SUPPORTED_FORMATS:
                return False, f"Invalid or unsupported file extension: {ext}", None
            
            # Create directory
            os.makedirs(base_path, exist_ok=True)
            
            # Combine paths
            full_path = os.path.join(base_path, filename)
            
            return True, "", full_path
            
        except Exception as e:
            return False, f"Error resolving save path: {str(e)}", None

    def convert_tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array with proper dimension handling."""
        try:
            # Normalize tensor dimensions first
            tensor, dim_info = normalize_tensor_dimensions(tensor)
            print(f"Converting tensor with dimensions: {dim_info}")
            
            # Move to CPU and remove batch dimension
            array = tensor.cpu().numpy()
            array = array[0]  # Remove batch dimension
            
            # Convert to HWC format and normalize
            array = array.transpose(1, 2, 0)  # CHW -> HWC
            array = (array * 255).clip(0, 255).astype(np.uint8)
            
            print(f"Converted to numpy array with shape: {array.shape}")
            return array
            
        except Exception as e:
            print(f"Error converting tensor to numpy: {str(e)}")
            raise

    def export_tiff(self, layers: List[np.ndarray], output_path: str) -> None:
        """Export layers as TIFF file with dimension validation."""
        try:
            # Validate layer dimensions
            if not layers:
                raise ValueError("No layers to export")
            
            base_shape = layers[0].shape
            for idx, layer in enumerate(layers):
                if layer.shape != base_shape:
                    raise ValueError(
                        f"Layer {idx} has inconsistent dimensions: "
                        f"expected {base_shape}, got {layer.shape}"
                    )
            
            # Stack layers and save
            image_stack = np.stack(layers)
            print(f"Exporting TIFF with shape: {image_stack.shape}")
            
            # Determine if we have alpha channels
            has_alpha = image_stack.shape[-1] == 4
            
            tifffile.imwrite(
                output_path,
                image_stack,
                photometric='rgba' if has_alpha else 'rgb',
                compression='lzw'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save TIFF file: {str(e)}")
    
    def export_psd(self, layers: List[np.ndarray], names: List[str], output_path: str) -> None:
        """Export layers as PSD file with dimension validation."""
        try:
            # Validate dimensions
            if not layers:
                raise ValueError("No layers to export")
                
            first_layer = layers[0]
            height, width, channels = first_layer.shape
            has_alpha = channels == 4
            
            print(f"Creating PSD with dimensions: {width}x{height}, {channels} channels")
            
            # Create new PSD
            psd = PSDImage.new(width, height, color_mode=ColorMode.RGB)
            
            # Add each layer
            for layer_array, name in zip(layers, names):
                if layer_array.shape != (height, width, channels):
                    print(f"Warning: Layer '{name}' has different dimensions: {layer_array.shape}")
                    continue
                    
                layer_image = Image.fromarray(layer_array)
                psd.compose([{
                    'name': name,
                    'image': layer_image
                }])
            
            # Save PSD
            psd.save(output_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save PSD file: {str(e)}")
    
    def export_layers(
        self,
        save_path: str,
        filename: str,
        layer_stack: List[LayerData]
    ) -> Tuple:
        """Export layers to file with improved error handling."""
        try:
            if not layer_stack:
                print("Warning: Empty layer stack, nothing to export")
                return ()
            
            # Resolve and validate paths
            success, error_msg, full_path = self._resolve_save_path(save_path, filename)
            if not success:
                raise ValueError(error_msg)
                
            print(f"\nPreparing to export:")
            print(f"Save directory: {os.path.dirname(full_path)}")
            print(f"Filename: {os.path.basename(full_path)}")
            
            # Convert layers to numpy arrays
            print("\nProcessing layers:")
            numpy_layers = []
            for idx, layer in enumerate(layer_stack):
                print(f"Layer {idx}: '{layer.name}'")
                array = self.convert_tensor_to_numpy(layer.image)
                numpy_layers.append(array)
                print(f"- Converted shape: {array.shape}")
            
            # Get format and export
            ext = os.path.splitext(filename)[1].lower()
            format_name = self.SUPPORTED_FORMATS[ext]
            print(f"\nExporting as {format_name}...")
            
            if format_name == 'TIFF':
                self.export_tiff(numpy_layers, full_path)
            elif format_name == 'PSD':
                layer_names = [layer.name for layer in layer_stack]
                self.export_psd(numpy_layers, layer_names, full_path)
            
            print(f"Successfully exported to: {full_path}")
            return ()
            
        except Exception as e:
            logger.error(f"Error in export_layers: {str(e)}")
            print(f"\nERROR in export_layers: {str(e)}")
            raise