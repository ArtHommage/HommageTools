"""
File: homage_tools/nodes/ht_save_image_plus.py
Version: 1.1.0
Description: Enhanced image saving node with multiple format support, mask handling, and text output capabilities

Sections:
1. Imports and Constants
2. Helper Functions
3. Main Node Class
4. Format Conversion Functions
5. File Management
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Constants
#------------------------------------------------------------------------------
import os
import json
from PIL import Image, PngImagePlugin
import numpy as np
import torch
import logging
import folder_paths
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger('HommageTools')

VERSION = "1.1.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def ensure_directory_exists(directory: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False

def get_sequential_filename(base_path: str, extension: str) -> Tuple[str, int]:
    """Get next available filename with sequence number."""
    counter = 1
    while True:
        file_name = f"{base_path}_{counter:05d}{extension}"
        if not os.path.exists(file_name):
            return file_name, counter
        counter += 1

def write_text_file(file_path: str, content: str, encoding: str = 'UTF-8') -> bool:
    """Write content to text file with specified encoding."""
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {str(e)}")
        return False

#------------------------------------------------------------------------------
# Section 3: Main Node Class
#------------------------------------------------------------------------------
class HTSaveImagePlus:
    """Enhanced image saving node with multiple format and mask support."""
    
    CATEGORY = "HommageTools/output"
    FUNCTION = "save_image_plus"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "images": ("IMAGE", {
                    "tooltip": "Images in BHWC format to save"
                }),
                "output_format": (["PNG", "JPEG", "TIFF"], {
                    "default": "PNG",
                    "tooltip": "Output image format"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_",
                    "tooltip": "Prefix or full filename for output"
                }),
                "add_sequence_number": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append sequential number to filename"
                }),
                "output_dir": ("STRING", {
                    "default": folder_paths.get_output_directory(),
                    "tooltip": "Output directory path"
                })
            },
            "optional": {
                # Image Format Quality Options
                "jpeg_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG Quality (1-100)"
                }),
                "tiff_compression": (["adobe_deflate", "none"], {
                    "default": "adobe_deflate",
                    "tooltip": "TIFF compression method"
                }),
                # Mask/Alpha Options
                "mask": ("MASK", {
                    "tooltip": "Optional mask to use as alpha channel"
                }),
                "save_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save alpha channel if present"
                }),
                # Text Output Options  
                "text_content": ("STRING", {
                    "multiline": True,
                    "tooltip": "Optional text content to save alongside image"
                }),
                "text_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Extension for text file"
                }),
                "text_encoding": (["UTF-8", "ASCII", "UTF-16", "UTF-32"], {
                    "default": "UTF-8",
                    "tooltip": "Text file encoding"
                }),
                # Metadata control
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save prompt metadata in PNG files"
                })
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    #--------------------------------------------------------------------------
    # Section 4: Format Conversion Functions
    #--------------------------------------------------------------------------
    def _prepare_image(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Image.Image:
        """Convert BHWC tensor to PIL Image, handling alpha channel."""
        # Ensure CPU tensors
        image = image.cpu()
        
        # Handle alpha channel
        if image.shape[-1] == 4:
            # Image already has alpha
            i = 255. * image.numpy()
            img_data = np.clip(i, 0, 255).astype(np.uint8)
        elif mask is not None:
            # Use provided mask as alpha
            mask = mask.cpu()
            i = 255. * image.numpy()
            rgb = np.clip(i, 0, 255).astype(np.uint8)
            # Ensure mask is 2D and matches image dimensions
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            a = 255. * (1 - mask.numpy())  # Invert mask for alpha
            img_data = np.concatenate([rgb, a[..., None]], axis=-1)
        else:
            # No alpha channel
            i = 255. * image.numpy()
            img_data = np.clip(i, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_data)

    def _save_image_with_format(
        self,
        image: Image.Image,
        file_path: str,
        format: str,
        save_alpha: bool = True,
        metadata: Optional[dict] = None,
        **kwargs
    ) -> bool:
        """Save image with specified format and options."""
        try:
            if not save_alpha and image.mode == 'RGBA':
                image = image.convert('RGB')
                
            if format == "PNG":
                # Create new PngInfo for metadata if needed
                # Create metadata for PNG if needed
                png_info = None
                if metadata and kwargs.get('save_metadata', True):
                    png_info = PngImagePlugin.PngInfo()
                    for k, v in metadata.items():
                        if isinstance(v, dict) or isinstance(v, list):
                            v = json.dumps(v)
                        png_info.add_text(str(k), str(v))

                image.save(file_path, format='PNG', pnginfo=png_info)
            elif format == "JPEG":
                # JPEG doesn't support alpha, always convert
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(file_path, format='JPEG', 
                         quality=kwargs.get('jpeg_quality', 90))
            elif format == "TIFF":
                # TIFF can handle alpha channel
                image.save(file_path, format='TIFF',
                         compression=kwargs.get('tiff_compression', 'adobe_deflate'))
            return True
        except Exception as e:
            logger.error(f"Failed to save image {file_path}: {str(e)}")
            return False

    #--------------------------------------------------------------------------
    # Section 5: Main Processing
    #--------------------------------------------------------------------------
    def save_image_plus(
        self,
        images: torch.Tensor,
        output_format: str,
        filename_prefix: str,
        add_sequence_number: bool,
        output_dir: str,
        prompt: Optional[Dict] = None,
        extra_pnginfo: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None,
        save_alpha: bool = True,
        jpeg_quality: int = 90,
        tiff_compression: str = "adobe_deflate",
        text_content: Optional[str] = None,
        text_extension: str = ".txt",
        text_encoding: str = "UTF-8",
        save_metadata: bool = True
    ) -> Dict:
        """Process and save images with optional mask and text output."""
        print(f"\nHTSaveImagePlus v{VERSION} - Processing")
        print(f"Target directory: {output_dir}")
        print(f"Format: {output_format}, Sequence numbering: {'Enabled' if add_sequence_number else 'Disabled'}")
        
        # Ensure output directory exists
        if not ensure_directory_exists(output_dir):
            return {"ui": {"error": f"Failed to create output directory: {output_dir}"}}

        results = []
        
        # Prepare metadata for PNG
        metadata = None
        if output_format == "PNG" and save_metadata:
            metadata = {}
            if prompt is not None:
                metadata["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)

        # Process each image in batch
        for idx, image in enumerate(images):
            # Create base filename
            base_name = os.path.join(output_dir, filename_prefix)
            extension = f".{output_format.lower()}"
            
            if add_sequence_number:
                file_path, counter = get_sequential_filename(base_name, extension)
            else:
                file_path = base_name + extension
                counter = idx
            
            # Get mask for this image if provided
            current_mask = None
            if mask is not None:
                if len(mask.shape) == 4:  # BHWC format
                    current_mask = mask[idx] if idx < mask.shape[0] else mask[0]
                else:  # Single mask for all images
                    current_mask = mask
            
            # Convert and save image
            pil_image = self._prepare_image(image, current_mask)
            if self._save_image_with_format(
                pil_image, 
                file_path,
                output_format,
                save_alpha=save_alpha,
                metadata=metadata,
                jpeg_quality=jpeg_quality,
                tiff_compression=tiff_compression,
                save_metadata=save_metadata
            ):
                # Log successful image save to console
                print(f"Successfully saved image: {file_path}")
                
                results.append({
                    "filename": os.path.basename(file_path),
                    "subfolder": os.path.relpath(output_dir, folder_paths.get_output_directory()),
                    "type": "output"
                })
                
                # Save accompanying text file if provided
                if text_content:
                    text_path = os.path.splitext(file_path)[0] + text_extension
                    if write_text_file(text_path, text_content, text_encoding):
                        print(f"Successfully saved text file: {text_path}")
                    else:
                        logger.warning(f"Failed to save text file for image {file_path}")

        return {"ui": {"images": results}}