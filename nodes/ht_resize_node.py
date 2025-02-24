"""
File: homage_tools/nodes/ht_resize_node.py
Description: Node for intelligent image and latent resizing with proper BHWC handling
Version: 1.2.0
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants
#------------------------------------------------------------------------------
VERSION = "1.2.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_dimensions(tensor: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """Verify and extract tensor dimensions."""
    shape = tensor.shape
    print(f"{context} - Shape: {shape}")
    
    if len(shape) == 3:
        height, width, channels = shape
        batch = 1
        print(f"{context} - HWC format detected")
    elif len(shape) == 4:
        batch, height, width, channels = shape
        print(f"{context} - BHWC format detected")
    else:
        raise ValueError(f"Invalid shape: {shape}")
        
    print(f"{context} - Dims: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def get_nearest_divisible_size(size: int, divisor: int) -> int:
    """Calculate nearest size divisible by divisor."""
    return ((size + divisor - 1) // divisor) * divisor

#------------------------------------------------------------------------------
# Section 3: Node Definition
#------------------------------------------------------------------------------
class HTResizeNode:
    """Smart resize node with format verification."""
    
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("resized_image", "resized_latent")
    FUNCTION = "resize_media"

    INTERPOLATION_MODES = ["nearest", "linear", "bilinear", "bicubic", "area", "lanczos"]
    SCALING_MODES = ["short_side", "long_side"]
    CROP_MODES = ["center", "top", "bottom", "left", "right"]
    DIVISIBILITY_OPTIONS = ["8", "64"]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "divisible_by": (cls.DIVISIBILITY_OPTIONS, {
                    "default": "8"
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "bicubic"
                }),
                "scaling_mode": (cls.SCALING_MODES, {
                    "default": "short_side"
                }),
                "crop_or_pad_mode": (cls.CROP_MODES, {
                    "default": "center"
                })
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",)
            }
        }

    def resize_media(
        self,
        divisible_by: str,
        interpolation: str,
        scaling_mode: str,
        crop_or_pad_mode: str,
        image: Optional[torch.Tensor] = None,
        latent: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Main resize function with tensor verification."""
        print(f"\nHTResizeNode v{VERSION} - Processing")
        
        result_image = None
        result_latent = None
        divisor = int(divisible_by)

        if image is not None:
            print("\nProcessing Image:")
            batch, height, width, channels = verify_tensor_dimensions(image, "Input Image")
            print(f"Value range: min={image.min():.3f}, max={image.max():.3f}")

            # Convert to BCHW for interpolation
            x = image.permute(0, 3, 1, 2)
            print("Converted to BCHW for processing")

            # Calculate target size
            target_width, target_height = self.calculate_target_dimensions(
                width, height, divisor, scaling_mode
            )
            print(f"Target size: {target_width}x{target_height}")

            # Apply interpolation
            processed = F.interpolate(
                x,
                size=(target_height, target_width),
                mode=interpolation if interpolation != 'lanczos' else 'bicubic',
                antialias=True if interpolation in ['bilinear', 'bicubic', 'lanczos'] else None,
                align_corners=False if interpolation in ['linear', 'bilinear', 'bicubic'] else None
            )

            # Convert back to BHWC
            processed = processed.permute(0, 2, 3, 1)
            print("Converted back to BHWC")

            # Apply cropping/padding
            result_image = self.apply_crop_or_pad(
                processed, target_width, target_height, crop_or_pad_mode
            )

            print(f"Output shape: {result_image.shape}")
            print(f"Value range: min={result_image.min():.3f}, max={result_image.max():.3f}")

        if latent is not None:
            print("\nProcessing Latent:")
            latent_tensor = latent["samples"]
            batch, height, width, channels = verify_tensor_dimensions(latent_tensor, "Input Latent")

            # Scale dimensions (latents are 1/8 size)
            width *= 8
            height *= 8
            target_width, target_height = self.calculate_target_dimensions(
                width, height, divisor, scaling_mode
            )
            target_width //= 8
            target_height //= 8
            print(f"Scaled target size: {target_width}x{target_height}")

            # Process similar to image
            x = latent_tensor.permute(0, 3, 1, 2)
            processed = F.interpolate(
                x,
                size=(target_height, target_width),
                mode=interpolation if interpolation != 'lanczos' else 'bicubic',
                antialias=True if interpolation in ['bilinear', 'bicubic', 'lanczos'] else None,
                align_corners=False if interpolation in ['linear', 'bilinear', 'bicubic'] else None
            )
            processed = processed.permute(0, 2, 3, 1)

            result_latent = {"samples": self.apply_crop_or_pad(
                processed, target_width, target_height, crop_or_pad_mode
            )}

            print(f"Output latent shape: {result_latent['samples'].shape}")

        return (result_image, result_latent)

    def calculate_target_dimensions(
        self, 
        width: int,
        height: int,
        divisor: int,
        scaling_mode: str
    ) -> Tuple[int, int]:
        """Calculate target dimensions with divisibility."""
        aspect_ratio = width / height
        
        if scaling_mode == "short_side":
            if height < width:
                new_height = get_nearest_divisible_size(height, divisor)
                new_width = get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                new_width = get_nearest_divisible_size(width, divisor)
                new_height = get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)
        else:
            if height > width:
                new_height = get_nearest_divisible_size(height, divisor)
                new_width = get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                new_width = get_nearest_divisible_size(width, divisor)
                new_height = get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)
                
        return new_width, new_height

    def apply_crop_or_pad(
        self,
        tensor: torch.Tensor,
        target_width: int,
        target_height: int,
        mode: str
    ) -> torch.Tensor:
        """Apply cropping or padding while maintaining BHWC format."""
        current_height, current_width = tensor.shape[1:3]
        
        if current_height == target_height and current_width == target_width:
            return tensor

        # Calculate padding/cropping
        if mode == "center":
            pad_left = max(0, (target_width - current_width) // 2)
            pad_right = max(0, target_width - current_width - pad_left)
            pad_top = max(0, (target_height - current_height) // 2)
            pad_bottom = max(0, target_height - current_height - pad_top)
        elif mode in ["top", "bottom"]:
            pad_left = max(0, (target_width - current_width) // 2)
            pad_right = max(0, target_width - current_width - pad_left)
            if mode == "top":
                pad_top, pad_bottom = 0, max(0, target_height - current_height)
            else:
                pad_top, pad_bottom = max(0, target_height - current_height), 0
        else:  # left or right
            pad_top = max(0, (target_height - current_height) // 2)
            pad_bottom = max(0, target_height - current_height - pad_top)
            if mode == "left":
                pad_left, pad_right = 0, max(0, target_width - current_width)
            else:
                pad_left, pad_right = max(0, target_width - current_width), 0

        if pad_left + pad_right + pad_top + pad_bottom > 0:
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            print(f"Applying padding: {padding}")
            tensor = F.pad(tensor, padding, mode='constant', value=0)

        return tensor