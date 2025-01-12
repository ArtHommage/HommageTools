"""
File: homage_tools/nodes/ht_resize_node.py
Version: 1.1.0
Description: Node for intelligent image and latent resizing with Lanczos interpolation
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

class HTResizeNode:
    """
    Provides advanced resizing for images and latents with extended interpolation modes.
    """
    
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

    def get_nearest_divisible_size(self, size: int, divisor: int) -> int:
        """Calculate nearest size divisible by divisor."""
        return ((size + divisor - 1) // divisor) * divisor

    def calculate_target_dimensions(
        self, 
        current_width: int, 
        current_height: int, 
        divisor: int,
        scaling_mode: str
    ) -> Tuple[int, int]:
        """Calculate dimensions maintaining aspect ratio."""
        aspect_ratio = current_width / current_height
        
        if scaling_mode == "short_side":
            if current_height < current_width:
                new_height = self.get_nearest_divisible_size(current_height, divisor)
                new_width = self.get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                new_width = self.get_nearest_divisible_size(current_width, divisor)
                new_height = self.get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)
        else:
            if current_height > current_width:
                new_height = self.get_nearest_divisible_size(current_height, divisor)
                new_width = self.get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                new_width = self.get_nearest_divisible_size(current_width, divisor)
                new_height = self.get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)
                
        return new_width, new_height

    def apply_crop_or_pad(
        self,
        tensor: torch.Tensor,
        target_width: int,
        target_height: int,
        mode: str
    ) -> torch.Tensor:
        """Apply cropping or padding based on mode."""
        current_height, current_width = tensor.shape[-2:]
        
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

        # Apply padding if needed
        if pad_left + pad_right + pad_top + pad_bottom > 0:
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            tensor = F.pad(tensor, padding, mode='constant', value=0)

        return tensor

    def resize_media(
        self,
        divisible_by: str,
        interpolation: str,
        scaling_mode: str,
        crop_or_pad_mode: str,
        image: Optional[torch.Tensor] = None,
        latent: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Main resizing function."""
        divisor = int(divisible_by)
        result_image = None
        result_latent = None
        
        if image is not None:
            current_height, current_width = image.shape[-2:]
            target_width, target_height = self.calculate_target_dimensions(
                current_width, current_height, divisor, scaling_mode
            )
            
            needs_unsqueeze = len(image.shape) == 3
            if needs_unsqueeze:
                image = image.unsqueeze(0)
            
            # Special handling for lanczos interpolation
            if interpolation == 'lanczos':
                result_image = F.interpolate(
                    image,
                    size=(target_height, target_width),
                    mode='bicubic',  # Torch doesn't have native Lanczos, so we use bicubic as closest approximation
                    align_corners=False
                )
            else:
                result_image = F.interpolate(
                    image,
                    size=(target_height, target_width),
                    mode=interpolation,
                    align_corners=False if interpolation in ["linear", "bilinear", "bicubic"] else None
                )
            
            if needs_unsqueeze:
                result_image = result_image.squeeze(0)
                
            result_image = self.apply_crop_or_pad(
                result_image, target_width, target_height, crop_or_pad_mode
            )
            
        if latent is not None:
            latent_tensor = latent["samples"]
            current_height, current_width = latent_tensor.shape[-2:]
            
            target_width, target_height = self.calculate_target_dimensions(
                current_width * 8, current_height * 8, divisor, scaling_mode
            )
            target_width = target_width // 8
            target_height = target_height // 8
            
            # Special handling for lanczos interpolation in latents
            if interpolation == 'lanczos':
                resized_latent = F.interpolate(
                    latent_tensor,
                    size=(target_height, target_width),
                    mode='bicubic',  # Torch doesn't have native Lanczos, so we use bicubic as closest approximation
                    align_corners=False
                )
            else:
                resized_latent = F.interpolate(
                    latent_tensor,
                    size=(target_height, target_width),
                    mode=interpolation,
                    align_corners=False if interpolation in ["linear", "bilinear", "bicubic"] else None
                )
            
            resized_latent = self.apply_crop_or_pad(
                resized_latent, target_width, target_height, crop_or_pad_mode
            )
            
            result_latent = {"samples": resized_latent}
        
        return (result_image, result_latent)