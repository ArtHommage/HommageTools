"""
HommageTools Resize Node
Version: 1.0.0
Description: A node that handles intelligent resizing of images and latents to dimensions
divisible by 8 or 64, with multiple scaling and cropping options.

This node provides:
- Support for both image and latent inputs
- Multiple interpolation methods
- Smart scaling to maintain aspect ratios
- Various cropping and padding options
- Proper handling of both image and latent space dimensions
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

class HTResizeNode:
    """
    A ComfyUI node that provides advanced resizing capabilities for both images and latents.
    Ensures output dimensions are divisible by 8 or 64 while maintaining aspect ratios
    and providing various scaling and cropping options.
    """
    
    # Node identification
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("resized_image", "resized_latent")
    FUNCTION = "resize_media"

    # Available options for the node
    INTERPOLATION_MODES = [
        "nearest", "linear", "bilinear", "bicubic", 
        "trilinear", "area", "nearest-exact"
    ]
    SCALING_MODES = ["short_side", "long_side"]
    CROP_MODES = ["center", "top", "bottom", "left", "right"]
    DIVISIBILITY_OPTIONS = ["8", "64"]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Define the input types and default values for the node.
        
        Returns:
            dict: Dictionary containing the input specifications
        """
        return {
            "required": {
                "divisible_by": (cls.DIVISIBILITY_OPTIONS, {
                    "default": "8",
                    "description": "Output dimensions will be divisible by this number"
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "bicubic",
                    "description": "Method used for interpolation during resizing"
                }),
                "scaling_mode": (cls.SCALING_MODES, {
                    "default": "short_side",
                    "description": "Whether to scale based on the short or long side"
                }),
                "crop_or_pad_mode": (cls.CROP_MODES, {
                    "default": "center",
                    "description": "How to handle excess space or cropping"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            }
        }

    def get_nearest_divisible_size(self, size: int, divisor: int) -> int:
        """
        Calculate the nearest size that is divisible by the specified divisor.
        
        Args:
            size (int): Original size
            divisor (int): Number to be divisible by
            
        Returns:
            int: Nearest size divisible by divisor
        """
        return ((size + divisor - 1) // divisor) * divisor

    def calculate_target_dimensions(
        self, 
        current_width: int, 
        current_height: int, 
        divisor: int,
        scaling_mode: str
    ) -> Tuple[int, int]:
        """
        Calculate target dimensions that maintain aspect ratio and divisibility requirements.
        
        Args:
            current_width (int): Original width
            current_height (int): Original height
            divisor (int): Number to be divisible by (8 or 64)
            scaling_mode (str): Either 'short_side' or 'long_side'
            
        Returns:
            Tuple[int, int]: New width and height
        """
        # Calculate aspect ratio
        aspect_ratio = current_width / current_height

        # Determine which dimension to use as reference based on scaling mode
        if scaling_mode == "short_side":
            if current_height < current_width:
                # Height is shorter, use it as reference
                new_height = self.get_nearest_divisible_size(current_height, divisor)
                new_width = self.get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                # Width is shorter, use it as reference
                new_width = self.get_nearest_divisible_size(current_width, divisor)
                new_height = self.get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)
        else:  # long_side
            if current_height > current_width:
                # Height is longer, use it as reference
                new_height = self.get_nearest_divisible_size(current_height, divisor)
                new_width = self.get_nearest_divisible_size(int(new_height * aspect_ratio), divisor)
            else:
                # Width is longer, use it as reference
                new_width = self.get_nearest_divisible_size(current_width, divisor)
                new_height = self.get_nearest_divisible_size(int(new_width / aspect_ratio), divisor)

        return new_width, new_height

    def apply_crop_or_pad(
        self,
        tensor: torch.Tensor,
        target_width: int,
        target_height: int,
        mode: str,
        is_latent: bool = False
    ) -> torch.Tensor:
        """
        Apply cropping or padding based on the specified mode.
        
        Args:
            tensor: Input tensor (image or latent)
            target_width: Desired width
            target_height: Desired height
            mode: Cropping/padding mode (center, top, bottom, left, right)
            is_latent: Whether the tensor is a latent
            
        Returns:
            torch.Tensor: Processed tensor
        """
        current_height = tensor.shape[-2]
        current_width = tensor.shape[-1]
        
        # If dimensions already match, return unchanged
        if current_height == target_height and current_width == target_width:
            return tensor

        # Calculate padding/cropping amounts based on mode
        if mode == "center":
            pad_left = max(0, (target_width - current_width) // 2)
            pad_right = max(0, target_width - current_width - pad_left)
            pad_top = max(0, (target_height - current_height) // 2)
            pad_bottom = max(0, target_height - current_height - pad_top)
            
            crop_left = max(0, (current_width - target_width) // 2)
            crop_right = max(0, current_width - target_width - crop_left)
            crop_top = max(0, (current_height - target_height) // 2)
            crop_bottom = max(0, current_height - target_height - crop_top)
            
        elif mode in ["top", "bottom"]:
            pad_left = max(0, (target_width - current_width) // 2)
            pad_right = max(0, target_width - current_width - pad_left)
            
            if mode == "top":
                pad_top, pad_bottom = 0, max(0, target_height - current_height)
                crop_top, crop_bottom = max(0, current_height - target_height), 0
            else:
                pad_top, pad_bottom = max(0, target_height - current_height), 0
                crop_top, crop_bottom = 0, max(0, current_height - target_height)
                
        else:  # left or right
            pad_top = max(0, (target_height - current_height) // 2)
            pad_bottom = max(0, target_height - current_height - pad_top)
            
            if mode == "left":
                pad_left, pad_right = 0, max(0, target_width - current_width)
                crop_left, crop_right = max(0, current_width - target_width), 0
            else:
                pad_left, pad_right = max(0, target_width - current_width), 0
                crop_left, crop_right = 0, max(0, current_width - target_width)

        # Apply cropping if needed
        if crop_left + crop_right + crop_top + crop_bottom > 0:
            tensor = tensor[..., 
                          crop_top:current_height-crop_bottom,
                          crop_left:current_width-crop_right]

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
        """
        Main processing function to resize either image or latent inputs.
        
        Args:
            divisible_by: String "8" or "64"
            interpolation: Interpolation mode for resizing
            scaling_mode: Whether to scale based on short or long side
            crop_or_pad_mode: How to handle excess space
            image: Optional image tensor
            latent: Optional latent dict
            
        Returns:
            Tuple containing processed image and latent (None for unused inputs)
        """
        try:
            divisor = int(divisible_by)
            result_image = None
            result_latent = None
            
            # Process image if provided
            if image is not None:
                current_height, current_width = image.shape[-2:]
                target_width, target_height = self.calculate_target_dimensions(
                    current_width, current_height, divisor, scaling_mode
                )
                
                # Handle single image vs batch
                needs_unsqueeze = len(image.shape) == 3
                if needs_unsqueeze:
                    image = image.unsqueeze(0)
                
                # Resize
                result_image = F.interpolate(
                    image,
                    size=(target_height, target_width),
                    mode=interpolation,
                    align_corners=False if interpolation in ["linear", "bilinear", "bicubic", "trilinear"] else None
                )
                
                # Remove batch dimension if it was added
                if needs_unsqueeze:
                    result_image = result_image.squeeze(0)
                    
                # Apply cropping/padding
                result_image = self.apply_crop_or_pad(
                    result_image, target_width, target_height, crop_or_pad_mode
                )
                
            # Process latent if provided
            if latent is not None:
                latent_tensor = latent["samples"]
                current_height, current_width = latent_tensor.shape[-2:]
                
                # Calculate target dimensions in image space
                target_width, target_height = self.calculate_target_dimensions(
                    current_width * 8, current_height * 8, divisor, scaling_mode
                )
                
                # Convert target dimensions to latent space
                target_width = target_width // 8
                target_height = target_height // 8
                
                # Resize latent
                resized_latent = F.interpolate(
                    latent_tensor,
                    size=(target_height, target_width),
                    mode=interpolation,
                    align_corners=False if interpolation in ["linear", "bilinear", "bicubic", "trilinear"] else None
                )
                
                # Apply cropping/padding
                resized_latent = self.apply_crop_or_pad(
                    resized_latent, target_width, target_height, crop_or_pad_mode, is_latent=True
                )
                
                result_latent = {"samples": resized_latent}
            
            return (result_image, result_latent)
            
        except Exception as e:
            print(f"Error in HTResizeNode: {str(e)}")
            return (image, latent)  # Return original inputs on error