"""
File: homage_tools/nodes/ht_image_adjuster_node.py
Version: 1.0.0
Description: Comprehensive image adjustment node with Photoshop-like controls and BHWC tensor handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import math

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.0.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_format(tensor: torch.Tensor, context: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Verify and normalize tensor to BHWC format."""
    print(f"\nVerifying {context} tensor:")
    print(f"- Input shape: {tensor.shape}")
    
    # Handle different dimension formats
    if len(tensor.shape) == 3:  # HWC format
        tensor = tensor.unsqueeze(0)
        print("- Added batch dimension (HWC → BHWC)")
        
    # Ensure BHWC format
    if len(tensor.shape) == 4:
        if tensor.shape[-1] not in [1, 3, 4]:  # Not BHWC
            # Might be BCHW format
            if tensor.shape[1] in [1, 3, 4]:
                tensor = tensor.permute(0, 2, 3, 1)
                print("- Converted from BCHW to BHWC format")
            else:
                print(f"- WARNING: Unusual tensor shape: {tensor.shape}")
    else:
        print(f"- ERROR: Unsupported tensor format: {tensor.shape}")
        raise ValueError(f"Unsupported tensor format: {tensor.shape}")
    
    # Extract dimensions
    b, h, w, c = tensor.shape
    print(f"- Normalized shape: {tensor.shape} (BHWC)")
    print(f"- Value range: min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")
    
    return tensor, {
        "batch_size": b,
        "height": h,
        "width": w,
        "channels": c
    }

def apply_to_batch(func, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Apply a function to each item in a batch."""
    result = []
    batch_size = tensor.shape[0]
    
    for i in range(batch_size):
        # Process single image
        processed = func(tensor[i:i+1], *args, **kwargs)
        result.append(processed)
    
    # Combine results
    return torch.cat(result, dim=0)

def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to HSV color space."""
    # Ensure input is BHWC format and in range [0, 1]
    rgb = torch.clamp(rgb, 0, 1)
    
    # Extract RGB channels
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # Compute max and min values
    max_rgb, _ = torch.max(rgb, dim=-1)
    min_rgb, _ = torch.min(rgb, dim=-1)
    diff = max_rgb - min_rgb
    
    # Compute hue
    h = torch.zeros_like(max_rgb)
    
    # Red is max
    mask = (max_rgb == r) & (diff != 0)
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    
    # Green is max
    mask = (max_rgb == g) & (diff != 0)
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120)
    
    # Blue is max
    mask = (max_rgb == b) & (diff != 0)
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240)
    
    # Normalize hue to [0, 1]
    h = h / 360.0
    
    # Compute saturation
    s = torch.zeros_like(max_rgb)
    mask = max_rgb != 0
    s[mask] = diff[mask] / max_rgb[mask]
    
    # Compute value
    v = max_rgb
    
    # Stack the channels
    hsv = torch.stack([h, s, v], dim=-1)
    
    return hsv

def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV to RGB color space."""
    # Extract HSV channels
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Convert hue to range [0, 6)
    h = h * 6.0
    
    # Compute helper values
    i = torch.floor(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    # Initialize RGB with zeros
    rgb = torch.zeros_like(hsv)
    
    # Case 0: h in [0, 1)
    mask = (i % 6 == 0)
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = t[mask]
    rgb[mask, 2] = p[mask]
    
    # Case 1: h in [1, 2)
    mask = (i % 6 == 1)
    rgb[mask, 0] = q[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = p[mask]
    
    # Case 2: h in [2, 3)
    mask = (i % 6 == 2)
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = v[mask]
    rgb[mask, 2] = t[mask]
    
    # Case 3: h in [3, 4)
    mask = (i % 6 == 3)
    rgb[mask, 0] = p[mask]
    rgb[mask, 1] = q[mask]
    rgb[mask, 2] = v[mask]
    
    # Case 4: h in [4, 5)
    mask = (i % 6 == 4)
    rgb[mask, 0] = t[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = v[mask]
    
    # Case 5: h in [5, 6)
    mask = (i % 6 == 5)
    rgb[mask, 0] = v[mask]
    rgb[mask, 1] = p[mask]
    rgb[mask, 2] = q[mask]
    
    return rgb

#------------------------------------------------------------------------------
# Section 3: Image Adjustment Implementations
#------------------------------------------------------------------------------
def adjust_brightness_contrast(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image: Input image tensor (BHWC format)
        brightness: Brightness adjustment factor (-1.0 to 1.0)
        contrast: Contrast adjustment factor (-1.0 to 1.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Adjust brightness
    if brightness != 0:
        if brightness > 0:
            image = image * (1 - brightness) + brightness
        else:
            image = image * (1 + brightness)
    
    # Adjust contrast
    if contrast != 0:
        # Calculate mean
        mean = torch.mean(image, dim=[1, 2], keepdim=True)
        
        if contrast > 0:
            # Increase contrast
            factor = 1 + contrast
            image = (image - mean) * factor + mean
        else:
            # Decrease contrast
            factor = 1 / (1 - contrast)
            image = (image - mean) * factor + mean
    
    # Clamp values to valid range
    image = torch.clamp(image, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        image = torch.cat([image, alpha], dim=-1)
    
    return image

def adjust_gamma(
    image: torch.Tensor,
    gamma: float,
    device: torch.device
) -> torch.Tensor:
    """
    Apply gamma correction to image.
    
    Args:
        image: Input image tensor (BHWC format)
        gamma: Gamma value (0.1 to 5.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Gamma-corrected image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Apply gamma correction
    image = torch.pow(image, 1.0 / gamma)
    
    # Clamp values to valid range
    image = torch.clamp(image, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        image = torch.cat([image, alpha], dim=-1)
    
    return image

def adjust_hue_saturation(
    image: torch.Tensor,
    hue: float,
    saturation: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust hue and saturation of an image.
    
    Args:
        image: Input image tensor (BHWC format)
        hue: Hue adjustment (-1.0 to 1.0)
        saturation: Saturation adjustment (-1.0 to 1.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv(image)
    
    # Adjust hue (cyclic, so add and modulo)
    if hue != 0:
        hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
    
    # Adjust saturation
    if saturation != 0:
        if saturation > 0:
            hsv[..., 1] = hsv[..., 1] * (1 - saturation) + saturation
        else:
            hsv[..., 1] = hsv[..., 1] * (1 + saturation)
    
    # Clamp saturation to valid range
    hsv[..., 1] = torch.clamp(hsv[..., 1], 0, 1)
    
    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)
    
    # Clamp values to valid range
    rgb = torch.clamp(rgb, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        rgb = torch.cat([rgb, alpha], dim=-1)
    
    return rgb

def apply_sharpness(
    image: torch.Tensor,
    sharpness: float,
    device: torch.device
) -> torch.Tensor:
    """
    Apply sharpening filter to image.
    
    Args:
        image: Input image tensor (BHWC format)
        sharpness: Sharpness amount (0.0 to 5.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Sharpened image
    """
    # Move to device
    image = image.to(device)
    
    # Skip if no sharpening
    if sharpness <= 0:
        return image
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Convert to BCHW for convolution
    x = image.permute(0, 3, 1, 2)
    
    # Create sharpening kernel
    kernel = torch.tensor([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 9.0
    
    # Apply individually to each channel
    channels = []
    for c in range(x.shape[1]):
        channel = x[:, c:c+1]
        sharpened = F.conv2d(
            channel,
            kernel,
            padding=1
        )
        channels.append(sharpened)
    
    # Combine channels
    sharpened = torch.cat(channels, dim=1)
    
    # Convert back to BHWC
    sharpened = sharpened.permute(0, 2, 3, 1)
    
    # Blend with original based on sharpness amount
    result = image * (1 - sharpness) + sharpened * sharpness
    
    # Clamp values to valid range
    result = torch.clamp(result, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        result = torch.cat([result, alpha], dim=-1)
    
    return result

def adjust_exposure(
    image: torch.Tensor,
    exposure: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust exposure of an image.
    
    Args:
        image: Input image tensor (BHWC format)
        exposure: Exposure adjustment (-2.0 to 2.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Calculate exposure factor
    factor = 2.0 ** exposure
    
    # Apply exposure adjustment
    adjusted = image * factor
    
    # Clamp values to valid range
    adjusted = torch.clamp(adjusted, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        adjusted = torch.cat([adjusted, alpha], dim=-1)
    
    return adjusted

def adjust_shadows_highlights(
    image: torch.Tensor,
    shadows: float,
    highlights: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust shadows and highlights of an image.
    
    Args:
        image: Input image tensor (BHWC format)
        shadows: Shadow adjustment (-1.0 to 1.0)
        highlights: Highlight adjustment (-1.0 to 1.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Calculate luminance
    luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    
    # Shadow mask (low luminance areas)
    shadow_mask = torch.pow(1.0 - luminance, 2).unsqueeze(-1)
    
    # Highlight mask (high luminance areas)
    highlight_mask = torch.pow(luminance, 2).unsqueeze(-1)
    
    # Apply shadow adjustment
    if shadows != 0:
        if shadows > 0:
            # Lighten shadows
            shadow_adjust = image * (1.0 - shadow_mask) + torch.pow(image, 0.5) * shadow_mask
            image = image * (1.0 - shadows) + shadow_adjust * shadows
        else:
            # Darken shadows
            shadow_adjust = image * shadow_mask
            image = image * (1.0 + shadows) - shadow_adjust * (-shadows)
    
    # Apply highlight adjustment
    if highlights != 0:
        if highlights > 0:
            # Darken highlights
            highlight_adjust = image * highlight_mask
            image = image * (1.0 - highlights) + highlight_adjust * (1.0 - highlights)
        else:
            # Lighten highlights
            highlight_adjust = torch.pow(image, 0.5) * highlight_mask + image * (1.0 - highlight_mask)
            image = image * (1.0 + highlights) - highlight_adjust * (-highlights)
    
    # Clamp values to valid range
    image = torch.clamp(image, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        image = torch.cat([image, alpha], dim=-1)
    
    return image

def adjust_vibrance(
    image: torch.Tensor,
    vibrance: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust vibrance of an image (saturation that preserves skin tones).
    
    Args:
        image: Input image tensor (BHWC format)
        vibrance: Vibrance adjustment (-1.0 to 1.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Skip if no vibrance adjustment
    if vibrance == 0:
        return image
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv(image)
    
    # Calculate average saturation
    avg_saturation = torch.mean(hsv[..., 1], dim=[1, 2], keepdim=True)
    
    # Adjust saturation based on current saturation (less saturated colors get more adjustment)
    saturation_mask = (1.0 - hsv[..., 1]).unsqueeze(-1)
    saturation_adjustment = vibrance * saturation_mask
    
    # Apply adjustment
    hsv[..., 1] = torch.clamp(hsv[..., 1] + saturation_adjustment[..., 0], 0, 1)
    
    # Convert back to RGB
    rgb = hsv_to_rgb(hsv)
    
    # Clamp values to valid range
    rgb = torch.clamp(rgb, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        rgb = torch.cat([rgb, alpha], dim=-1)
    
    return rgb

def adjust_color_balance(
    image: torch.Tensor,
    red: float,
    green: float,
    blue: float,
    device: torch.device
) -> torch.Tensor:
    """
    Adjust color balance of an image.
    
    Args:
        image: Input image tensor (BHWC format)
        red: Red channel adjustment (-1.0 to 1.0)
        green: Green channel adjustment (-1.0 to 1.0)
        blue: Blue channel adjustment (-1.0 to 1.0)
        device: Computation device
        
    Returns:
        torch.Tensor: Adjusted image
    """
    # Move to device
    image = image.to(device)
    
    # Handle alpha channel if present
    has_alpha = image.shape[-1] == 4
    alpha = None
    if has_alpha:
        alpha = image[..., 3:4]
        image = image[..., :3]
    
    # Adjust each channel
    adjusted = torch.clone(image)
    
    # Red channel
    if red != 0:
        if red > 0:
            adjusted[..., 0] = adjusted[..., 0] * (1 - red) + red
        else:
            adjusted[..., 0] = adjusted[..., 0] * (1 + red)
    
    # Green channel
    if green != 0:
        if green > 0:
            adjusted[..., 1] = adjusted[..., 1] * (1 - green) + green
        else:
            adjusted[..., 1] = adjusted[..., 1] * (1 + green)
    
    # Blue channel
    if blue != 0:
        if blue > 0:
            adjusted[..., 2] = adjusted[..., 2] * (1 - blue) + blue
        else:
            adjusted[..., 2] = adjusted[..., 2] * (1 + blue)
    
    # Clamp values to valid range
    adjusted = torch.clamp(adjusted, 0, 1)
    
    # Reattach alpha if needed
    if has_alpha:
        adjusted = torch.cat([adjusted, alpha], dim=-1)
    
    return adjusted

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTImageAdjusterNode:
    """
    Comprehensive image adjustment node with Photoshop-like controls.
    Supports batch processing and BHWC tensor format with proper channel handling.
    """
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "adjust_image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("adjusted_image",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                
                "use_cpu": ("BOOLEAN", {
                    "default": False,
                    "description": "Use CPU instead of CUDA for processing"
                }),
                
                # Exposure and Levels
                "exposure": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                
                # Color adjustments
                "hue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "saturation": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "vibrance": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                
                # Detail adjustments
                "sharpness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                
                # Shadow/Highlight
                "shadows": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "highlights": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                })
            },
            "optional": {
                # Color balance
                "red": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "green": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "blue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.05
                })
            }
        }

#------------------------------------------------------------------------------
# Section 5: Processing Logic
#------------------------------------------------------------------------------
    def adjust_image(
        self,
        image: torch.Tensor,
        use_cpu: bool,
        exposure: float,
        brightness: float,
        contrast: float,
        gamma: float,
        hue: float,
        saturation: float,
        vibrance: float,
        sharpness: float,
        shadows: float,
        highlights: float,
        red: float = 0.0,
        green: float = 0.0,
        blue: float = 0.0
    ) -> Tuple[torch.Tensor]:
        """
        Apply comprehensive image adjustments with proper BHWC handling.
        
        Args:
            image: Input image tensor
            use_cpu: Whether to use CPU instead of CUDA
            exposure: Exposure adjustment
            brightness: Brightness adjustment
            contrast: Contrast adjustment
            gamma: Gamma correction
            hue: Hue adjustment
            saturation: Saturation adjustment
            vibrance: Vibrance adjustment
            sharpness: Sharpness enhancement
            shadows: Shadow adjustment
            highlights: Highlight adjustment
            red: Red channel adjustment
            green: Green channel adjustment
            blue: Blue channel adjustment
            
        Returns:
            Tuple[torch.Tensor]: Adjusted image
        """
        try:
            print(f"\nHTImageAdjusterNode v{VERSION} - Processing")
            
            # Verify and normalize input format
            image, dims = verify_tensor_format(image, "Input")
            
            # Set processing device
            device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Processing device: {device}")
            
            # Apply adjustments in a logical order (similar to Photoshop)
            result = image
            
            # 1. Exposure (must come first as it affects overall range)
            if exposure != 0:
                print(f"Applying exposure adjustment: {exposure}")
                result = adjust_exposure(result, exposure, device)
            
            # 2. Color balance
            if red != 0 or green != 0 or blue != 0:
                print(f"Applying color balance: R={red}, G={green}, B={blue}")
                result = adjust_color_balance(result, red, green, blue, device)
            
            # 3. Brightness and contrast
            if brightness != 0 or contrast != 0:
                print(f"Applying brightness/contrast: brightness={brightness}, contrast={contrast}")
                result = adjust_brightness_contrast(result, brightness, contrast, device)
            
            # 4. Gamma correction
            if gamma != 1.0:
                print(f"Applying gamma correction: {gamma}")
                result = adjust_gamma(result, gamma, device)
            
            # 5. Shadows and highlights
            if shadows != 0 or highlights != 0:
                print(f"Adjusting shadows/highlights: shadows={shadows}, highlights={highlights}")
                result = adjust_shadows_highlights(result, shadows, highlights, device)
            
            # 6. Saturation adjustments
            if hue != 0 or saturation != 0:
                print(f"Adjusting hue/saturation: hue={hue}, saturation={saturation}")
                result = adjust_hue_saturation(result, hue, saturation, device)
            
            # 7. Vibrance (sophisticated saturation)
            if vibrance != 0:
                print(f"Applying vibrance adjustment: {vibrance}")
                result = adjust_vibrance(result, vibrance, device)
            
            # 8. Sharpness (comes last to operate on adjusted colors)
            if sharpness > 0:
                print(f"Applying sharpness filter: {sharpness}")
                result = apply_sharpness(result, sharpness, device)
            
            # Final verification and cleanup
            print("\nFinal output:")
            print(f"- Shape: {result.shape}")
            print(f"- Value range: min={result.min().item():.3f}, max={result.max().item():.3f}")
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in image adjustment: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return original image on error
            return (image,)