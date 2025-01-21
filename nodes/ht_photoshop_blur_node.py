"""
File: homage_tools/nodes/ht_photoshop_blur_node.py
Version: 1.0.0
Description: Comprehensive blur node emulating Photoshop blur filters

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Blur Implementation Classes
4. Node Class Definition
5. Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import math
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def create_gaussian_kernel(radius: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create 2D Gaussian kernel."""
    size = 2 * radius + 1
    x = torch.linspace(-radius, radius, size, device=device)
    y = torch.linspace(-radius, radius, size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma * sigma))
    return kernel / kernel.sum()

def create_motion_kernel(length: int, angle: float, device: torch.device) -> torch.Tensor:
    """Create motion blur kernel."""
    kernel = torch.zeros((length, length), device=device)
    center = length // 2
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    for i in range(length):
        offset = i - center
        x = round(offset * cos_a)
        y = round(offset * sin_a)
        if abs(x) < center and abs(y) < center:
            kernel[center + y, center + x] = 1.0
            
    return kernel / kernel.sum()

def create_radial_kernel(size: int, zoom: bool, device: torch.device) -> torch.Tensor:
    """Create radial/zoom blur kernel."""
    kernel = torch.zeros((size, size), device=device)
    center = size // 2
    for i in range(size):
        for j in range(size):
            y, x = i - center, j - center
            if zoom:
                angle = math.atan2(y, x)
                kernel[i, j] = angle / (2 * math.pi) + 0.5
            else:
                dist = math.sqrt(x*x + y*y)
                kernel[i, j] = dist / math.sqrt(2 * center*center)
    return kernel / kernel.sum()

#------------------------------------------------------------------------------
# Section 3: Blur Implementation Classes
#------------------------------------------------------------------------------
class BlurImplementations:
    """Container for different blur implementations."""
    
    @staticmethod
    def average_blur(
        image: torch.Tensor,
        radius: int,
        device: torch.device
    ) -> torch.Tensor:
        """Simple box averaging blur."""
        kernel_size = 2 * radius + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) 
        kernel = kernel / kernel.numel()
        
        # Apply separable convolution for efficiency
        x = F.conv2d(image, kernel, padding=radius)
        return F.conv2d(x, kernel.transpose(2, 3), padding=radius)

    @staticmethod
    def gaussian_blur(
        image: torch.Tensor,
        radius: int,
        sigma: float,
        device: torch.device
    ) -> torch.Tensor:
        """Gaussian blur implementation."""
        kernel = create_gaussian_kernel(radius, sigma, device)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        return F.conv2d(image, kernel, padding=radius)

    @staticmethod
    def motion_blur(
        image: torch.Tensor,
        length: int,
        angle: float,
        device: torch.device
    ) -> torch.Tensor:
        """Motion blur implementation."""
        kernel = create_motion_kernel(length, angle, device)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        padding = length // 2
        return F.conv2d(image, kernel, padding=padding)

    @staticmethod
    def radial_blur(
        image: torch.Tensor,
        radius: int,
        zoom: bool,
        device: torch.device
    ) -> torch.Tensor:
        """Radial/zoom blur implementation."""
        kernel = create_radial_kernel(2 * radius + 1, zoom, device)
        kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
        return F.conv2d(image, kernel, padding=radius)

    @staticmethod
    def lens_blur(
        image: torch.Tensor,
        radius: int,
        blade_count: int,
        rotation: float,
        device: torch.device
    ) -> torch.Tensor:
        """Lens blur (bokeh) implementation."""
        # Create polygon-shaped kernel for bokeh
        kernel_size = 2 * radius + 1
        kernel = torch.zeros((kernel_size, kernel_size), device=device)
        center = radius
        
        # Generate polygon points
        for i in range(blade_count):
            angle = rotation + (2 * math.pi * i / blade_count)
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            kernel[int(y), int(x)] = 1.0
            
        # Normalize and apply
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return F.conv2d(image, kernel, padding=radius)

    @staticmethod
    def smart_blur(
        image: torch.Tensor,
        radius: int,
        threshold: float,
        device: torch.device
    ) -> torch.Tensor:
        """Smart blur with edge preservation."""
        # Similar to surface blur but with additional edge detection
        kernel_size = 2 * radius + 1
        threshold = threshold / 255.0  # Normalize threshold
        
        # Create edge detection kernel
        edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], device=device).view(1, 1, 3, 3)
        
        # Detect edges
        edges = F.conv2d(image, edge_kernel, padding=1)
        edge_mask = (edges.abs() < threshold).float()
        
        # Apply gaussian blur
        blurred = BlurImplementations.gaussian_blur(
            image, radius, radius/2, device
        )
        
        # Blend based on edge mask
        return image * (1 - edge_mask) + blurred * edge_mask

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTPhotoshopBlurNode:
    """
    Comprehensive blur node emulating Photoshop's blur filters.
    Provides multiple blur algorithms with configurable parameters.
    """
    
    CATEGORY = "HommageTools/Filters"
    FUNCTION = "apply_blur"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blurred_image",)
    
    BLUR_TYPES = [
        "average",
        "gaussian",
        "motion",
        "radial",
        "zoom",
        "lens",
        "smart"
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_type": (cls.BLUR_TYPES, {
                    "default": "gaussian"
                }),
                "radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                })
            },
            "optional": {
                # Gaussian blur params
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                # Motion blur params
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0
                }),
                # Lens blur params
                "blade_count": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 12,
                    "step": 1
                }),
                "blade_rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0
                }),
                # Smart blur params
                "threshold": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 255.0,
                    "step": 0.1
                })
            }
        }

#------------------------------------------------------------------------------
# Section 5: Processing Logic
#------------------------------------------------------------------------------
    def apply_blur(
        self,
        image: torch.Tensor,
        blur_type: str,
        radius: int,
        sigma: float = 1.0,
        angle: float = 0.0,
        blade_count: int = 5,
        blade_rotation: float = 0.0,
        threshold: float = 15.0
    ) -> Tuple[torch.Tensor]:
        """
        Apply selected blur effect to image.
        
        Args:
            image: Input image tensor
            blur_type: Type of blur to apply
            radius: Blur radius
            sigma: Gaussian blur sigma
            angle: Motion blur angle
            blade_count: Lens blur blade count
            blade_rotation: Lens blur rotation
            threshold: Smart blur threshold
            
        Returns:
            Tuple[torch.Tensor]: Processed image
        """
        try:
            device = image.device
            
            # Handle input tensor dimensions
            if image.ndim == 3:
                image = image.unsqueeze(0)
                
            # Process each channel separately
            result = []
            for c in range(image.shape[1]):
                channel = image[:, c:c+1]
                
                # Apply selected blur
                if blur_type == "average":
                    blurred = BlurImplementations.average_blur(
                        channel, radius, device
                    )
                elif blur_type == "gaussian":
                    blurred = BlurImplementations.gaussian_blur(
                        channel, radius, sigma, device
                    )
                elif blur_type == "motion":
                    blurred = BlurImplementations.motion_blur(
                        channel, 2 * radius + 1, angle, device
                    )
                elif blur_type == "radial":
                    blurred = BlurImplementations.radial_blur(
                        channel, radius, False, device
                    )
                elif blur_type == "zoom":
                    blurred = BlurImplementations.radial_blur(
                        channel, radius, True, device
                    )
                elif blur_type == "lens":
                    blurred = BlurImplementations.lens_blur(
                        channel, radius, blade_count, 
                        math.radians(blade_rotation), device
                    )
                elif blur_type == "smart":
                    blurred = BlurImplementations.smart_blur(
                        channel, radius, threshold, device
                    )
                else:
                    raise ValueError(f"Unknown blur type: {blur_type}")
                    
                result.append(blurred)
                
            # Combine channels
            result = torch.cat(result, dim=1)
            
            # Restore original dimensions
            if image.shape[0] == 1:
                result = result.squeeze(0)
                
            return (result,)
            
        except Exception as e:
            logger.error(f"Error in blur processing: {str(e)}")
            return (image,)  # Return original image on error