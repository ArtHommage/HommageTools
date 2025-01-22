"""
File: custom_nodes/HommageTools/nodes/ht_oidn_node.py
Version: 1.0.0
Description: Enhanced OIDN denoising node with GPU support and adjustable parameters

Sections:
1. Imports and Setup
2. OIDN Wrapper Class
3. ComfyUI Node Definition
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Setup
#------------------------------------------------------------------------------
import os
import logging
import comfy.model_management as mm
from comfy.utils import ProgressBar
import torch
import numpy as np
from PIL import Image
import oidn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HommageTools.OIDN')

#------------------------------------------------------------------------------
# Section 2: OIDN Wrapper Class
#------------------------------------------------------------------------------
class EnhancedOIDN:
    """Enhanced OIDN implementation with GPU support and adjustable parameters."""
    
    def __init__(self, device_type="cpu", strength=1.0, quality=1.0):
        """Initialize OIDN with specified device and parameters."""
        self.device = oidn.NewDevice(device_type)
        try:
            oidn.CommitDevice(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize {device_type} device: {e}")
            logger.info("Falling back to CPU")
            self.device = oidn.NewDevice("cpu")
            oidn.CommitDevice(self.device)
            
        self.filter = oidn.NewFilter(self.device, "RT")
        self.active = True
        self.strength = strength
        self.quality = quality
        
        # Set filter quality parameters
        try:
            if hasattr(oidn, 'SetFilter1b'):
                oidn.SetFilter1b(self.filter, "hdr", False)
            if hasattr(oidn, 'SetFilter1f'):
                oidn.SetFilter1f(self.filter, "quality", self.quality)
        except Exception as e:
            logger.warning(f"Failed to set advanced parameters: {e}")

    def process_image(self, img_array):
        """Process image with current settings."""
        if not self.active:
            self.__init__()
            
        # Convert to float32 and normalize
        self.img = img_array.astype(np.float32) / 255.0
        
        # Create output buffer
        self.result = np.zeros_like(self.img, dtype=np.float32)
        
        # Apply filter with current parameters
        try:
            oidn.SetSharedFilterImage(
                self.filter, 
                "color",
                self.img, 
                oidn.FORMAT_FLOAT3, 
                self.img.shape[1], 
                self.img.shape[0]
            )
            oidn.SetSharedFilterImage(
                self.filter,
                "output",
                self.result,
                oidn.FORMAT_FLOAT3,
                self.img.shape[1],
                self.img.shape[0]
            )
            
            oidn.CommitFilter(self.filter)
            oidn.ExecuteFilter(self.filter)
            
            # Blend based on strength
            if self.strength < 1.0:
                self.result = (self.result * self.strength + 
                             self.img * (1.0 - self.strength))
            
            # Convert back to uint8
            return np.clip(self.result * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return np.clip(self.img * 255, 0, 255).astype(np.uint8)
        
    def cleanup(self):
        """Release OIDN resources."""
        if self.active:
            oidn.ReleaseFilter(self.filter)
            oidn.ReleaseDevice(self.device)
            self.active = False

#------------------------------------------------------------------------------
# Section 3: ComfyUI Node Definition
#------------------------------------------------------------------------------
class HTOIDNNode:
    """ComfyUI node for enhanced OIDN denoising."""
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "denoise_image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["cpu", "gpu"], {
                    "default": "cpu",
                    "description": "Processing device"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Denoising strength"
                }),
                "quality": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "Processing quality"
                }),
                "passes": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "description": "Number of denoising passes"
                })
            }
        }
    
    def denoise_image(self, image, device, strength, quality, passes):
        """Process image batch with OIDN."""
        try:
            # Initialize denoiser
            denoiser = EnhancedOIDN(
                device_type=device,
                strength=strength,
                quality=quality
            )
            
            # Handle input dimensions
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            
            # Process each image in batch
            results = []
            for img in image:
                # Convert to numpy and process
                img_np = img.cpu().numpy()
                
                # Apply multiple passes if requested
                for _ in range(passes):
                    img_np = denoiser.process_image(img_np)
                
                # Convert back to tensor
                results.append(torch.from_numpy(img_np))
            
            # Stack results and cleanup
            denoiser.cleanup()
            result = torch.stack(results)
            
            # Remove batch dimension if input wasn't batched
            if len(image) == 1:
                result = result.squeeze(0)
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return (image,)