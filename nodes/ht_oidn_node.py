"""
File: custom_nodes/HommageTools/nodes/ht_oidn_node.py
Version: 1.0.4
Description: Enhanced OIDN denoising node with maximum strength options
"""

import os
import logging
import comfy.model_management as mm
import torch
import numpy as np
import oidn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HommageTools.OIDN')

# Global OIDN instance
d = None

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
                    "max": 2.0,  # Increased max strength
                    "step": 0.05,
                    "description": "Denoising strength"
                }),
                "passes": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,  # Increased max passes
                    "step": 1,
                    "description": "Number of denoising passes"
                }),
                "aggressive": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Denoising aggressiveness"
                })
            }
        }

    def denoise_image(self, image, device="cpu", strength=1.0, passes=1, aggressive=1.0):
        """Process image batch with OIDN."""
        global d
        
        try:
            # Initialize global denoiser if needed
            if d is None:
                d = oidn.NewDevice()
                oidn.CommitDevice(d)

            # Empty cache
            mm.soft_empty_cache()
            
            # Handle input dimensions
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            # Process each image in batch
            results = []
            for idx in range(len(image)):
                img = image[idx]
                img_np = img.cpu().numpy().astype(np.float32)
                
                # Apply multiple passes if requested
                for _ in range(passes):
                    # Create filter for this pass
                    filter = oidn.NewFilter(d, "RT")
                    
                    # Normalize if needed
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                    
                    # Process image
                    result = np.zeros_like(img_np)
                    
                    # Set up filter
                    oidn.SetSharedFilterImage(
                        filter, "color", img_np, 
                        oidn.FORMAT_FLOAT3, 
                        img_np.shape[1], img_np.shape[0]
                    )
                    oidn.SetSharedFilterImage(
                        filter, "output", result,
                        oidn.FORMAT_FLOAT3, 
                        img_np.shape[1], img_np.shape[0]
                    )
                    
                    # Apply filter
                    oidn.CommitFilter(filter)
                    oidn.ExecuteFilter(filter)
                    oidn.ReleaseFilter(filter)
                    
                    # Apply strength (now can go beyond 1.0)
                    if strength != 1.0:
                        if strength > 1.0:
                            # Enhance the denoising effect
                            result = result + (result - img_np) * (strength - 1.0)
                        else:
                            # Regular blend
                            result = result * strength + img_np * (1.0 - strength)
                            
                    # Apply aggressiveness by enhancing the difference
                    if aggressive > 1.0:
                        diff = result - img_np
                        result = img_np + diff * aggressive
                    
                    # Update for next pass
                    img_np = result
                
                # Convert back to original range
                if image.max() > 1.0:
                    result = np.clip(result * 255.0, 0, 255)
                
                # Convert back to tensor
                results.append(torch.from_numpy(result.astype(img_np.dtype)))
            
            # Stack results
            result = torch.stack(results) if len(results) > 1 else results[0]
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return (image,)