"""
File: custom_nodes/HommageTools/nodes/ht_oidn_node.py
Version: 1.0.3
Description: Enhanced OIDN denoising node with GPU support and adjustable parameters
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

# Global OIDN instance (matches original pattern)
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
                    "max": 1.0,
                    "step": 0.05,
                    "description": "Denoising strength"
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

    def denoise_image(self, image, device="cpu", strength=1.0, passes=1):
        """Process image batch with OIDN."""
        global d
        
        try:
            # Initialize global denoiser if needed
            if d is None:
                d = oidn.Device()
                d.commit()

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
                    filter = d.Filter(filter_type="RT")
                    
                    # Normalize if needed
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
                    
                    # Process image
                    result = np.zeros_like(img_np)
                    filter.set_image("color", img_np)
                    filter.set_image("output", result)
                    filter.commit()
                    filter.execute()
                    
                    # Apply strength
                    if strength < 1.0:
                        result = result * strength + img_np * (1.0 - strength)
                    
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