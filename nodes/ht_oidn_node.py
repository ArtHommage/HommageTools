"""
File: custom_nodes/HommageTools/nodes/ht_oidn_node.py
Version: 1.0.5
Description: Enhanced OIDN denoising node with progress reporting
"""

import os
import logging
import comfy.model_management as mm
import torch
import numpy as np
import oidn
import time

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
                    "max": 2.0,
                    "step": 0.05,
                    "description": "Denoising strength"
                }),
                "passes": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
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
                print("\nInitializing OIDN...")
                d = oidn.NewDevice()
                oidn.CommitDevice(d)

            # Empty cache
            mm.soft_empty_cache()
            
            # Print processing parameters
            print(f"\nOIDN Processing Parameters:")
            print(f"Device: {device}")
            print(f"Strength: {strength:.2f}")
            print(f"Passes: {passes}")
            print(f"Aggressiveness: {aggressive:.2f}")
            
            # Handle input dimensions
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                
            # Get dimensions (BHWC format)
            batch_size, height, width, channels = image.shape
            print(f"\nInput dimensions: {width}x{height}, {channels} channels")
            print(f"Processing {batch_size} images...")

            # Process each image in batch
            results = []
            total_passes = batch_size * passes
            current_pass = 0
            start_time = time.time()
            
            for idx in range(batch_size):
                img = image[idx]
                img_np = img.cpu().numpy().astype(np.float32)
                
                # Apply multiple passes
                for pass_num in range(passes):
                    current_pass += 1
                    elapsed = time.time() - start_time
                    progress = (current_pass / total_passes) * 100
                    
                    print(f"\rProgress: {progress:.1f}% | "
                          f"Image {idx + 1}/{batch_size}, "
                          f"Pass {pass_num + 1}/{passes} | "
                          f"Elapsed: {elapsed:.1f}s", end="")
                    
                    # Create and configure filter
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
                        width, height
                    )
                    oidn.SetSharedFilterImage(
                        filter, "output", result,
                        oidn.FORMAT_FLOAT3, 
                        width, height
                    )
                    
                    # Apply filter
                    oidn.CommitFilter(filter)
                    oidn.ExecuteFilter(filter)
                    oidn.ReleaseFilter(filter)
                    
                    # Apply strength
                    if strength != 1.0:
                        if strength > 1.0:
                            result = result + (result - img_np) * (strength - 1.0)
                        else:
                            result = result * strength + img_np * (1.0 - strength)
                            
                    # Apply aggressiveness
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
            
            # Final processing stats
            total_time = time.time() - start_time
            print(f"\n\nProcessing complete!")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average time per pass: {total_time/total_passes:.1f}s")
            
            # Stack results
            result = torch.stack(results) if len(results) > 1 else results[0]
            
            return (result,)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return (image,)