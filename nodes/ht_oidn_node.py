"""
File: homage_tools/nodes/ht_oidn_node.py
Version: 1.1.1
Description: Enhanced OIDN denoising node with proper imports
"""

import os
import logging
import comfy.model_management as mm
import torch
import numpy as np
import oidn
import time
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger('HommageTools.OIDN')
d = None

class HTOIDNNode:
    CATEGORY = "HommageTools/Image"
    FUNCTION = "denoise_image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "passes": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1
                })
            }
        }

    def _process_single_image(
        self,
        image_np: np.ndarray,
        filter_obj: Any,
        width: int,
        height: int
    ) -> np.ndarray:
        result = np.zeros_like(image_np)
        oidn.SetSharedFilterImage(filter_obj, "color", image_np, oidn.FORMAT_FLOAT3, width, height)
        oidn.SetSharedFilterImage(filter_obj, "output", result, oidn.FORMAT_FLOAT3, width, height)
        oidn.CommitFilter(filter_obj)
        oidn.ExecuteFilter(filter_obj)
        return result

    def denoise_image(
        self,
        image: torch.Tensor,
        device: str = "cpu",
        strength: float = 1.0,
        passes: int = 1
    ) -> Tuple[torch.Tensor]:
        global d
        
        try:
            if d is None:
                print("\nInitializing OIDN...")
                d = oidn.NewDevice()
                oidn.CommitDevice(d)

            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                
            batch_size, height, width, channels = image.shape
            print(f"\nProcessing {batch_size} images ({width}x{height})")
            
            results = []
            for idx in range(batch_size):
                img = image[idx].cpu().numpy()
                
                for pass_num in range(passes):
                    filter_obj = oidn.NewFilter(d, "RT")
                    img = self._process_single_image(img, filter_obj, width, height)
                    
                    if strength != 1.0:
                        original = image[idx].cpu().numpy()
                        if strength > 1.0:
                            img = img + (img - original) * (strength - 1.0)
                        else:
                            img = img * strength + original * (1.0 - strength)
                            
                    oidn.ReleaseFilter(filter_obj)
                
                results.append(torch.from_numpy(img))
            
            result = torch.stack(results) if len(results) > 1 else results[0].unsqueeze(0)
            return (result,)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return (image,)