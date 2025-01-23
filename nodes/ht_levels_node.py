"""
File: homage_tools/nodes/ht_levels_node.py
Version: 1.1.1
Description: Node for levels correction using reference images with BHWC format handling
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class HTLevelsNode:
    """Provides image levels correction using reference images."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_levels"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "source_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "method": (["luminance_curve", "histogram_match"], {
                    "default": "luminance_curve"
                }),
                "adjustment_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            }
        }

    def _compute_histogram(self, image: torch.Tensor) -> torch.Tensor:
        """Compute histogram for each channel in BHWC format."""
        # image shape: [B,H,W,C]
        hist = torch.zeros(image.shape[-1], 256, device=image.device)
        for i in range(image.shape[-1]):
            channel = image[..., i]
            bins = torch.histc(channel, bins=256, min=0, max=1)
            hist[i] = bins
        return hist

    def _apply_levels(
        self,
        source: torch.Tensor,  # [B,H,W,C]
        reference: torch.Tensor,  # [B,H,W,C]
        strength: float = 1.0
    ) -> torch.Tensor:
        """Apply levels adjustment using BHWC tensors."""
        # Compute histograms (source shape: [B,H,W,C])
        source_hist = self._compute_histogram(source[0])  # Use first batch item
        ref_hist = self._compute_histogram(reference[0])
        
        # Compute CDFs
        source_cdf = torch.cumsum(source_hist, dim=1)
        ref_cdf = torch.cumsum(ref_hist, dim=1)
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[:, -1:].clamp(min=1e-5)
        ref_cdf = ref_cdf / ref_cdf[:, -1:].clamp(min=1e-5)
        
        # Create lookup indices
        indices = torch.arange(256, device=source.device).float() / 255.0
        indices = indices.view(1, -1).expand(source.shape[-1], -1)
        
        # Find matching values
        matched = torch.zeros_like(indices)
        for i in range(source.shape[-1]):
            source_vals = source_cdf[i]
            ref_vals = ref_cdf[i]
            
            for j in range(256):
                val = indices[i,j]
                diff = torch.abs(ref_vals - val)
                idx = torch.argmin(diff)
                matched[i,j] = idx / 255.0
        
        # Apply strength
        if strength != 1.0:
            matched = indices + (matched - indices) * strength
        
        # Apply correction (maintaining BHWC format)
        result = torch.zeros_like(source)
        for b in range(source.shape[0]):  # Process each batch
            for i in range(source.shape[-1]):  # Process each channel
                channel = source[b, ..., i]
                scaled = (channel * 255).long().clamp(0, 255)
                correction = matched[i][scaled]
                result[b, ..., i] = correction
            
        return result

    def process_levels(
        self,
        source_image: torch.Tensor,  # [B,H,W,C] or [H,W,C]
        reference_image: torch.Tensor,  # [B,H,W,C] or [H,W,C]
        method: str,
        adjustment_strength: float
    ) -> Tuple[torch.Tensor]:
        """Process image levels using reference in BHWC format."""
        try:
            # Handle batch dimension
            if len(source_image.shape) == 3:
                source_image = source_image.unsqueeze(0)
            if len(reference_image.shape) == 3:
                reference_image = reference_image.unsqueeze(0)
                
            # Apply levels correction
            output = self._apply_levels(
                source_image,
                reference_image,
                adjustment_strength
            )
            
            # Remove batch dim if input wasn't batched
            if len(source_image.shape) == 3:
                output = output.squeeze(0)
                
            return (output,)
            
        except Exception as e:
            print(f"Error in HTLevelsNode: {str(e)}")
            return (source_image,)