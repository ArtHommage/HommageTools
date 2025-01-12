"""
File: homage_tools/nodes/ht_levels_node.py
Version: 1.1.0
Description: Node for levels correction using reference images with proper dimension handling

Sections:
1. Imports and Setup
2. Node Class Definition
3. Image Processing Methods
4. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Setup
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTLevelsNode:
    """
    Provides image levels correction using reference images.
    Maintains correct image dimensions throughout processing.
    """
    
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

#------------------------------------------------------------------------------
# Section 3: Image Processing Methods
#------------------------------------------------------------------------------
    def _compute_histogram(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute histogram for each channel.
        
        Args:
            image: Input tensor [C,H,W] scaled 0-1
            
        Returns:
            torch.Tensor: Histogram per channel [C,256]
        """
        hist = torch.zeros(image.shape[0], 256, device=image.device)
        for i in range(image.shape[0]):
            channel = image[i]
            bins = torch.histc(channel, bins=256, min=0, max=1)
            hist[i] = bins
        return hist

    def _compute_cdf(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative distribution function.
        
        Args:
            hist: Histogram tensor [C,256]
            
        Returns:
            torch.Tensor: CDF per channel [C,256]
        """
        cdf = torch.cumsum(hist, dim=1)
        # Normalize
        cdf = cdf / cdf[:, -1:].clamp(min=1e-5)
        return cdf

    def _apply_levels(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply levels adjustment using tensors.
        
        Args:
            source: Source image tensor [C,H,W]
            reference: Reference image tensor [C,H,W]
            strength: Adjustment strength
            
        Returns:
            torch.Tensor: Adjusted image tensor [C,H,W]
        """
        # Compute histograms
        source_hist = self._compute_histogram(source)
        ref_hist = self._compute_histogram(reference)
        
        # Compute CDFs
        source_cdf = self._compute_cdf(source_hist)
        ref_cdf = self._compute_cdf(ref_hist)
        
        # Create lookup indices
        indices = torch.arange(256, device=source.device).float() / 255.0
        indices = indices.view(1, -1).expand(source.shape[0], -1)
        
        # Find matching values
        matched = torch.zeros_like(indices)
        for i in range(source.shape[0]):
            source_vals = source_cdf[i]
            ref_vals = ref_cdf[i]
            
            # Find closest matching CDF values
            for j in range(256):
                # Get source CDF value
                val = indices[i,j]
                # Find closest value in reference CDF
                diff = torch.abs(ref_vals - val)
                idx = torch.argmin(diff)
                matched[i,j] = idx / 255.0
        
        # Apply strength
        if strength != 1.0:
            matched = indices + (matched - indices) * strength
        
        # Apply correction
        result = torch.zeros_like(source)
        for i in range(source.shape[0]):
            channel = source[i]
            # Scale to indices
            scaled = (channel * 255).long().clamp(0, 255)
            # Gather correction values
            correction = matched[i][scaled]
            result[i] = correction
            
        return result

#------------------------------------------------------------------------------
# Section 4: Main Processing Logic
#------------------------------------------------------------------------------
    def process_levels(
        self,
        source_image: torch.Tensor,
        reference_image: torch.Tensor,
        method: str,
        adjustment_strength: float
    ) -> Tuple[torch.Tensor]:
        """
        Process image levels using reference.
        
        Args:
            source_image: Source image tensor [B,C,H,W] or [C,H,W]
            reference_image: Reference image tensor [B,C,H,W] or [C,H,W]
            method: Processing method
            adjustment_strength: Adjustment strength
            
        Returns:
            Tuple[torch.Tensor]: Processed image tensor
        """
        try:
            # Handle batch dimension
            source_was_batched = len(source_image.shape) == 4
            ref_was_batched = len(reference_image.shape) == 4
            
            if not source_was_batched:
                source_image = source_image.unsqueeze(0)
            if not ref_was_batched:
                reference_image = reference_image.unsqueeze(0)
                
            # Process each image in batch
            results = []
            for i in range(source_image.shape[0]):
                src = source_image[i]
                ref = reference_image[min(i, reference_image.shape[0]-1)]
                
                # Apply levels correction
                processed = self._apply_levels(
                    src, ref, adjustment_strength
                )
                results.append(processed)
                
            # Combine results
            output = torch.stack(results, dim=0)
            
            # Remove batch dim if input wasn't batched
            if not source_was_batched:
                output = output.squeeze(0)
                
            return (output,)
            
        except Exception as e:
            print(f"Error in HTLevelsNode: {str(e)}")
            return (source_image,)