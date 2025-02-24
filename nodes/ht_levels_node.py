"""
File: homage_tools/nodes/ht_levels_node.py
Description: Node for levels correction using reference images with BHWC format handling
Version: 1.2.0
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants
#------------------------------------------------------------------------------
VERSION = "1.2.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_format(tensor: torch.Tensor, context: str) -> None:
    """Verify tensor format and dimensions."""
    shape = tensor.shape
    print(f"{context} - Shape: {shape}")
    if len(shape) == 3:  # HWC
        print(f"{context} - HWC format detected")
        print(f"Dims: {shape[0]}x{shape[1]}x{shape[2]}")
    elif len(shape) == 4:  # BHWC
        print(f"{context} - BHWC format detected")
        print(f"Dims: {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}")
    else:
        raise ValueError(f"Invalid tensor shape: {shape}")
    print(f"Value range: min={tensor.min():.3f}, max={tensor.max():.3f}")

def compute_histogram(image: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """Compute histogram for each channel in BHWC format."""
    print(f"Computing histogram: shape={image.shape}, bins={num_bins}")
    hist = torch.zeros(image.shape[-1], num_bins, device=image.device)
    
    for i in range(image.shape[-1]):
        channel = image[..., i]
        bins = torch.histc(channel, bins=num_bins, min=0, max=1)
        hist[i] = bins
    
    print(f"Histogram shape: {hist.shape}")
    return hist

#------------------------------------------------------------------------------
# Section 3: Node Definition
#------------------------------------------------------------------------------
class HTLevelsNode:
    """Levels correction using reference images."""
    
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
                "method": (["histogram_match", "luminance_curve"], {
                    "default": "histogram_match"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                })
            }
        }

    def apply_histogram_matching(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """Apply histogram matching with proper BHWC handling."""
        print("\nApplying histogram matching:")
        verify_tensor_format(source, "Source")
        verify_tensor_format(reference, "Reference")
        
        # Compute histograms
        source_hist = compute_histogram(source[0])
        ref_hist = compute_histogram(reference[0])
        
        # Compute CDFs
        source_cdf = torch.cumsum(source_hist, dim=1)
        ref_cdf = torch.cumsum(ref_hist, dim=1)
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[:, -1:].clamp(min=1e-5)
        ref_cdf = ref_cdf / ref_cdf[:, -1:].clamp(min=1e-5)
        print("CDFs computed and normalized")
        
        # Create lookup indices
        indices = torch.linspace(0, 1, 256, device=source.device)
        indices = indices.view(1, -1).expand(source.shape[-1], -1)
        
        # Match histograms
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
        
        # Apply correction
        result = torch.zeros_like(source)
        for b in range(source.shape[0]):
            for i in range(source.shape[-1]):
                channel = source[b, ..., i]
                scaled = (channel * 255).long().clamp(0, 255)
                result[b, ..., i] = matched[i][scaled]
        
        print(f"Output shape: {result.shape}")
        print(f"Value range: min={result.min():.3f}, max={result.max():.3f}")
        return result

    def process_levels(
        self,
        source_image: torch.Tensor,
        reference_image: torch.Tensor,
        method: str,
        strength: float
    ) -> Tuple[torch.Tensor]:
        """Process image levels with proper format handling."""
        print(f"\nHTLevelsNode v{VERSION} - Processing")
        
        try:
            # Ensure BHWC format
            if len(source_image.shape) == 3:
                source_image = source_image.unsqueeze(0)
            if len(reference_image.shape) == 3:
                reference_image = reference_image.unsqueeze(0)
            
            # Process based on method
            if method == "histogram_match":
                output = self.apply_histogram_matching(
                    source_image,
                    reference_image,
                    strength
                )
            else:  # luminance_curve
                # TODO: Implement luminance curve method
                output = source_image
            
            # Restore original dimensions
            if len(source_image.shape) == 3:
                output = output.squeeze(0)
            
            return (output,)
            
        except Exception as e:
            logger.error(f"Error in levels processing: {str(e)}")
            print(f"Error: {str(e)}")
            return (source_image,)