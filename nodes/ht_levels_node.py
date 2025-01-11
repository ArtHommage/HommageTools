"""
File: homage_tools/nodes/ht_levels_node.py

HommageTools Levels Correction Node
Version: 1.0.0
Description: A node that performs levels correction on an image using either
luminance curve matching or histogram matching with a reference image.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Image Processing Methods
4. Histogram Methods
5. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Literal

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTLevelsNode:
    """
    A ComfyUI node that provides image levels correction using reference images.
    
    Features:
    - Two correction methods: Luminance curve and Histogram matching
    - Adjustment intensity control
    - Support for both RGB and grayscale processing
    - Reference image-based correction
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_levels"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    
    # Available correction methods
    CORRECTION_METHODS = [
        "luminance_curve",  # Match the luminance curve of reference
        "histogram_match"   # Direct histogram matching
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types and their default values."""
        return {
            "required": {
                "source_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "method": (cls.CORRECTION_METHODS, {
                    "default": "luminance_curve",
                    "description": "Method for levels correction"
                }),
                "adjustment_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Strength of the correction (-1.0 to 2.0)"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 3: Image Processing Methods
#------------------------------------------------------------------------------
    def _tensor_to_cv2(self, image: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to OpenCV format."""
        # Convert to numpy and scale to 0-255 range
        numpy_image = (image.cpu().numpy() * 255).astype(np.uint8)
        # Convert from RGB to BGR for OpenCV
        if len(numpy_image.shape) == 3:
            numpy_image = np.transpose(numpy_image, (1, 2, 0))
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return numpy_image

    def _cv2_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert an OpenCV image back to PyTorch tensor."""
        # Convert from BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
        # Scale back to 0-1 range and convert to tensor
        return torch.from_numpy(image.astype(np.float32) / 255.0)

    def _apply_luminance_curve(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply luminance curve correction using the reference image.
        
        Args:
            source: Source image in OpenCV format
            reference: Reference image in OpenCV format
            strength: Adjustment strength (-1.0 to 2.0)
            
        Returns:
            np.ndarray: Processed image
        """
        # Convert images to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        # Extract L channels
        source_l = source_lab[:,:,0]
        reference_l = reference_lab[:,:,0]
        
        # Calculate cumulative histograms
        source_hist = cv2.calcHist([source_l], [0], None, [256], [0,256])
        reference_hist = cv2.calcHist([reference_l], [0], None, [256], [0,256])
        
        source_cdf = source_hist.cumsum()
        reference_cdf = reference_hist.cumsum()
        
        # Normalize CDFs
        source_cdf_normalized = source_cdf / source_cdf.max()
        reference_cdf_normalized = reference_cdf / reference_cdf.max()
        
        # Create lookup table for mapping
        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            src_value = source_cdf_normalized[i]
            for j in range(256):
                if reference_cdf_normalized[j] >= src_value:
                    lookup_table[i] = j
                    break
        
        # Apply strength adjustment
        if strength != 1.0:
            # Interpolate between original and mapped values
            original_values = np.arange(256, dtype=np.float32)
            adjusted_values = lookup_table.astype(np.float32)
            
            if strength < 0:
                # Invert effect for negative strength
                strength = abs(strength)
                adjusted_values = original_values + (original_values - adjusted_values) * strength
            else:
                adjusted_values = original_values + (adjusted_values - original_values) * strength
                
            # Clip to valid range
            lookup_table = np.clip(adjusted_values, 0, 255).astype(np.uint8)
        
        # Apply correction to L channel
        corrected_l = cv2.LUT(source_l, lookup_table)
        
        # Reconstruct LAB image
        source_lab[:,:,0] = corrected_l
        
        # Convert back to BGR
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

#------------------------------------------------------------------------------
# Section 4: Histogram Methods
#------------------------------------------------------------------------------
    def _match_histograms(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply direct histogram matching between source and reference.
        
        Args:
            source: Source image in OpenCV format
            reference: Reference image in OpenCV format
            strength: Adjustment strength (-1.0 to 2.0)
            
        Returns:
            np.ndarray: Processed image
        """
        # Split into channels
        source_channels = cv2.split(source)
        reference_channels = cv2.split(reference)
        matched_channels = []
        
        for src_chan, ref_chan in zip(source_channels, reference_channels):
            # Calculate histograms
            src_hist = cv2.calcHist([src_chan], [0], None, [256], [0,256])
            ref_hist = cv2.calcHist([ref_chan], [0], None, [256], [0,256])
            
            # Calculate CDFs
            src_cdf = src_hist.cumsum()
            ref_cdf = ref_hist.cumsum()
            
            # Normalize CDFs
            src_cdf_normalized = src_cdf / src_cdf.max()
            ref_cdf_normalized = ref_cdf / ref_cdf.max()
            
            # Create lookup table
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                src_value = src_cdf_normalized[i]
                for j in range(256):
                    if ref_cdf_normalized[j] >= src_value:
                        lookup_table[i] = j
                        break
            
            # Apply strength adjustment
            if strength != 1.0:
                original_values = np.arange(256, dtype=np.float32)
                adjusted_values = lookup_table.astype(np.float32)
                
                if strength < 0:
                    strength = abs(strength)
                    adjusted_values = original_values + (original_values - adjusted_values) * strength
                else:
                    adjusted_values = original_values + (adjusted_values - original_values) * strength
                    
                lookup_table = np.clip(adjusted_values, 0, 255).astype(np.uint8)
            
            # Apply correction
            matched_channels.append(cv2.LUT(src_chan, lookup_table))
        
        # Merge channels
        return cv2.merge(matched_channels)

#------------------------------------------------------------------------------
# Section 5: Main Processing Logic
#------------------------------------------------------------------------------
    def process_levels(
        self,
        source_image: torch.Tensor,
        reference_image: torch.Tensor,
        method: Literal["luminance_curve", "histogram_match"],
        adjustment_strength: float
    ) -> Tuple[torch.Tensor]:
        """
        Main processing function to apply levels correction.
        
        Args:
            source_image: Source image tensor
            reference_image: Reference image tensor
            method: Correction method to use
            adjustment_strength: Strength of the correction
            
        Returns:
            Tuple[torch.Tensor]: Single-element tuple containing processed image
        """
        try:
            # Convert tensors to OpenCV format
            source_cv = self._tensor_to_cv2(source_image)
            reference_cv = self._tensor_to_cv2(reference_image)
            
            # Process based on selected method
            if method == "luminance_curve":
                processed = self._apply_luminance_curve(
                    source_cv,
                    reference_cv,
                    adjustment_strength
                )
            else:  # histogram_match
                processed = self._match_histograms(
                    source_cv,
                    reference_cv,
                    adjustment_strength
                )
            
            # Convert back to tensor
            result = self._cv2_to_tensor(processed)
            
            return (result,)
            
        except Exception as e:
            print(f"Error in HTLevelsNode: {str(e)}")
            return (source_image,)  # Return original image on error