"""
File: homage_tools/nodes/ht_mask_validator_node.py
Description: Node for validating mask inputs and detecting meaningful mask data
Version: 1.1.0
"""

import torch
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants
#------------------------------------------------------------------------------
VERSION = "1.1.0"

#------------------------------------------------------------------------------
# Section 2: Validation Functions
#------------------------------------------------------------------------------
def verify_mask_dimensions(mask: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """Verify and extract mask dimensions."""
    shape = mask.shape
    print(f"{context} - Shape: {shape}")
    
    if len(shape) == 3:  # HWC
        height, width, channels = shape
        batch = 1
        print(f"{context} - HWC format detected")
    elif len(shape) == 4:  # BHWC
        batch, height, width, channels = shape
        print(f"{context} - BHWC format detected")
    else:
        raise ValueError(f"Invalid mask shape: {shape}")
        
    print(f"{context} - Dims: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def compare_dimensions(
    input_dims: Tuple[int, int, int, int],
    output_dims: Tuple[int, int, int, int]
) -> bool:
    """Compare input and output dimensions."""
    in_b, in_h, in_w, in_c = input_dims
    out_b, out_h, out_w, out_c = output_dims
    
    print("\nDimension comparison:")
    print(f"Input:  {in_b}x{in_h}x{in_w}x{in_c}")
    print(f"Output: {out_b}x{out_h}x{out_w}x{out_c}")
    
    matches = (
        in_h == out_h and 
        in_w == out_w and 
        (in_c == out_c or out_c == 1)
    )
    print(f"Dimensions match: {matches}")
    return matches

def is_valid_mask_tensor(mask: torch.Tensor) -> bool:
    """Validate mask tensor format."""
    try:
        batch, height, width, channels = verify_mask_dimensions(mask, "Validation")
        
        # Check dimensions
        valid_channels = channels in [1, 3, 4]
        print(f"Channel validation: {channels} -> {valid_channels}")
        
        # Check value range
        min_val = mask.min().item()
        max_val = mask.max().item()
        valid_range = min_val >= 0.0 and max_val <= 1.0
        print(f"Value range validation: {min_val:.3f}-{max_val:.3f} -> {valid_range}")
        
        return valid_channels and valid_range
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False

def has_mask_data(mask: torch.Tensor, threshold: float = 0.0) -> bool:
    """Check for meaningful mask data."""
    print(f"Checking mask data (threshold={threshold:.3f})")
    print(f"Value range: min={mask.min():.3f}, max={mask.max():.3f}")
    
    # Convert to single channel if needed
    if mask.shape[-1] > 1:
        mask = mask.mean(dim=-1, keepdim=True)
        print("Averaged multiple channels")
        
    # Calculate statistics
    total_pixels = mask.numel()
    active_pixels = torch.sum(mask > threshold).item()
    active_percentage = (active_pixels / total_pixels) * 100
    
    print(f"Active pixels: {active_pixels}/{total_pixels} ({active_percentage:.2f}%)")
    
    has_data = bool(active_pixels > 0)
    print(f"Mask contains data: {has_data}")
    return has_data

def normalize_mask_format(mask: torch.Tensor) -> torch.Tensor:
    """Ensure consistent mask format."""
    # Ensure BHWC format
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(0)
        print("Added batch dimension")
        
    # Convert to single channel if needed
    if mask.shape[-1] > 1:
        mask = mask.mean(dim=-1, keepdim=True)
        print("Converted to single channel")
        
    # Ensure float values
    if not mask.is_floating_point():
        mask = mask.float()
        print("Converted to float type")
        
    # Normalize value range
    if mask.max() > 1.0:
        mask = mask / 255.0
        print("Normalized value range to 0-1")
        
    return mask

#------------------------------------------------------------------------------
# Section 3: Node Definition
#------------------------------------------------------------------------------
class HTMaskValidatorNode:
    """Validates mask inputs and detects mask data."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "validate_mask"
    RETURN_TYPES = ("BOOLEAN", "MASK")
    RETURN_NAMES = ("has_mask_data", "normalized_mask")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Minimum value to consider as masked"
                })
            },
            "optional": {
                "mask": ("MASK",)
            }
        }

    def validate_mask(
        self,
        threshold: float,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Process mask input and validate data."""
        print(f"\nHTMaskValidatorNode v{VERSION} - Processing")
        
        try:
            if mask is None:
                print("No mask provided")
                return (False, None)
                
            # Store original dimensions
            orig_dims = verify_mask_dimensions(mask, "Input")
            
            # Validate format
            if not is_valid_mask_tensor(mask):
                return (False, None)
                
            # Normalize format
            normalized = normalize_mask_format(mask)
            
            # Verify dimensions maintained
            norm_dims = verify_mask_dimensions(normalized, "Normalized")
            if not compare_dimensions(orig_dims, norm_dims):
                print("Dimension mismatch after normalization")
                return (False, None)
                
            return (has_mask_data(normalized, threshold), normalized)
            
        except Exception as e:
            logger.error(f"Mask validation error: {str(e)}")
            print(f"Error: {str(e)}")
            return (False, None)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Ensure updates on every execution."""
        return float("nan")