"""
File: homage_tools/nodes/ht_mask_validator_node.py
Version: 1.0.0
Description: Node for validating mask inputs and detecting meaningful mask data

Sections:
1. Imports and Type Definitions
2. Validation Functions
3. Node Class Definition
4. Mask Processing Logic
5. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
from typing import Dict, Any, Tuple, Optional

#------------------------------------------------------------------------------
# Section 2: Validation Functions
#------------------------------------------------------------------------------
def is_valid_mask_tensor(mask: torch.Tensor) -> bool:
    """
    Validate if tensor has correct shape and type for mask.
    
    Args:
        mask: Input tensor to validate
        
    Returns:
        bool: True if tensor is valid mask format
    """
    if mask is None:
        return False
        
    # Check dimensions (should be 3D or 4D)
    if mask.ndim not in [3, 4]:
        return False
        
    # For 3D tensor (C,H,W), check channel count
    if mask.ndim == 3:
        return mask.shape[0] in [1, 3, 4]
        
    # For 4D tensor (B,C,H,W), check channel count
    return mask.shape[1] in [1, 3, 4]

def has_mask_data(mask: torch.Tensor, threshold: float = 0.0) -> bool:
    """
    Check if mask contains any non-zero (masked) pixels.
    
    Args:
        mask: Input mask tensor
        threshold: Minimum value to consider as masked (default: 0.0)
        
    Returns:
        bool: True if mask contains data above threshold
    """
    # Handle 4D batch tensor
    if mask.ndim == 4:
        mask = mask[0]  # Take first batch item
        
    # Convert to single channel if needed
    if mask.shape[0] > 1:
        mask = mask.mean(dim=0, keepdim=True)  # Average channels
        
    return bool(torch.any(mask > threshold))

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTMaskValidatorNode:
    """
    Validates mask inputs and detects presence of mask data.
    Returns boolean indicating if valid mask data is present.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "validate_mask"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_mask_data",)
    
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

#------------------------------------------------------------------------------
# Section 4: Mask Processing Logic
#------------------------------------------------------------------------------
    def validate_mask(
        self,
        threshold: float,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[bool]:
        """
        Process mask input and determine if valid mask data is present.
        
        Args:
            threshold: Minimum value to consider as masked
            mask: Optional input mask tensor
            
        Returns:
            Tuple[bool]: Single-item tuple containing validation result
        """
        try:
            # Check if mask is provided
            if mask is None:
                return (False,)
                
            # Validate mask tensor format
            if not is_valid_mask_tensor(mask):
                print(f"Invalid mask format: {mask.shape if mask is not None else None}")
                return (False,)
                
            # Check for mask data
            return (has_mask_data(mask, threshold),)
            
        except Exception as e:
            print(f"Error in mask validation: {str(e)}")
            return (False,)

#------------------------------------------------------------------------------
# Section 5: Error Handling
#------------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Ensure node updates on every execution.
        Necessary for proper error state handling.
        """
        return float("nan")