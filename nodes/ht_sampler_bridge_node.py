"""
File: homage_tools/nodes/ht_sampler_bridge_node.py
Version: 1.0.0
Description: Bridge node for converting string inputs to sampler selections

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Node Class Definition
4. Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple
import comfy.samplers

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def validate_sampler_name(name: str) -> str:
    """
    Validate and normalize sampler name.
    
    Args:
        name: Input sampler name
        
    Returns:
        str: Validated sampler name or default
    """
    # Convert to lowercase and remove spaces for comparison
    normalized = name.lower().strip()
    
    # Get list of valid samplers
    valid_samplers = [s.lower() for s in comfy.samplers.SAMPLER_NAMES]
    
    # Check for exact match
    if normalized in valid_samplers:
        return comfy.samplers.SAMPLER_NAMES[valid_samplers.index(normalized)]
        
    # Check for partial match
    matches = [s for s in valid_samplers if normalized in s]
    if matches:
        return comfy.samplers.SAMPLER_NAMES[valid_samplers.index(matches[0])]
        
    # Default to euler for safety
    print(f"Warning: Invalid sampler name '{name}'. Using 'euler' as default.")
    return "euler"

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTSamplerBridgeNode:
    """
    Converts string inputs to valid sampler selections with fallback handling.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_sampler"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "sampler_name": ("STRING", {
                    "default": "euler",
                    "multiline": False
                })
            }
        }

#------------------------------------------------------------------------------
# Section 4: Processing Logic
#------------------------------------------------------------------------------
    def process_sampler(self, sampler_name: str) -> Tuple[Any]:
        """
        Process sampler name input and return valid sampler object.
        
        Args:
            sampler_name: Input sampler name string
            
        Returns:
            Tuple[Any]: Tuple containing sampler object
        """
        # Validate and get proper sampler name
        valid_name = validate_sampler_name(sampler_name)
        
        # Create sampler object
        sampler = comfy.samplers.sampler_object(valid_name)
        
        return (sampler,)