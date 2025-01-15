"""
File: homage_tools/nodes/ht_scheduler_bridge_node.py
Version: 1.0.0
Description: Bridge node for converting string inputs to scheduler selections

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
import torch

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def validate_scheduler_name(name: str) -> str:
    """
    Validate and normalize scheduler name.
    
    Args:
        name: Input scheduler name
        
    Returns:
        str: Validated scheduler name or default
    """
    # Convert to lowercase and remove spaces for comparison
    normalized = name.lower().strip()
    
    # Get list of valid schedulers
    valid_schedulers = [s.lower() for s in comfy.samplers.SCHEDULER_NAMES]
    
    # Check for exact match
    if normalized in valid_schedulers:
        return comfy.samplers.SCHEDULER_NAMES[valid_schedulers.index(normalized)]
        
    # Check for partial match
    matches = [s for s in valid_schedulers if normalized in s]
    if matches:
        return comfy.samplers.SCHEDULER_NAMES[valid_schedulers.index(matches[0])]
        
    # Default to normal for safety
    print(f"Warning: Invalid scheduler name '{name}'. Using 'normal' as default.")
    return "normal"

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTSchedulerBridgeNode:
    """
    Bridge node for converting string inputs to scheduler selections with proper validation.
    Includes support for denoise strength and step configuration.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_scheduler"
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler_name": ("STRING", {
                    "default": "normal",
                    "multiline": False
                }),
                "steps": ("STRING", {
                    "default": "20",
                    "multiline": False
                }),
                "denoise": ("STRING", {
                    "default": "1.0",
                    "multiline": False
                })
            }
        }

#------------------------------------------------------------------------------
# Section 4: Processing Logic
#------------------------------------------------------------------------------
    def process_scheduler(
        self, 
        model: Any,
        scheduler_name: str,
        steps: str,
        denoise: str
    ) -> Tuple[torch.Tensor]:
        """
        Process scheduler configuration and return sigmas.
        
        Args:
            model: ComfyUI model object
            scheduler_name: Input scheduler name string
            steps: Number of steps as string
            denoise: Denoise strength as string
            
        Returns:
            Tuple[torch.Tensor]: Tuple containing sigmas tensor
        """
        try:
            # Parse numeric inputs
            num_steps = max(1, int(float(steps)))
            denoise_val = max(0.0, min(1.0, float(denoise)))
            
        except ValueError:
            print(f"Warning: Invalid numeric inputs. Using defaults: steps=20, denoise=1.0")
            num_steps = 20
            denoise_val = 1.0
            
        # Validate scheduler name
        valid_name = validate_scheduler_name(scheduler_name)
        
        # Calculate total steps for denoising
        total_steps = num_steps
        if denoise_val < 1.0:
            if denoise_val <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(num_steps / denoise_val)
        
        # Calculate sigmas
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"),
            valid_name,
            total_steps
        ).cpu()
        
        # Trim sigmas based on denoise
        sigmas = sigmas[-(num_steps + 1):]
        
        return (sigmas,)