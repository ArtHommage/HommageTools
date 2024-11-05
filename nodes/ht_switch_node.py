"""
File: homage_tools/nodes/ht_switch_node.py

HommageTools Switch Node
Version: 1.0.0
Description: A simple toggle switch that triggers only once when activated.
Maintains state to ensure single triggering until reset.

Sections:
1. Imports and Setup
2. Node Class Definition
3. Input/Output Configuration
4. State Management
5. Switch Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Setup
#------------------------------------------------------------------------------
from typing import Tuple

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTSwitchNode:
    """
    A switch node that outputs True only on first execution after being enabled.
    Automatically resets to False until re-enabled.
    
    Features:
    - Single trigger per activation
    - Automatic reset after triggering
    - Default off state
    - Clear trigger state management
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_switch"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("trigger",)
    
    #--------------------------------------------------------------------------
    # Section 3: Input/Output Configuration
    #--------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Define the input types for the switch node.
        
        Returns:
            dict: Input specification dictionary
        """
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable switch (triggers once when turned on)"
                }),
            },
        }
    
    #--------------------------------------------------------------------------
    # Section 4: State Management
    #--------------------------------------------------------------------------
    def __init__(self):
        """Initialize the switch node with default state."""
        self._triggered = False  # Internal state tracking
    
    #--------------------------------------------------------------------------
    # Section 5: Switch Processing Logic
    #--------------------------------------------------------------------------
    def process_switch(self, enabled: bool) -> Tuple[bool]:
        """
        Process the switch state and determine output value.
        
        The switch follows these rules:
        1. Outputs True only once when first enabled
        2. Remains False after triggering until disabled and re-enabled
        3. Resets trigger state when disabled
        
        Args:
            enabled (bool): Current switch enabled state
            
        Returns:
            Tuple[bool]: Single-element tuple containing trigger state
        """
        try:
            # Handle enabling case - trigger once if not already triggered
            if enabled and not self._triggered:
                self._triggered = True
                return (True,)
                
            # Handle disabling case - reset triggered state
            if not enabled and self._triggered:
                self._triggered = False
                
            # Default case - return False
            return (False,)
            
        except Exception as e:
            print(f"Error in HTSwitchNode: {str(e)}")
            return (False,)  # Fail safe to off state

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Ensure the node updates on every execution to maintain switch state.
        
        Returns:
            float: NaN to indicate state should always be checked
        """
        return float("nan")