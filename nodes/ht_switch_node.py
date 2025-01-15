"""
File: homage_tools/nodes/ht_switch_node.py
Version: 1.0.0
Description: Simple switch node that triggers once when activated
"""

from typing import Tuple, Dict, Any

class HTSwitchNode:
    """Single-trigger switch node."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_switch"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("trigger",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False
                })
            }
        }
    
    def __init__(self):
        self._triggered = False
    
    def process_switch(self, enabled: bool) -> Tuple[bool]:
        """Process switch state."""
        if enabled and not self._triggered:
            self._triggered = True
            return (True,)
            
        if not enabled:
            self._triggered = False
            
        return (False,)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Ensure node updates on every execution."""
        return float("nan")