"""
File: ComfyUI-HommageTools/nodes/ht_pause_node.py

HommageTools Pause Node
Version: 1.0.0
Description: A node that pauses workflow execution until user interaction.
Accepts and passes through any input type unchanged.

Sections:
1. Imports and Type Definitions
2. Node Class Definition
3. Input/Output Configuration
4. Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Any, Dict, List, Tuple, Optional

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTPauseNode:
    """
    A node that pauses workflow execution until a user clicks a button.
    Accepts any input type and passes it through unchanged.
    
    Features:
    - Universal input acceptance
    - Workflow execution control
    - Visual status indication
    - Non-destructive data passthrough
    """
    
    CATEGORY = "HommageTools"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "pause_workflow"
    PAUSABLE = True  # Special ComfyUI flag to indicate this node can pause execution
    
    #--------------------------------------------------------------------------
    # Section 3: Input/Output Configuration
    #--------------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "pause_title": ("STRING", {
                    "default": "Paused",
                    "multiline": False,
                    "description": "Title to display when paused"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ()  # Empty tuple indicates pass-through behavior
    RETURN_NAMES = ()  # Empty tuple indicates pass-through behavior

    #--------------------------------------------------------------------------
    # Section 4: Processing Logic
    #--------------------------------------------------------------------------
    def pause_workflow(
        self,
        pause_title: str,
        prompt: Dict[str, Any],
        unique_id: str,
        **kwargs
    ) -> Tuple[Any, ...]:
        """
        Pause the workflow execution until resumed by user interaction.
        
        Args:
            pause_title: Text to display on the node while paused
            prompt: Internal ComfyUI prompt data
            unique_id: Internal ComfyUI node identifier
            **kwargs: Any additional inputs passed to the node
            
        Returns:
            Tuple: All inputs passed through unchanged
        """
        # Store the pause title for UI display
        self._pause_title = pause_title
        
        # Get all input values except our required/hidden ones
        passthrough_values = []
        for key, value in kwargs.items():
            if key not in ['pause_title', 'prompt', 'unique_id']:
                passthrough_values.append(value)
        
        # Return all inputs unchanged as a tuple
        return tuple(passthrough_values)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Indicate that this node should always be considered "changed" to ensure
        it executes each time.
        
        Returns:
            float: NaN to indicate always changed
        """
        return float("nan")
    
    @classmethod
    def VALIDATE_INPUTS(cls, *args, **kwargs) -> bool:
        """
        Validate all inputs. Always returns True since we accept any input type.
        
        Returns:
            bool: Always True
        """
        return True