"""
File: homage_tools/nodes/ht_node_state_controller.py
Version: 1.0.0
Description: Node for controlling multiple node states with boolean flip control
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, List
from server import PromptServer

class AnyType(str):
   """A special class that is always equal in not equal comparisons."""
   def __ne__(self, __value: object) -> bool:
       return False

# Define universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def parse_node_ids(id_string: str) -> List[int]:
   """
   Parse comma-separated node IDs into list of integers.
   
   Args:
       id_string: String of comma-separated node IDs
       
   Returns:
       List[int]: List of parsed node IDs
   """
   try:
       # Split string, strip whitespace, and convert to integers
       return [int(id.strip()) for id in id_string.split(',') if id.strip()]
   except ValueError as e:
       print(f"Error parsing node IDs: {str(e)}")
       return []

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTNodeStateController:
   """
   Controls multiple node states (active/mute) with boolean flip capability.
   Accepts comma-separated list of node IDs and applies state changes to all.
   """
   
   CATEGORY = "HommageTools/Control"
   FUNCTION = "control_state"
   RETURN_TYPES = (any_type,)
   RETURN_NAMES = ("signal_out",)
   OUTPUT_NODE = True
   
   @classmethod
   def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
       return {
           "required": {
               "node_ids": ("STRING", {
                   "default": "0",
                   "multiline": False,
                   "placeholder": "Enter node IDs (comma-separated)"
               }),
               "default_state": ("BOOLEAN", {
                   "default": True,
                   "label_on": "active",
                   "label_off": "mute"
               }),
               "boolean_input": ("BOOLEAN", {
                   "forceInput": True
               })
           },
           "optional": {
               "signal_in": (any_type, {})
           }
       }

   def control_state(
       self,
       node_ids: str,
       default_state: bool,
       boolean_input: bool,
       signal_in: Any = None
   ) -> Tuple[Any]:
       """
       Process node state control for multiple nodes based on inputs.
       
       Args:
           node_ids: Comma-separated string of node IDs to control
           default_state: Default state setting (True=active, False=mute)
           boolean_input: Boolean input that determines if default_state is used or flipped
           signal_in: Optional pass-through signal
           
       Returns:
           Tuple[Any]: Pass-through signal
       """
       # Calculate final state
       final_state = default_state if boolean_input else not default_state
       
       # Parse node IDs
       target_nodes = parse_node_ids(node_ids)
       
       if not target_nodes:
           print("Warning: No valid node IDs provided")
           return (signal_in,)
           
       # Apply state change to all target nodes
       for node_id in target_nodes:
           try:
               PromptServer.instance.send_sync("impact-node-mute-state", {
                   "node_id": node_id,
                   "is_active": final_state
               })
           except Exception as e:
               print(f"Error setting state for node {node_id}: {str(e)}")
       
       return (signal_in,)