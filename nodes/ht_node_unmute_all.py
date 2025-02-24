"""
File: homage_tools/nodes/ht_node_unmute_all.py
Version: 1.0.0
Description: Node for unmuting all nodes in the workflow with signal pass-through
"""

from typing import Dict, Any, Tuple
from server import PromptServer

class AnyType(str):
   def __ne__(self, __value: object) -> bool:
       return False

any_type = AnyType("*")

class HTNodeUnmuteAll:
   CATEGORY = "HommageTools/Control"
   FUNCTION = "unmute_all"
   RETURN_TYPES = (any_type,)
   RETURN_NAMES = ("signal_out",)
   OUTPUT_NODE = True
   
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {},
           "optional": {
               "signal_in": (any_type, {})
           },
           "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
       }

   def unmute_all(self, prompt=None, extra_pnginfo=None, signal_in: Any = None):
       if extra_pnginfo and "workflow" in extra_pnginfo:
           workflow = extra_pnginfo["workflow"]
           for node in workflow["nodes"]:
               try:
                   node_id = node["id"]
                   PromptServer.instance.send_sync("impact-node-mute-state", {
                       "node_id": node_id,
                       "is_active": True
                   })
               except Exception as e:
                   print(f"Error unmuting node {node_id}: {str(e)}")
       return (signal_in,)