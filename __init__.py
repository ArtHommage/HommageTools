"""
File: ComfyUI-HommageTools/__init__.py

HommageTools Node Collection for ComfyUI
Version: 1.0.0
Description: A collection of utility nodes for ComfyUI
"""

import importlib
import sys
import traceback
from typing import Dict, Any, Optional

# Dictionary to store our node mappings
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

def import_node_class(module_name: str, class_name: str) -> Optional[Any]:
    """
    Safely import a node class from a module.
    
    Args:
        module_name: Name of the module to import from
        class_name: Name of the class to import
        
    Returns:
        Optional[Any]: The imported class or None if import fails
    """
    try:
        module = importlib.import_module(f".nodes.{module_name}", package="ComfyUI-HommageTools")
        return getattr(module, class_name)
    except Exception as e:
        print(f"Error importing {class_name} from {module_name}: {str(e)}")
        traceback.print_exc()
        return None

# Node definitions with their modules and classes
NODE_DEFINITIONS = [
    ("ht_regex_node", "HTRegexNode", "HT Regex Parser"),
    ("ht_resize_node", "HTResizeNode", "HT Smart Resize"),
    ("ht_resolution_node", "HTResolutionNode", "HT Resolution Recommender"),
    ("ht_pause_node", "HTPauseNode", "HT Pause Workflow"),
    ("ht_conversion_node", "HTConversionNode", "HT Type Converter"),
    ("ht_switch_node", "HTSwitchNode", "HT Switch"),
    ("ht_file_queue_node", "HTFileQueueNode", "HT File Queue")
]

# Import each node class safely
for module_name, class_name, display_name in NODE_DEFINITIONS:
    node_class = import_node_class(module_name, class_name)
    if node_class is not None:
        NODE_CLASS_MAPPINGS[class_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name

# If no nodes were loaded successfully, raise an error
if not NODE_CLASS_MAPPINGS:
    print("Warning: No HommageTools nodes were loaded successfully")

# Required exports
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']