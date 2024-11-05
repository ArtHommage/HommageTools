"""
File: ComfyUI-HommageTools/__init__.py

HommageTools Node Collection for ComfyUI
Version: 1.0.0
Description: A collection of utility nodes for ComfyUI
"""

from .nodes.ht_regex_node import HTRegexNode
from .nodes.ht_resize_node import HTResizeNode
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_pause_node import HTPauseNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_file_queue_node import HTFileQueueNode

# Dictionary of nodes to be registered with ComfyUI
NODE_CLASS_MAPPINGS = {
    "HTRegexNode": HTRegexNode,
    "HTResizeNode": HTResizeNode,
    "HTResolutionNode": HTResolutionNode,
    "HTPauseNode": HTPauseNode,
    "HTConversionNode": HTConversionNode,
    "HTSwitchNode": HTSwitchNode,
    "HTFileQueueNode": HTFileQueueNode
}

# Display names for the nodes in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "HTRegexNode": "HT Regex Parser",
    "HTResizeNode": "HT Smart Resize",
    "HTResolutionNode": "HT Resolution Recommender",
    "HTPauseNode": "HT Pause Workflow",
    "HTConversionNode": "HT Type Converter",
    "HTSwitchNode": "HT Switch",
    "HTFileQueueNode": "HT File Queue"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']