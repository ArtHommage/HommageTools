"""
File: ComfyUI-HommageTools/__init__.py

HommageTools Node Collection for ComfyUI
Version: 1.0.0
Description: A collection of utility nodes for ComfyUI that provide additional
functionality for text processing and image manipulation.
"""

print("[HommageTools] Starting node initialization...")

try:
    print("[HommageTools] Importing Regex Node...")
    from .nodes.ht_regex_node import HTRegexNode
    print("[HommageTools] Successfully loaded Regex Node")
except Exception as e:
    print(f"[HommageTools] Error loading Regex Node: {str(e)}")

try:
    print("[HommageTools] Importing Resize Node...")
    from .nodes.ht_resize_node import HTResizeNode
    print("[HommageTools] Successfully loaded Resize Node")
except Exception as e:
    print(f"[HommageTools] Error loading Resize Node: {str(e)}")

try:
    print("[HommageTools] Importing Resolution Node...")
    from .nodes.ht_resolution_node import HTResolutionNode
    print("[HommageTools] Successfully loaded Resolution Node")
except Exception as e:
    print(f"[HommageTools] Error loading Resolution Node: {str(e)}")

try:
    print("[HommageTools] Importing Pause Node...")
    from .nodes.ht_pause_node import HTPauseNode
    print("[HommageTools] Successfully loaded Pause Node")
except Exception as e:
    print(f"[HommageTools] Error loading Pause Node: {str(e)}")

try:
    print("[HommageTools] Importing Conversion Node...")
    from .nodes.ht_conversion_node import HTConversionNode
    print("[HommageTools] Successfully loaded Conversion Node")
except Exception as e:
    print(f"[HommageTools] Error loading Conversion Node: {str(e)}")

try:
    print("[HommageTools] Importing Switch Node...")
    from .nodes.ht_switch_node import HTSwitchNode
    print("[HommageTools] Successfully loaded Switch Node")
except Exception as e:
    print(f"[HommageTools] Error loading Switch Node: {str(e)}")

try:
    print("[HommageTools] Importing File Queue Node...")
    from .nodes.ht_file_queue_node import HTFileQueueNode
    print("[HommageTools] Successfully loaded File Queue Node")
except Exception as e:
    print(f"[HommageTools] Error loading File Queue Node: {str(e)}")

print("[HommageTools] Creating node mappings...")

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

print("[HommageTools] Created node mappings")

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

print("[HommageTools] Created display name mappings")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[HommageTools] Initialization complete!")