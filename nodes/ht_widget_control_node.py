"""
File: homage_tools/nodes/ht_widget_control_node.py
Version: 1.0.0
Description: Node for controlling widget values at the system level with targeting
"""

import copy
from server import PromptServer

#------------------------------------------------------------------------------
# Section 1: Node Definition
#------------------------------------------------------------------------------
class HTWidgetControlNode:
    """
    Controls widget values by intercepting them at the system level.
    Allows modification of specific widget types for targeted nodes.
    """
    
    CATEGORY = "HommageTools/System"
    RETURN_TYPES = tuple()
    FUNCTION = "control_widget"
    OUTPUT_NODE = True  # Mark as output node since it affects system state
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["fixed", "increment", "decrement", "randomize"], {
                    "default": "fixed"
                }),
                "target_widget": (["control_after_generate"], {
                    "default": "control_after_generate"
                }),
                "target_type": (["global", "by_class", "by_id"], {
                    "default": "global",
                    "description": "How to target nodes"
                })
            },
            "optional": {
                "class_name": ("STRING", {
                    "default": "KSampler",
                    "description": "Node class name to target (for by_class mode)"
                }),
                "node_id": ("STRING", {
                    "default": "",
                    "description": "Specific node ID to target (for by_id mode)"
                })
            }
        }

    #--------------------------------------------------------------------------
    # Section 2: Widget Control Logic
    #--------------------------------------------------------------------------
    def _inject_widget_value(self, mode: str, widget_name: str, target_type: str, class_name: str = "", node_id: str = ""):
        """
        Inject widget value into ComfyUI's widget system with targeting.
        
        Args:
            mode: Value to inject ('fixed', 'increment', etc)
            widget_name: Name of widget to target
            target_type: How to target nodes (global, by_class, by_id)
            class_name: Optional class name to target
            node_id: Optional node ID to target
        """
        try:
            # Get server instance
            server = PromptServer.instance
            
            # Initialize widget values if needed
            if not hasattr(server, 'widget_values'):
                server.widget_values = {}
                
            # Initialize target tracking if needed
            if not hasattr(server, 'widget_targets'):
                server.widget_targets = {}
            
            # Create unique key for this widget control
            if target_type == "global":
                key = widget_name
            elif target_type == "by_class":
                key = f"{widget_name}_{class_name}"
            else:  # by_id
                key = f"{widget_name}_{node_id}"
                
            # Store both value and targeting info
            server.widget_values[key] = mode
            server.widget_targets[key] = {
                "type": target_type,
                "class": class_name,
                "id": node_id
            }
            
            print(f"\nWidget Control Debug:")
            print(f"Set {key} = {mode}")
            print(f"Target type: {target_type}")
            if class_name:
                print(f"Class name: {class_name}")
            if node_id:
                print(f"Node ID: {node_id}")
            
        except Exception as e:
            print(f"Error injecting widget value: {str(e)}")

    #--------------------------------------------------------------------------
    # Section 3: Main Processing
    #--------------------------------------------------------------------------
    def control_widget(
        self, 
        mode: str, 
        target_widget: str, 
        target_type: str,
        class_name: str = "",
        node_id: str = ""
    ):
        """
        Process widget control request with targeting.
        
        Args:
            mode: Control mode to set
            target_widget: Widget type to target
            target_type: How to target nodes
            class_name: Optional class name to target
            node_id: Optional node ID to target
        """
        # Validate targeting parameters
        if target_type == "by_class" and not class_name:
            print("Warning: Class targeting selected but no class name provided")
            return tuple()
            
        if target_type == "by_id" and not node_id:
            print("Warning: ID targeting selected but no node ID provided")
            return tuple()
            
        # Inject the value with targeting
        self._inject_widget_value(mode, target_widget, target_type, class_name, node_id)
        return tuple()