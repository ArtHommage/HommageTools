"""
File: ComfyUI-HommageTools/__init__.py
Version: 1.0.0
Description: Initialization with remaining nodes commented out
"""

import os
from typing import Dict, Type, Any

#------------------------------------------------------------------------------
# Section 1: Basic Setup
#------------------------------------------------------------------------------
NODES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nodes")

#------------------------------------------------------------------------------
# Section 2: Node Registration
#------------------------------------------------------------------------------
# Import nodes directly
from .nodes.ht_regex_node import HTRegexNode
from .nodes.ht_parameter_extractor import HTParameterExtractorNode
from .nodes.ht_text_cleanup_node import HTTextCleanupNode
from .nodes.ht_resize_node import HTResizeNode
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_levels_node import HTLevelsNode
from .nodes.ht_baseshift_node import HTBaseShiftNode
from .nodes.ht_training_size_node import HTTrainingSizeNode
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode

# Register nodes
NODE_CLASS_MAPPINGS: Dict[str, Type[Any]] = {
    "HTRegexNode": HTRegexNode,
    "HTParameterExtractorNode": HTParameterExtractorNode,
    "HTTextCleanupNode": HTTextCleanupNode,
    "HTResizeNode": HTResizeNode,
    "HTResolutionNode": HTResolutionNode,
    "HTLevelsNode": HTLevelsNode,
    "HTBaseShiftNode": HTBaseShiftNode,
    "HTTrainingSizeNode": HTTrainingSizeNode,
    "HTSwitchNode": HTSwitchNode,
    "HTConversionNode": HTConversionNode,
    "HTLayerCollectorNode": HTLayerCollectorNode,
    "HTLayerExportNode": HTLayerExportNode
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "HTRegexNode": "HT Regex",
    "HTParameterExtractorNode": "HT Parameter Extractor",
    "HTTextCleanupNode": "HT Text Cleanup",
    "HTResizeNode": "HT Smart Resize",
    "HTResolutionNode": "HT Resolution",
    "HTLevelsNode": "HT Levels",
    "HTBaseShiftNode": "HT Base Shift",
    "HTTrainingSizeNode": "HT Training Size",
    "HTSwitchNode": "HT Switch",
    "HTConversionNode": "HT Conversion",
    "HTLayerCollectorNode": "HT Layer Collector",
    "HTLayerExportNode": "HT Layer Export"
}

# Export mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']