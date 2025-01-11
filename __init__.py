"""
File: ComfyUI-HommageTools/__init__.py

HommageTools Node Collection for ComfyUI
Version: 1.0.0
Description: A collection of utility nodes for ComfyUI
"""

from .nodes.ht_regex_node import HTRegexNode
from .nodes.ht_resize_node import HTResizeNode
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_parameter_extractor import HTParameterExtractorNode
from .nodes.ht_text_cleanup_node import HTTextCleanupNode
from .nodes.ht_baseshift_node import HTBaseShiftNode
from .nodes.ht_levels_node import HTLevelsNode
from .nodes.ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode
from .nodes.ht_training_size_node import HTTrainingSizeNode
from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode

# Dictionary of nodes to be registered with ComfyUI
NODE_CLASS_MAPPINGS = {
    "HTRegexNode": HTRegexNode,
    "HTResizeNode": HTResizeNode,
    "HTResolutionNode": HTResolutionNode,
    "HTConversionNode": HTConversionNode,
    "HTSwitchNode": HTSwitchNode,
    "HTParameterExtractorNode": HTParameterExtractorNode,
    "HTTextCleanupNode": HTTextCleanupNode,
    "HTBaseShiftNode": HTBaseShiftNode,
    "HTLevelsNode": HTLevelsNode,
    "HTLayerCollectorNode": HTLayerCollectorNode,
    "HTLayerExportNode": HTLayerExportNode,
    "HTTrainingSizeNode": HTTrainingSizeNode,
    "HTDimensionFormatterNode": HTDimensionFormatterNode
}

# Display names for the nodes in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "HTRegexNode": "HT Regex Parser",
    "HTResizeNode": "HT Smart Resize",
    "HTResolutionNode": "HT Resolution Recommender",
    "HTPauseNode": "HT Pause Workflow",
    "HTConversionNode": "HT Type Converter",
    "HTSwitchNode": "HT Switch",
    "HTParameterExtractorNode": "HT Parameter Extractor",
    "HTTextCleanupNode": "HT Text Cleanup",
    "HTBaseShiftNode": "HT Base Shift Calculator",
    "HTLevelsNode": "HT Levels Correction",
    "HTLayerCollectorNode": "HT Layer Collector",
    "HTLayerExportNode": "HT Layer Export",
    "HTTrainingSizeNode": "HT Training Size Calculator",
    "HTDimensionFormatterNode": "HT Dimension Formatter"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']