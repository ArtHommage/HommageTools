"""
File: ComfyUI-HommageTools/__init__.py
Version: 1.0.3
Description: Initialization file with added inspector node
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
# Import existing nodes
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
from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode
from .nodes.ht_value_mapper_node import HTValueMapperNode
from .nodes.ht_dimension_analyzer_node import HTDimensionAnalyzerNode
from .nodes.ht_mask_validator_node import HTMaskValidatorNode
from .nodes.ht_sampler_bridge_node import HTSamplerBridgeNode
from .nodes.ht_scheduler_bridge_node import HTSchedulerBridgeNode
from .nodes.ht_flexible_node import HTFlexibleNode
from .nodes.ht_inspector_node import HTInspectorNode

# Register all nodes
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
    "HTLayerExportNode": HTLayerExportNode,
    "HTDimensionFormatterNode": HTDimensionFormatterNode,
    "HTValueMapperNode": HTValueMapperNode,
    "HTDimensionAnalyzerNode": HTDimensionAnalyzerNode,
    "HTMaskValidatorNode": HTMaskValidatorNode,
    "HTSamplerBridgeNode": HTSamplerBridgeNode,
    "HTSchedulerBridgeNode": HTSchedulerBridgeNode,
    "HTFlexibleNode": HTFlexibleNode,
    "HTInspectorNode": HTInspectorNode,
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
    "HTLayerExportNode": "HT Layer Export",
    "HTDimensionFormatterNode": "HT Dimension Formatter",
    "HTValueMapperNode": "HT Value Mapper",
    "HTDimensionAnalyzerNode": "HT Dimension Analyzer",
    "HTMaskValidatorNode": "HT Mask Validator",
    "HTSamplerBridgeNode": "HT Sampler Bridge",
    "HTSchedulerBridgeNode": "HT Scheduler Bridge",
    "HTFlexibleNode": "HT Flexible",
    "HTInspectorNode": "HT Inspector",
}

# Export mappings
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']