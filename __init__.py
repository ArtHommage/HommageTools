"""
File: ComfyUI-HommageTools/__init__.py
Version: 1.0.3
Description: Initialization file for HommageTools node collection
"""

import os
from typing import Dict, Type, Any

#------------------------------------------------------------------------------
# Section 1: Basic Setup
#------------------------------------------------------------------------------
NODES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nodes")

#------------------------------------------------------------------------------
# Section 2: Node Imports
#------------------------------------------------------------------------------
# Text Processing Nodes
from .nodes.ht_regex_node import HTRegexNode
from .nodes.ht_parameter_extractor import HTParameterExtractorNode
from .nodes.ht_text_cleanup_node import HTTextCleanupNode

# Image Processing Nodes
from .nodes.ht_surface_blur_node import HTSurfaceBlurNode
from .nodes.ht_photoshop_blur_node import HTPhotoshopBlurNode
from .nodes.ht_levels_node import HTLevelsNode
from .nodes.ht_resize_node import HTResizeNode

# Dimension Handling Nodes
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode
from .nodes.ht_dimension_analyzer_node import HTDimensionAnalyzerNode
from .nodes.ht_training_size_node import HTTrainingSizeNode

# Layer Management Nodes
from .nodes.ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode
from .nodes.ht_mask_validator_node import HTMaskValidatorNode

# AI Pipeline Nodes
from .nodes.ht_sampler_bridge_node import HTSamplerBridgeNode
from .nodes.ht_scheduler_bridge_node import HTSchedulerBridgeNode
from .nodes.ht_baseshift_node import HTBaseShiftNode

# Utility and Control Nodes
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_value_mapper_node import HTValueMapperNode
from .nodes.ht_flexible_node import HTFlexibleNode
from .nodes.ht_inspector_node import HTInspectorNode
from .nodes.ht_widget_control_node import HTWidgetControlNode

#------------------------------------------------------------------------------
# Section 3: Node Registration
#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS: Dict[str, Type[Any]] = {
    # Text Processing
    "HTRegexNode": HTRegexNode,
    "HTParameterExtractorNode": HTParameterExtractorNode,
    "HTTextCleanupNode": HTTextCleanupNode,
    
    # Image Processing
    "HTSurfaceBlurNode": HTSurfaceBlurNode,
    "HTPhotoshopBlurNode": HTPhotoshopBlurNode,
    "HTLevelsNode": HTLevelsNode,
    "HTResizeNode": HTResizeNode,
    
    # Dimension Handling
    "HTResolutionNode": HTResolutionNode,
    "HTDimensionFormatterNode": HTDimensionFormatterNode,
    "HTDimensionAnalyzerNode": HTDimensionAnalyzerNode,
    "HTTrainingSizeNode": HTTrainingSizeNode,
    
    # Layer Management
    "HTLayerCollectorNode": HTLayerCollectorNode,
    "HTLayerExportNode": HTLayerExportNode,
    "HTMaskValidatorNode": HTMaskValidatorNode,
    
    # AI Pipeline
    "HTSamplerBridgeNode": HTSamplerBridgeNode,
    "HTSchedulerBridgeNode": HTSchedulerBridgeNode,
    "HTBaseShiftNode": HTBaseShiftNode,
    
    # Utility and Control
    "HTSwitchNode": HTSwitchNode,
    "HTConversionNode": HTConversionNode,
    "HTValueMapperNode": HTValueMapperNode,
    "HTFlexibleNode": HTFlexibleNode,
    "HTInspectorNode": HTInspectorNode,
    "HTWidgetControlNode": HTWidgetControlNode
}

#------------------------------------------------------------------------------
# Section 4: Display Name Mappings
#------------------------------------------------------------------------------
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    # Text Processing
    "HTRegexNode": "HT Regex",
    "HTParameterExtractorNode": "HT Parameter Extractor",
    "HTTextCleanupNode": "HT Text Cleanup",
    
    # Image Processing
    "HTSurfaceBlurNode": "HT Surface Blur",
    "HTPhotoshopBlurNode": "HT Photoshop Blur",
    "HTLevelsNode": "HT Levels",
    "HTResizeNode": "HT Smart Resize",
    
    # Dimension Handling
    "HTResolutionNode": "HT Resolution",
    "HTDimensionFormatterNode": "HT Dimension Formatter",
    "HTDimensionAnalyzerNode": "HT Dimension Analyzer",
    "HTTrainingSizeNode": "HT Training Size",
    
    # Layer Management
    "HTLayerCollectorNode": "HT Layer Collector",
    "HTLayerExportNode": "HT Layer Export",
    "HTMaskValidatorNode": "HT Mask Validator",
    
    # AI Pipeline
    "HTSamplerBridgeNode": "HT Sampler Bridge",
    "HTSchedulerBridgeNode": "HT Scheduler Bridge",
    "HTBaseShiftNode": "HT Base Shift",
    
    # Utility and Control
    "HTSwitchNode": "HT Switch",
    "HTConversionNode": "HT Conversion",
    "HTValueMapperNode": "HT Value Mapper",
    "HTFlexibleNode": "HT Flexible",
    "HTInspectorNode": "HT Inspector",
    "HTWidgetControlNode": "HT Widget Control"
}

#------------------------------------------------------------------------------
# Section 5: Exports
#------------------------------------------------------------------------------
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']