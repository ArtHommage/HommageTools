"""
File: homage_tools/__init__.py
Version: 1.2.0
Description: Initialization file for HommageTools node collection
"""

import os
import sys
from typing import Dict, Type, Any

#------------------------------------------------------------------------------
# Section 1: Basic Setup
#------------------------------------------------------------------------------
# Add package to path
EXTENSION_DIR = os.path.dirname(os.path.realpath(__file__))
if EXTENSION_DIR not in sys.path:
    sys.path.append(EXTENSION_DIR)

# Validate dependencies
try:
    from .dependency_check import validate_environment
    validate_environment(silent=True)
except Exception as e:
    print(f"\nError initializing HommageTools:")
    print(str(e))
    print("\nPlease install required dependencies using:")
    print("pip install -r requirements.txt")
    raise

#------------------------------------------------------------------------------
# Section 2: Node Imports
#------------------------------------------------------------------------------
# Text Processing Nodes
from .nodes.ht_regex_node import HTRegexNode
from .nodes.ht_parameter_extractor import HTParameterExtractorNode
from .nodes.ht_text_cleanup_node import HTTextCleanupNode
from .nodes.ht_dynamic_prompt_node import HTDynamicPromptNode

# Image Processing Nodes
from .nodes.ht_surface_blur_node import HTSurfaceBlurNode
from .nodes.ht_downsample_node import HTResolutionDownsampleNode as HTDownsampleNode
from .nodes.ht_scale_by_node import HTScaleByNode
from .nodes.ht_resolution_downsample_node import HTResolutionDownsampleNode
from .nodes.ht_photoshop_blur_node import HTPhotoshopBlurNode
from .nodes.ht_levels_node import HTLevelsNode
from .nodes.ht_resize_node import HTResizeNode
from .nodes.ht_moire_removal_node import HTMoireRemovalNode
from .nodes.ht_save_image_plus import HTSaveImagePlus
from .nodes.ht_detection_batch_processor import HTDetectionBatchProcessor

# Dimension Handling Nodes
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode
from .nodes.ht_dimension_analyzer_node import HTDimensionAnalyzerNode
from .nodes.ht_training_size_node import HTTrainingSizeNode
from .nodes.ht_mask_dilation_node import HTMaskDilationNode
from .nodes.ht_multi_mask_dilation_node import HTMultiMaskDilationNode
from .nodes.ht_tensor_info_node import HTTensorInfoNode

# Layer Management Nodes
from .nodes.ht_mask_validator_node import HTMaskValidatorNode

# AI Pipeline Nodes
from .nodes.ht_sampler_bridge_node import HTSamplerBridgeNode
from .nodes.ht_scheduler_bridge_node import HTSchedulerBridgeNode
from .nodes.ht_baseshift_node import HTBaseShiftNode
from .nodes.ht_dynamic_switch_node import HTDynamicSwitchNode
from .nodes.ht_seed_node import HTSeedNode
from .nodes.ht_seed_advanced_node import HTSeedAdvancedNode  # Advanced seed generator

# Utility and Control Nodes
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_status_indicator_node import HTStatusIndicatorNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_value_mapper_node import HTValueMapperNode
from .nodes.ht_flexible_node import HTFlexibleNode
from .nodes.ht_inspector_node import HTInspectorNode
from .nodes.ht_widget_control_node import HTWidgetControlNode
from .nodes.ht_splitter_node import HTSplitterNode
from .nodes.ht_node_state_controller import HTNodeStateController
from .nodes.ht_node_unmute_all import HTNodeUnmuteAll
from .nodes.ht_null_node import HTNullNode
from .nodes.ht_console_logger_node import HTConsoleLoggerNode

# Model Management Nodes
from .nodes.ht_diffusion_loader_multi import HTDiffusionLoaderMulti

#------------------------------------------------------------------------------
# Section 3: Node Registration
#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    # Text Processing
    "HTRegexNode": HTRegexNode,
    "HTParameterExtractorNode": HTParameterExtractorNode,
    "HTTextCleanupNode": HTTextCleanupNode,
    "HTDynamicPromptNode": HTDynamicPromptNode,
    
    # Image Processing
    "HTSurfaceBlurNode": HTSurfaceBlurNode,
    "HTDownsampleNode": HTDownsampleNode,
    "HTScaleByNode": HTScaleByNode,
    "HTResolutionDownsampleNode": HTResolutionDownsampleNode,
    "HTPhotoshopBlurNode": HTPhotoshopBlurNode,
    "HTLevelsNode": HTLevelsNode,
    "HTResizeNode": HTResizeNode,
    "HTMoireRemovalNode": HTMoireRemovalNode,
    "HTSaveImagePlus": HTSaveImagePlus,
    "HTDetectionBatchProcessor": HTDetectionBatchProcessor,
    
    # Dimension Handling
    "HTResolutionNode": HTResolutionNode,
    "HTDimensionFormatterNode": HTDimensionFormatterNode,
    "HTDimensionAnalyzerNode": HTDimensionAnalyzerNode,
    "HTTrainingSizeNode": HTTrainingSizeNode,
    "HTMaskDilationNode": HTMaskDilationNode,
    "HTMultiMaskDilationNode": HTMultiMaskDilationNode,
    "HTTensorInfoNode": HTTensorInfoNode,
    
    # Layer Management
    "HTMaskValidatorNode": HTMaskValidatorNode,
    
    # AI Pipeline
    "HTSamplerBridgeNode": HTSamplerBridgeNode,
    "HTSchedulerBridgeNode": HTSchedulerBridgeNode,
    "HTBaseShiftNode": HTBaseShiftNode,
    "HTDynamicSwitchNode": HTDynamicSwitchNode,
    "HTSeedNode": HTSeedNode,
    "HTSeedAdvancedNode": HTSeedAdvancedNode,  # Advanced seed generator
    
    # Utility and Control
    "HTSwitchNode": HTSwitchNode,
    "HTStatusIndicatorNode": HTStatusIndicatorNode,
    "HTConversionNode": HTConversionNode,
    "HTValueMapperNode": HTValueMapperNode,
    "HTFlexibleNode": HTFlexibleNode,
    "HTInspectorNode": HTInspectorNode,
    "HTWidgetControlNode": HTWidgetControlNode,
    "HTSplitterNode": HTSplitterNode,
    "HTNodeStateController": HTNodeStateController,
    "HTNodeUnmuteAll": HTNodeUnmuteAll,
    "HTNullNode": HTNullNode,
    "HTConsoleLoggerNode": HTConsoleLoggerNode,
    
    # Model Management
    "HTDiffusionLoaderMulti": HTDiffusionLoaderMulti
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Text Processing
    "HTRegexNode": "HT Regex",
    "HTParameterExtractorNode": "HT Parameter Extractor",
    "HTTextCleanupNode": "HT Text Cleanup",
    "HTDynamicPromptNode": "HT Dynamic Prompt",
    
    # Image Processing
    "HTSurfaceBlurNode": "HT Surface Blur",
    "HTDownsampleNode": "HT Downsample",
    "HTScaleByNode": "HT Scale By",
    "HTResolutionDownsampleNode": "HT Resolution Downsample",
    "HTPhotoshopBlurNode": "HT Photoshop Blur",
    "HTLevelsNode": "HT Levels",
    "HTResizeNode": "HT Smart Resize",
    "HTMoireRemovalNode": "HT Moiré Removal",
    "HTSaveImagePlus": "HT Save Image Plus",
    "HTDetectionBatchProcessor": "HT Detection Batch Processor",
    
    # Dimension Handling
    "HTResolutionNode": "HT Resolution",
    "HTDimensionFormatterNode": "HT Dimension Formatter",
    "HTDimensionAnalyzerNode": "HT Dimension Analyzer",
    "HTTrainingSizeNode": "HT Training Size",
    "HTMaskDilationNode": "HT Mask Dilate",
    "HTMultiMaskDilationNode": "HT Multi Mask Dilate",
    "HTTensorInfoNode": "HT Tensor Info",
    
    # Layer Management
    "HTMaskValidatorNode": "HT Mask Validator",
    
    # AI Pipeline
    "HTSamplerBridgeNode": "HT Sampler Bridge",
    "HTSchedulerBridgeNode": "HT Scheduler Bridge",
    "HTBaseShiftNode": "HT Base Shift",
    "HTDynamicSwitchNode": "HT Dynamic Switch",
    "HTSeedNode": "HT Seed",
    "HTSeedAdvancedNode": "HT Seed Advanced",  # Advanced seed generator
    
    # Utility and Control
    "HTSwitchNode": "HT Switch",
    "HTStatusIndicatorNode": "HT Status Indicator",
    "HTConversionNode": "HT Conversion",
    "HTValueMapperNode": "HT Value Mapper",
    "HTFlexibleNode": "HT Flexible",
    "HTInspectorNode": "HT Inspector",
    "HTWidgetControlNode": "HT Widget Control",
    "HTSplitterNode": "HT Splitter",
    "HTNodeStateController": "HT Node State Controller",
    "HTNodeUnmuteAll": "HT Unmute All",
    "HTNullNode": "HT Null Value",
    "HTConsoleLoggerNode": "HT Console Logger",
    
    # Model Management
    "HTDiffusionLoaderMulti": "HT Multi Model Loader"
}

#------------------------------------------------------------------------------
# Section 4: Exports
#------------------------------------------------------------------------------
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

#------------------------------------------------------------------------------
# Section 5: Web UI Setup
#------------------------------------------------------------------------------
# Create web directory if it doesn't exist
web_dir = os.path.join(EXTENSION_DIR, "web")
js_dir = os.path.join(web_dir, "js")

if not os.path.exists(js_dir):
    try:
        os.makedirs(js_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create web directories: {e}")