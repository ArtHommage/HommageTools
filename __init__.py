"""
File: homage_tools/__init__.py
Version: 1.3.0
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
from .nodes.ht_photoshop_blur_node import HTPhotoshopBlurNode
from .nodes.ht_moire_removal_node import HTMoireRemovalNode
from .nodes.ht_levels_node import HTLevelsNode

# Image Resizing and Dimension Nodes
from .nodes.ht_resize_node import HTResizeNode
from .nodes.ht_scale_by_node import HTScaleByNode
from .nodes.ht_resolution_downsample_node import HTResolutionDownsampleNode
from .nodes.ht_resolution_node import HTResolutionNode
from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode
from .nodes.ht_dimension_analyzer_node import HTDimensionAnalyzerNode
from .nodes.ht_training_size_node import HTTrainingSizeNode

# Mask Processing Nodes
from .nodes.ht_mask_dilation_node import HTMaskDilationNode
from .nodes.ht_multi_mask_dilation_node import HTMultiMaskDilationNode
from .nodes.ht_mask_validator_node import HTMaskValidatorNode

# Seed and Sampling Nodes
from .nodes.ht_seed_node import HTSeedNode
from .nodes.ht_seed_advanced_node import HTSeedAdvancedNode
from .nodes.ht_sampler_bridge_node import HTSamplerBridgeNode
from .nodes.ht_scheduler_bridge_node import HTSchedulerBridgeNode
from .nodes.ht_baseshift_node import HTBaseShiftNode

# Layer Management Nodes
from .nodes.ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode

# Output and Export Nodes
from .nodes.ht_save_image_plus import HTSaveImagePlus
from .nodes.ht_detection_batch_processor import HTDetectionBatchProcessor

# Workflow Control Nodes
from .nodes.ht_switch_node import HTSwitchNode
from .nodes.ht_dynamic_switch_node import HTDynamicSwitchNode
from .nodes.ht_splitter_node import HTSplitterNode
from .nodes.ht_node_state_controller import HTNodeStateController
from .nodes.ht_node_unmute_all import HTNodeUnmuteAll
from .nodes.ht_widget_control_node import HTWidgetControlNode

# Utility and Debug Nodes
from .nodes.ht_status_indicator_node import HTStatusIndicatorNode
from .nodes.ht_conversion_node import HTConversionNode
from .nodes.ht_value_mapper_node import HTValueMapperNode
from .nodes.ht_flexible_node import HTFlexibleNode
from .nodes.ht_inspector_node import HTInspectorNode
from .nodes.ht_null_node import HTNullNode
from .nodes.ht_console_logger_node import HTConsoleLoggerNode
from .nodes.ht_tensor_info_node import HTTensorInfoNode

# AI Integration Nodes
from .nodes.ht_diffusion_loader_multi import HTDiffusionLoaderMulti
from .nodes.ht_gemini_node import HTGeminiNode

#------------------------------------------------------------------------------
# Section 3: Node Registration
#------------------------------------------------------------------------------
# Create an alias for the resolution downsample node to maintain backward compatibility
HTDownsampleNode = HTResolutionDownsampleNode

NODE_CLASS_MAPPINGS = {
    # Text Processing
    "HTRegexNode": HTRegexNode,
    "HTParameterExtractorNode": HTParameterExtractorNode,
    "HTTextCleanupNode": HTTextCleanupNode,
    "HTDynamicPromptNode": HTDynamicPromptNode,
    
    # Image Processing
    "HTSurfaceBlurNode": HTSurfaceBlurNode,
    "HTPhotoshopBlurNode": HTPhotoshopBlurNode,
    "HTMoireRemovalNode": HTMoireRemovalNode,
    "HTLevelsNode": HTLevelsNode,
    
    # Image Resizing and Dimension
    "HTResizeNode": HTResizeNode,
    "HTScaleByNode": HTScaleByNode,
    "HTDownsampleNode": HTDownsampleNode,
    "HTResolutionDownsampleNode": HTResolutionDownsampleNode,
    "HTResolutionNode": HTResolutionNode,
    "HTDimensionFormatterNode": HTDimensionFormatterNode,
    "HTDimensionAnalyzerNode": HTDimensionAnalyzerNode,
    "HTTrainingSizeNode": HTTrainingSizeNode,
    
    # Mask Processing
    "HTMaskDilationNode": HTMaskDilationNode,
    "HTMultiMaskDilationNode": HTMultiMaskDilationNode,
    "HTMaskValidatorNode": HTMaskValidatorNode,
    
    # Seed and Sampling
    "HTSeedNode": HTSeedNode,
    "HTSeedAdvancedNode": HTSeedAdvancedNode,
    "HTSamplerBridgeNode": HTSamplerBridgeNode,
    "HTSchedulerBridgeNode": HTSchedulerBridgeNode,
    "HTBaseShiftNode": HTBaseShiftNode,
    
    # Layer Management
    "HTLayerCollectorNode": HTLayerCollectorNode,
    "HTLayerExportNode": HTLayerExportNode,
    
    # Output and Export
    "HTSaveImagePlus": HTSaveImagePlus,
    "HTDetectionBatchProcessor": HTDetectionBatchProcessor,
    
    # Workflow Control
    "HTSwitchNode": HTSwitchNode,
    "HTDynamicSwitchNode": HTDynamicSwitchNode,
    "HTSplitterNode": HTSplitterNode,
    "HTNodeStateController": HTNodeStateController,
    "HTNodeUnmuteAll": HTNodeUnmuteAll,
    "HTWidgetControlNode": HTWidgetControlNode,
    
    # Utility and Debug
    "HTStatusIndicatorNode": HTStatusIndicatorNode,
    "HTConversionNode": HTConversionNode,
    "HTValueMapperNode": HTValueMapperNode,
    "HTFlexibleNode": HTFlexibleNode,
    "HTInspectorNode": HTInspectorNode,
    "HTNullNode": HTNullNode,
    "HTConsoleLoggerNode": HTConsoleLoggerNode,
    "HTTensorInfoNode": HTTensorInfoNode,
    
    # AI Integration
    "HTDiffusionLoaderMulti": HTDiffusionLoaderMulti,
    "HTGeminiNode": HTGeminiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Text Processing
    "HTRegexNode": "HT Regex",
    "HTParameterExtractorNode": "HT Parameter Extractor",
    "HTTextCleanupNode": "HT Text Cleanup",
    "HTDynamicPromptNode": "HT Dynamic Prompt",
    
    # Image Processing
    "HTSurfaceBlurNode": "HT Surface Blur",
    "HTPhotoshopBlurNode": "HT Photoshop Blur",
    "HTMoireRemovalNode": "HT Moiré Removal",
    "HTLevelsNode": "HT Levels",
    
    # Image Resizing and Dimension
    "HTResizeNode": "HT Smart Resize",
    "HTScaleByNode": "HT Scale By",
    "HTDownsampleNode": "HT Downsample",
    "HTResolutionDownsampleNode": "HT Resolution Downsample",
    "HTResolutionNode": "HT Resolution",
    "HTDimensionFormatterNode": "HT Dimension Formatter",
    "HTDimensionAnalyzerNode": "HT Dimension Analyzer",
    "HTTrainingSizeNode": "HT Training Size",
    
    # Mask Processing
    "HTMaskDilationNode": "HT Mask Dilate",
    "HTMultiMaskDilationNode": "HT Multi Mask Dilate",
    "HTMaskValidatorNode": "HT Mask Validator",
    
    # Seed and Sampling
    "HTSeedNode": "HT Seed",
    "HTSeedAdvancedNode": "HT Seed Advanced",
    "HTSamplerBridgeNode": "HT Sampler Bridge",
    "HTSchedulerBridgeNode": "HT Scheduler Bridge",
    "HTBaseShiftNode": "HT Base Shift",
    
    # Layer Management
    "HTLayerCollectorNode": "HT Layer Collector",
    "HTLayerExportNode": "HT Layer Export",
    
    # Output and Export
    "HTSaveImagePlus": "HT Save Image Plus",
    "HTDetectionBatchProcessor": "HT Detection Batch Processor",
    
    # Workflow Control
    "HTSwitchNode": "HT Switch",
    "HTDynamicSwitchNode": "HT Dynamic Switch",
    "HTSplitterNode": "HT Splitter",
    "HTNodeStateController": "HT Node State Controller",
    "HTNodeUnmuteAll": "HT Unmute All",
    "HTWidgetControlNode": "HT Widget Control",
    
    # Utility and Debug
    "HTStatusIndicatorNode": "HT Status Indicator",
    "HTConversionNode": "HT Conversion",
    "HTValueMapperNode": "HT Value Mapper",
    "HTFlexibleNode": "HT Flexible",
    "HTInspectorNode": "HT Inspector",
    "HTNullNode": "HT Null Value",
    "HTConsoleLoggerNode": "HT Console Logger",
    "HTTensorInfoNode": "HT Tensor Info",
    
    # AI Integration
    "HTDiffusionLoaderMulti": "HT Multi Model Loader",
    "HTGeminiNode": "HT Gemini"
}

# Try to conditionally import OIDN if available
try:
    from .nodes.ht_oidn_node import HTOIDNNode
    NODE_CLASS_MAPPINGS["HTOIDNNode"] = HTOIDNNode
    NODE_DISPLAY_NAME_MAPPINGS["HTOIDNNode"] = "HT Intel Denoiser"
except ImportError:
    print("Intel OIDN not available - denoising node disabled")

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