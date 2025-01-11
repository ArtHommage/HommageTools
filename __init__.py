"""
File: ComfyUI-HommageTools/__init__.py

HommageTools Node Collection for ComfyUI
Version: 1.0.0
Description: A collection of utility nodes for ComfyUI
"""

import logging

# Configure logging (adjust level as needed)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug("Entering __init__.py")

try:
    logger.debug("Importing ht_regex_node")
    from .nodes.ht_regex_node import HTRegexNode
    logger.debug("Imported ht_regex_node")
except ImportError as e:
    logger.exception(f"Error importing ht_regex_node: {e}")
    raise  # Re-raise the exception after logging

try:
    logger.debug("Importing ht_resize_node")
    from .nodes.ht_resize_node import HTResizeNode
    logger.debug("Imported ht_resize_node")
except ImportError as e:
    logger.exception(f"Error importing ht_resize_node: {e}")
    raise

try:
    logger.debug("Importing ht_resolution_node")
    from .nodes.ht_resolution_node import HTResolutionNode
    logger.debug("Imported ht_resolution_node")
except ImportError as e:
    logger.exception(f"Error importing ht_resolution_node: {e}")
    raise

try:
    logger.debug("Importing ht_conversion_node")
    from .nodes.ht_conversion_node import HTConversionNode
    logger.debug("Imported ht_conversion_node")
except ImportError as e:
    logger.exception(f"Error importing ht_conversion_node: {e}")
    raise

try:
    logger.debug("Importing ht_switch_node")
    from .nodes.ht_switch_node import HTSwitchNode
    logger.debug("Imported ht_switch_node")
except ImportError as e:
    logger.exception(f"Error importing ht_switch_node: {e}")
    raise

try:
    logger.debug("Importing ht_parameter_extractor")
    from .nodes.ht_parameter_extractor import HTParameterExtractorNode
    logger.debug("Imported ht_parameter_extractor")
except ImportError as e:
    logger.exception(f"Error importing ht_parameter_extractor: {e}")
    raise

try:
    logger.debug("Importing ht_text_cleanup_node")
    from .nodes.ht_text_cleanup_node import HTTextCleanupNode
    logger.debug("Imported ht_text_cleanup_node")
except ImportError as e:
    logger.exception(f"Error importing ht_text_cleanup_node: {e}")
    raise

try:
    logger.debug("Importing ht_baseshift_node")
    from .nodes.ht_baseshift_node import HTBaseShiftNode
    logger.debug("Imported ht_baseshift_node")
except ImportError as e:
    logger.exception(f"Error importing ht_baseshift_node: {e}")
    raise

try:
    logger.debug("Importing ht_levels_node")
    from .nodes.ht_levels_node import HTLevelsNode
    logger.debug("Imported ht_levels_node")
except ImportError as e:
    logger.exception(f"Error importing ht_levels_node: {e}")
    raise

try:
    logger.debug("Importing ht_layer_nodes")
    from .nodes.ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode
    logger.debug("Imported ht_layer_nodes")
except ImportError as e:
    logger.exception(f"Error importing ht_layer_nodes: {e}")
    raise

try:
    logger.debug("Importing ht_training_size_node")
    from .nodes.ht_training_size_node import HTTrainingSizeNode
    logger.debug("Imported ht_training_size_node")
except ImportError as e:
    logger.exception(f"Error importing ht_training_size_node: {e}")
    raise

try:
    logger.debug("Importing ht_dimension_formatter_node")
    from .nodes.ht_dimension_formatter_node import HTDimensionFormatterNode
    logger.debug("Imported ht_dimension_formatter_node")
except ImportError as e:
    logger.exception(f"Error importing ht_dimension_formatter_node: {e}")
    raise

logger.debug("Building NODE_CLASS_MAPPINGS")
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
logger.debug("Finished building NODE_CLASS_MAPPINGS")

logger.debug("Building NODE_DISPLAY_NAME_MAPPINGS")
NODE_DISPLAY_NAME_MAPPINGS = {
    "HTRegexNode": "HT Regex Parser",
    "HTResizeNode": "HT Smart Resize",
    "HTResolutionNode": "HT Resolution Recommender",
    "HTPauseNode": "HT Pause Workflow", #This was present in your original code. I left it.
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
logger.debug("Finished building NODE_DISPLAY_NAME_MAPPINGS")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

logger.debug("Exiting __init__.py")