"""
File: homage_tools/nodes/__init__.py

Node implementations for HommageTools
"""

import logging

# Configure logging (adjust level as needed)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.debug("Entering homage_tools/nodes/__init__.py")

try:
    logger.debug("Importing ht_regex_node")
    from .ht_regex_node import HTRegexNode
    logger.debug("Imported ht_regex_node")
except ImportError as e:
    logger.exception(f"Error importing ht_regex_node: {e}")
    raise

try:
    logger.debug("Importing ht_resize_node")
    from .ht_resize_node import HTResizeNode
    logger.debug("Imported ht_resize_node")
except ImportError as e:
    logger.exception(f"Error importing ht_resize_node: {e}")
    raise

try:
    logger.debug("Importing ht_resolution_node")
    from .ht_resolution_node import HTResolutionNode
    logger.debug("Imported ht_resolution_node")
except ImportError as e:
    logger.exception(f"Error importing ht_resolution_node: {e}")
    raise

try:
    logger.debug("Importing ht_conversion_node")
    from .ht_conversion_node import HTConversionNode
    logger.debug("Imported ht_conversion_node")
except ImportError as e:
    logger.exception(f"Error importing ht_conversion_node: {e}")
    raise

try:
    logger.debug("Importing ht_switch_node")
    from .ht_switch_node import HTSwitchNode
    logger.debug("Imported ht_switch_node")
except ImportError as e:
    logger.exception(f"Error importing ht_switch_node: {e}")
    raise

try:
    logger.debug("Importing ht_parameter_extractor")
    from .ht_parameter_extractor import HTParameterExtractorNode
    logger.debug("Imported ht_parameter_extractor")
except ImportError as e:
    logger.exception(f"Error importing ht_parameter_extractor: {e}")
    raise

try:
    logger.debug("Importing ht_text_cleanup_node")
    from .ht_text_cleanup_node import HTTextCleanupNode
    logger.debug("Imported ht_text_cleanup_node")
except ImportError as e:
    logger.exception(f"Error importing ht_text_cleanup_node: {e}")
    raise

try:
    logger.debug("Importing ht_baseshift_node")
    from .ht_baseshift_node import HTBaseShiftNode
    logger.debug("Imported ht_baseshift_node")
except ImportError as e:
    logger.exception(f"Error importing ht_baseshift_node: {e}")
    raise

try:
    logger.debug("Importing ht_levels_node")
    from .ht_levels_node import HTLevelsNode
    logger.debug("Imported ht_levels_node")
except ImportError as e:
    logger.exception(f"Error importing ht_levels_node: {e}")
    raise

try:
    logger.debug("Importing ht_layer_nodes")
    from .ht_layer_nodes import HTLayerCollectorNode, HTLayerExportNode
    logger.debug("Imported ht_layer_nodes")
except ImportError as e:
    logger.exception(f"Error importing ht_layer_nodes: {e}")
    raise

try:
    logger.debug("Importing ht_training_size_node")
    from .ht_training_size_node import HTTrainingSizeNode
    logger.debug("Imported ht_training_size_node")
except ImportError as e:
    logger.exception(f"Error importing ht_training_size_node: {e}")
    raise

try:
    logger.debug("Importing ht_dimension_formatter_node")
    from .ht_dimension_formatter_node import HTDimensionFormatterNode
    logger.debug("Imported ht_dimension_formatter_node")
except ImportError as e:
    logger.exception(f"Error importing ht_dimension_formatter_node: {e}")
    raise

__all__ = [
    'HTRegexNode',
    'HTResizeNode',
    'HTResolutionNode',
    'HTConversionNode',
    'HTSwitchNode',
    'HTParameterExtractorNode',
    'HTTextCleanupNode',
    'HTBaseShiftNode',
    'HTLevelsNode',
    'HTLayerCollectorNode',
    'HTLayerExportNode',
    'HTTrainingSizeNode',
    'HTDimensionFormatterNode'
]

logger.debug("Exiting homage_tools/nodes/__init__.py")