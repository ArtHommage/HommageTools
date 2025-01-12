"""
File: ComfyUI-HommageTools/__init__.py
Version: 1.0.1
Description: Entry point for HommageTools node collection for ComfyUI.
             Handles node registration, imports, and logging configuration.
             Enhanced with detailed debugging output.

Sections:
1. Module Configuration
2. Node Imports
3. Node Registration
4. Package Exports
"""

#------------------------------------------------------------------------------
# Section 1: Module Configuration
#------------------------------------------------------------------------------
import logging
import sys
from pathlib import Path
import traceback

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('homage_tools_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('HommageTools')
logger.debug("Initializing HommageTools...")

# Add nodes directory to Python path
NODES_DIR = Path(__file__).parent / "nodes"
logger.debug(f"Checking nodes directory: {NODES_DIR}")

if not NODES_DIR.exists():
    logger.error(f"Nodes directory not found at {NODES_DIR}")
    raise FileNotFoundError(f"Nodes directory not found at {NODES_DIR}")

logger.debug(f"Found nodes directory. Contents: {[f.name for f in NODES_DIR.iterdir() if f.is_file()]}")

#------------------------------------------------------------------------------
# Section 2: Node Imports
#------------------------------------------------------------------------------
# Dictionary to track successful imports
imported_nodes = {}

def import_node(node_name, import_path):
    """Helper function to import nodes with detailed error tracking"""
    logger.debug(f"Attempting to import {node_name} from {import_path}")
    try:
        module_path = f".nodes.{import_path}"
        module = __import__(module_path, fromlist=['*'], globals=globals())
        node_class = getattr(module, node_name)
        imported_nodes[node_name] = node_class
        logger.debug(f"Successfully imported {node_name}")
        return True
    except ImportError as e:
        logger.error(f"ImportError while loading {node_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    except AttributeError as e:
        logger.error(f"AttributeError while loading {node_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"Unexpected error while loading {node_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

logger.debug("Starting node imports...")

# Text Processing Nodes
import_node("HTRegexNode", "ht_regex_node")
import_node("HTParameterExtractorNode", "ht_parameter_extractor")
import_node("HTTextCleanupNode", "ht_text_cleanup_node")

# Image Processing Nodes
import_node("HTResizeNode", "ht_resize_node")
import_node("HTResolutionNode", "ht_resolution_node")
import_node("HTLevelsNode", "ht_levels_node")
import_node("HTBaseShiftNode", "ht_baseshift_node")
import_node("HTTrainingSizeNode", "ht_training_size_node")

# Utility Nodes
import_node("HTConversionNode", "ht_conversion_node")
import_node("HTSwitchNode", "ht_switch_node")

# Layer Management Nodes
import_node("HTLayerCollectorNode", "ht_layer_nodes")
import_node("HTLayerExportNode", "ht_layer_nodes")

logger.debug(f"Import results: {list(imported_nodes.keys())}")

#------------------------------------------------------------------------------
# Section 3: Node Registration
#------------------------------------------------------------------------------
logger.debug("Starting node registration...")

# Map node classes to their internal names
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Only register successfully imported nodes
for node_name, node_class in imported_nodes.items():
    try:
        logger.debug(f"Registering {node_name}")
        
        # Verify node class has required attributes
        required_attrs = ['CATEGORY', 'INPUT_TYPES', 'RETURN_TYPES']
        missing_attrs = [attr for attr in required_attrs if not hasattr(node_class, attr)]
        
        if missing_attrs:
            logger.error(f"{node_name} missing required attributes: {missing_attrs}")
            continue
            
        # Register the node
        NODE_CLASS_MAPPINGS[node_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = f"HT {' '.join(node_name.split('HT')[1].split('Node')[0].split('_'))}"
        
        logger.debug(f"Successfully registered {node_name}")
        logger.debug(f"Category: {node_class.CATEGORY}")
        logger.debug(f"Input Types: {node_class.INPUT_TYPES()}")
        logger.debug(f"Return Types: {node_class.RETURN_TYPES}")
        
    except Exception as e:
        logger.error(f"Error registering {node_name}: {str(e)}")
        logger.error(traceback.format_exc())

logger.debug(f"Final NODE_CLASS_MAPPINGS: {list(NODE_CLASS_MAPPINGS.keys())}")
logger.debug(f"Final NODE_DISPLAY_NAME_MAPPINGS: {NODE_DISPLAY_NAME_MAPPINGS}")

#------------------------------------------------------------------------------
# Section 4: Package Exports
#------------------------------------------------------------------------------
# These mappings will be imported by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

if len(NODE_CLASS_MAPPINGS) > 0:
    logger.info(f"HommageTools successfully registered {len(NODE_CLASS_MAPPINGS)} nodes")
else:
    logger.error("No nodes were successfully registered!")
    