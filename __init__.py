"""
File: ComfyUI-HommageTools/__init__.py
Version: 1.0.2
Description: Entry point for HommageTools node collection for ComfyUI.
             Handles node registration, imports, and logging configuration.

Sections:
1. Module Configuration
2. Node Imports
3. Node Registration
4. Package Exports
"""

#------------------------------------------------------------------------------
# Section 1: Module Configuration
#------------------------------------------------------------------------------
import os
import sys
import logging
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('homage_tools_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('HommageTools')
logger.debug("Initializing HommageTools...")

# Get the absolute path of the nodes directory
NODES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nodes")
logger.debug(f"Nodes directory path: {NODES_DIR}")

if not os.path.exists(NODES_DIR):
    logger.error(f"Nodes directory not found at {NODES_DIR}")
    raise FileNotFoundError(f"Nodes directory not found at {NODES_DIR}")

#------------------------------------------------------------------------------
# Section 2: Node Imports
#------------------------------------------------------------------------------
def load_module(file_path, module_name):
    """
    Load a Python module from file path.
    """
    logger.debug(f"Attempting to load module: {module_name} from {file_path}")
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"Failed to create spec for {module_name}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.debug(f"Successfully loaded module: {module_name}")
        return module
    except Exception as e:
        logger.error(f"Error loading module {module_name}: {str(e)}")
        return None

# Dictionary to track successful imports
imported_nodes = {}

# Node file mapping
NODE_FILES = {
    "HTRegexNode": "ht_regex_node.py",
    "HTParameterExtractorNode": "ht_parameter_extractor.py",
    "HTTextCleanupNode": "ht_text_cleanup_node.py",
    "HTResizeNode": "ht_resize_node.py",
    "HTResolutionNode": "ht_resolution_node.py",
    "HTLevelsNode": "ht_levels_node.py",
    "HTBaseShiftNode": "ht_baseshift_node.py",
    "HTTrainingSizeNode": "ht_training_size_node.py",
    "HTConversionNode": "ht_conversion_node.py",
    "HTSwitchNode": "ht_switch_node.py",
    "HTLayerCollectorNode": "ht_layer_nodes.py",
    "HTLayerExportNode": "ht_layer_nodes.py"
}

logger.debug("Starting node imports...")

# Import all nodes
for node_name, file_name in NODE_FILES.items():
    file_path = os.path.join(NODES_DIR, file_name)
    if not os.path.exists(file_path):
        logger.error(f"Node file not found: {file_path}")
        continue
        
    module = load_module(file_path, file_name[:-3])
    if module is not None:
        try:
            node_class = getattr(module, node_name)
            imported_nodes[node_name] = node_class
            logger.debug(f"Successfully imported {node_name}")
        except AttributeError as e:
            logger.error(f"Failed to get class {node_name} from module: {str(e)}")

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
        
    except Exception as e:
        logger.error(f"Error registering {node_name}: {str(e)}")

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