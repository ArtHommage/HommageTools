"""
File: homage_tools/nodes/ht_diffusion_loader_multi.py
Version: 1.1.0
Description: Node for loading multiple diffusion models from a text list with metadata extraction

Sections:
1. Imports and Type Definitions
2. Helper Functions
3. Node Class Definition
4. Model Loading Logic
5. Metadata Extraction
6. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import os
import torch
import hashlib
import json
import safetensors.torch
from typing import Dict, Any, Tuple, List, Optional
from collections import OrderedDict

import folder_paths
import comfy.sd
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def parse_model_list(model_list_text: str) -> List[str]:
    """Parse text input into list of model names."""
    return [name.strip() for name in model_list_text.split('\n') if name.strip()]

def validate_model_name(model_name: str) -> bool:
    """Validate if model name exists in checkpoints folder."""
    available_models = folder_paths.get_filename_list("checkpoints")
    return model_name in available_models

def calculate_model_hash(filepath: str) -> str:
    """
    Calculate SHA256 hash of model file compatible with CIVITAI.
    
    Args:
        filepath: Path to model file
        
    Returns:
        str: SHA256 hash of model
    """
    try:
        with open(filepath, 'rb') as f:
            hash_obj = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash: {str(e)}")
        return "Hash calculation failed"

def extract_model_metadata(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from model file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Dict[str, Any]: Extracted metadata
    """
    metadata = OrderedDict()
    try:
        # Handle SafeTensors format
        if filepath.endswith('.safetensors'):
            tensors = safetensors.torch.load_file(filepath, device="cpu")
            if isinstance(tensors, dict) and "_metadata" in tensors:
                meta = tensors["_metadata"]
                if isinstance(meta, dict):
                    metadata["format"] = "safetensors"
                    metadata.update(meta)

        # Extract any embedded network metadata
        try:
            state_dict = comfy.sd.load_torch_file(filepath, safe_load=True)
            if isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                # Look for standard metadata fields
                meta_fields = ["modelspec.architecture", "modelspec.description", 
                             "modelspec.trigger", "ss_dataset_dirs", "ss_tag_frequency"]
                
                for key in state_dict:
                    if any(field in key.lower() for field in meta_fields):
                        clean_key = key.split('.')[-1].replace('_', ' ').title()
                        metadata[clean_key] = state_dict[key]
                        
        except Exception as e:
            logger.warning(f"Non-critical error reading network metadata: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        metadata["error"] = str(e)
    
    return metadata

def format_model_info(model_name: str, model_path: str, metadata: Dict[str, Any]) -> str:
    """
    Format model information into readable text.
    
    Args:
        model_name: Name of the model file
        model_path: Full path to model file
        metadata: Extracted metadata dictionary
        
    Returns:
        str: Formatted model information
    """
    info = []
    info.append(f"Model Name: {model_name}")
    info.append(f"Model Hash: {calculate_model_hash(model_path)}")
    info.append(f"Format: {metadata.get('format', 'Unknown')}")
    
    # Add any available metadata in a clean format
    if metadata:
        info.append("\nMetadata:")
        for key, value in metadata.items():
            if key not in ["format", "error"]:
                # Clean up the value for display
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                info.append(f"  {key}: {value}")
                
    if "error" in metadata:
        info.append(f"\nMetadata Error: {metadata['error']}")
        
    return "\n".join(info)

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTDiffusionLoaderMulti:
    """
    Loads multiple diffusion models from a text list.
    Processes one model at a time and outputs both the model and its metadata.
    """
    
    CATEGORY = "HommageTools/loaders"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_name", "model_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model_list": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter model names (one per line)"
                }),
                "current_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number"
                }),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default"
                })
            }
        }

    #--------------------------------------------------------------------------
    # Section 4: Model Loading Logic
    #--------------------------------------------------------------------------
    def load_model(
        self,
        model_list: str,
        current_index: int,
        weight_dtype: str = "default"
    ) -> Tuple[Any, str, str]:
        """
        Load model at specified index from the list.
        
        Args:
            model_list: Text list of model names
            current_index: Index of model to load
            weight_dtype: Weight data type for model loading
            
        Returns:
            Tuple[Any, str, str]: (loaded_model, model_name, model_info)
        """
        try:
            # Parse model list
            models = parse_model_list(model_list)
            
            # Validate index
            if not models:
                raise ValueError("Model list is empty")
            
            # Handle index wrapping
            actual_index = current_index % len(models)
            model_name = models[actual_index]
            
            # Validate model exists
            if not validate_model_name(model_name):
                raise ValueError(f"Model not found: {model_name}")
                
            # Set up model options
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            
            # Load model
            print(f"\nLoading model {actual_index + 1} of {len(models)}: {model_name}")
            model_path = folder_paths.get_full_path_or_raise("checkpoints", model_name)
            
            # Extract metadata before loading model
            metadata = extract_model_metadata(model_path)
            model_info = format_model_info(model_name, model_path, metadata)
            
            # Load the actual model
            model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)
            
            return (model, model_name, model_info)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            error_info = f"Error loading model:\n{str(e)}"
            return (None, f"Error: {str(e)}", error_info)