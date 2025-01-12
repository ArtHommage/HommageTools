"""
File: homage_tools/nodes/ht_parameter_extractor.py
Version: 1.0.0
Description: Node for extracting labeled parameters from text
"""

import re
from typing import Tuple, Dict, Any, Optional, List

class HTParameterExtractorNode:
    """
    Extracts labeled parameter values from text strings with custom identifiers.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "extract_parameter"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("parsed_text", "label", "value_string", "value_float", "value_int")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "separator": ("STRING", {
                    "default": "&&"
                }),
                "identifier": ("STRING", {
                    "default": "$$"
                }),
                "label": ("STRING", {
                    "default": ""
                }),
                "clear_parameters": ("BOOLEAN", {
                    "default": False
                })
            }
        }

    def _find_parameter(
        self,
        text: str,
        identifier: str,
        separator: str,
        label: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Find and extract a labeled parameter."""
        pattern = f"{re.escape(identifier)}([^{re.escape(separator)}]+){re.escape(separator)}"
        matches = list(re.finditer(pattern, text))
        all_params = [match.group(0) for match in matches]
        
        for match in matches:
            param_text = match.group(1).strip()
            if "=" not in param_text:
                continue
            param_label, value = param_text.split("=", 1)
            if param_label.strip().lower() == label.lower():
                return match.group(0), param_label.strip(), value.strip(), all_params
                
        return None, None, None, all_params

    def extract_parameter(
        self,
        input_text: str,
        separator: str,
        identifier: str,
        label: str,
        clear_parameters: bool
    ) -> Tuple[str, str, str, float, int]:
        """Extract and convert parameters."""
        param_text, found_label, value, all_params = self._find_parameter(
            input_text, identifier, separator, label
        )
        
        output_text = input_text
        if clear_parameters:
            for param in all_params:
                output_text = output_text.replace(param, "")
        elif param_text:
            output_text = output_text.replace(param_text, "")
        
        output_text = " ".join(output_text.split())
        
        if value:
            try:
                float_val = float(value)
                int_val = int(float_val)
            except (ValueError, TypeError):
                float_val = 0.0
                int_val = 0
            return (output_text, found_label, value, float_val, int_val)
            
        return (output_text, "", "", 0.0, 0)