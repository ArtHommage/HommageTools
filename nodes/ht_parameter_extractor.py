"""
File: homage_tools/nodes/ht_parameter_extractor.py
Version: 1.3.0
Description: Robust parameter extractor with comprehensive error handling
"""

import re
from typing import Tuple, Dict, Any, Optional, List, Match

class HTParameterExtractorNode:
    """
    Extracts labeled parameter values from text strings with robust handling
    of multiline text, adjacent parameters, and complex formatting.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "extract_parameter"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("parsed_text", "label", "value_string", "value_float", "value_int")
    
    # Version tracking
    VERSION = "1.3.0"
    
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
                    "default": "%%"
                }),
                "label": ("STRING", {
                    "default": ""
                }),
                "clear_parameters": ("BOOLEAN", {
                    "default": False
                }),
                "debug_mode": ("BOOLEAN", {
                    "default": True
                })
            }
        }

    def _extract_params_with_regex(
        self,
        text: str,
        identifier: str,
        separator: str,
        debug_mode: bool
    ) -> List[Tuple[str, str, str]]:
        """
        Find all parameters using a more robust regex approach.
        
        Args:
            text: Input text to search
            identifier: Parameter identifier string
            separator: Parameter terminator string
            debug_mode: Whether to print debug info
            
        Returns:
            List of tuples (full_parameter_string, label, value)
        """
        results = []
        
        # First normalize newlines
        normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Create a pattern that captures everything between identifier and separator
        # Using re.DOTALL to match across line breaks
        pattern = f"{re.escape(identifier)}(.*?){re.escape(separator)}"
        matches = list(re.finditer(pattern, normalized_text, re.DOTALL))
        
        if debug_mode:
            print(f"Debug: Found {len(matches)} raw parameter matches")
        
        for i, match in enumerate(matches):
            full_param = match.group(0)  # The entire matched text
            param_content = match.group(1).strip()  # Just the content between identifier and separator
            
            # Skip if no equals sign
            if "=" not in param_content:
                if debug_mode:
                    print(f"Debug: Parameter {i+1} has no '=' sign, skipping: '{param_content}'")
                continue
            
            # Split on first equals sign only
            try:
                param_label, value = param_content.split("=", 1)
                param_label = param_label.strip()
                value = value.strip()
                
                if debug_mode:
                    print(f"Debug: Extracted parameter: '{param_label}' = '{value}'")
                
                results.append((full_param, param_label, value))
            except ValueError:
                if debug_mode:
                    print(f"Debug: Error processing parameter content: '{param_content}'")
        
        return results

    def extract_parameter(
        self,
        input_text: str,
        separator: str,
        identifier: str,
        label: str,
        clear_parameters: bool,
        debug_mode: bool = True
    ) -> Tuple[str, str, str, float, int]:
        """
        Extract and convert parameters from text.
        
        Args:
            input_text: Text containing parameters
            separator: Parameter terminator string
            identifier: Parameter identifier string
            label: Parameter label to extract
            clear_parameters: Whether to remove all parameters from output
            debug_mode: Whether to print debug info
            
        Returns:
            Tuple containing (parsed_text, label, value_string, value_float, value_int)
        """
        # Ensure we have input
        if not input_text:
            return ("", "", "", 0.0, 0)
            
        # Find all parameters
        all_params = self._extract_params_with_regex(input_text, identifier, separator, debug_mode)
        
        if debug_mode:
            print(f"Debug: Extracted {len(all_params)} valid parameters")
            for i, (full, param_label, value) in enumerate(all_params):
                print(f"Debug: Parameter {i+1}: '{param_label}' = '{value}'")
        
        # Find the requested parameter by label
        found_param = None
        for full_param, param_label, value in all_params:
            if param_label.lower() == label.lower():
                found_param = (full_param, param_label, value)
                if debug_mode:
                    print(f"Debug: Found requested parameter '{label}' = '{value}'")
                break
        
        # Handle output text
        output_text = input_text
        
        if clear_parameters:
            # Remove all parameters
            for full_param, _, _ in all_params:
                output_text = output_text.replace(full_param, "")
            if debug_mode:
                print(f"Debug: Removed all {len(all_params)} parameters")
        elif found_param:
            # Only remove the found parameter
            full_param, _, _ = found_param
            output_text = output_text.replace(full_param, "")
            if debug_mode:
                print(f"Debug: Removed parameter '{full_param}'")
        
        # Clean up result by normalizing whitespace while preserving intentional line breaks
        lines = [line.strip() for line in output_text.split('\n')]
        output_text = '\n'.join([' '.join(line.split()) for line in lines])
        output_text = output_text.strip()
        
        # Handle conversion of values
        if found_param:
            _, param_label, value = found_param
            try:
                float_val = float(value)
                int_val = int(float_val)
                if debug_mode:
                    print(f"Debug: Converted value '{value}' to float={float_val}, int={int_val}")
            except (ValueError, TypeError):
                float_val = 0.0
                int_val = 0
                if debug_mode:
                    print(f"Debug: Couldn't convert value '{value}' to numbers, using defaults")
            return (output_text, param_label, value, float_val, int_val)
            
        # No parameter found
        if debug_mode:
            print(f"Debug: No parameter with label '{label}' found")
        return (output_text, "", "", 0.0, 0)