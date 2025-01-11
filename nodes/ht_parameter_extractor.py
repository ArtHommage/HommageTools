"""
File: homage_tools/nodes/ht_parameter_extractor.py

HommageTools Parameter Extractor Node
Version: 1.0.0
Description: A node that extracts labeled parameter values from text strings using
customizable identifiers and separators, with optional parameter clearing.

Sections:
1. Imports and Types
2. Node Class Definition
3. Parameter Parsing Methods
4. Type Conversion Methods
5. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Types
#------------------------------------------------------------------------------
import re
from typing import Tuple, Dict, Any, Optional, List

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTParameterExtractorNode:
    """
    A ComfyUI node that extracts labeled parameter values from text strings.
    Supports custom identifiers and separators with type conversion.
    
    Features:
    - Customizable parameter identifier and separator
    - Case-insensitive label matching
    - Multiple output types (string, float, int)
    - Parameter removal from source text
    - Optional clearing of all parameters
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "extract_parameter"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("parsed_text", "label", "value_string", "value_float", "value_int")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Define the input types for the node."""
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "Input text containing parameters"
                }),
                "separator": ("STRING", {
                    "default": "&&",
                    "description": "String that separates parameters"
                }),
                "identifier": ("STRING", {
                    "default": "$$",
                    "description": "String that identifies start of parameter"
                }),
                "label": ("STRING", {
                    "default": "",
                    "description": "Label to match in parameters"
                }),
                "clear_parameters": ("BOOLEAN", {
                    "default": False,
                    "description": "Remove all parameters from output text"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 3: Parameter Parsing Methods
#------------------------------------------------------------------------------
    def find_parameter(
        self,
        text: str,
        identifier: str,
        separator: str,
        label: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """
        Find and extract a labeled parameter from text.
        
        Args:
            text: Input text to search
            identifier: String marking parameter start
            separator: String marking parameter end
            label: Label to match (case insensitive)
            
        Returns:
            Tuple containing:
            - Extracted parameter text (or None if not found)
            - Matched label (or None if not found)
            - Parameter value (or None if not found)
            - List of all parameter strings found
        """
        # Escape special regex characters in identifier and separator
        esc_id = re.escape(identifier)
        esc_sep = re.escape(separator)
        
        # Create pattern to match parameter
        pattern = f"{esc_id}([^{esc_sep}]+){esc_sep}"
        
        # Find all parameters
        matches = list(re.finditer(pattern, text))
        all_params = [match.group(0) for match in matches]
        
        # Look for matching parameter
        for match in matches:
            param_text = match.group(1).strip()
            
            # Split into label and value
            if "=" not in param_text:
                continue
                
            param_label, value = param_text.split("=", 1)
            param_label = param_label.strip()
            value = value.strip()
            
            # Check for label match (case insensitive)
            if param_label.lower() == label.lower():
                # Return full match, components, and all params
                return match.group(0), param_label, value, all_params
                
        return None, None, None, all_params

#------------------------------------------------------------------------------
# Section 4: Type Conversion Methods
#------------------------------------------------------------------------------
    def convert_value(self, value: str) -> Tuple[str, float, int]:
        """
        Convert a value string to multiple types.
        
        Args:
            value: String value to convert
            
        Returns:
            Tuple of (string, float, int) representations
        """
        # String conversion (strip whitespace)
        str_val = value.strip()
        
        # Float conversion (default to 0.0 on error)
        try:
            float_val = float(str_val)
        except (ValueError, TypeError):
            float_val = 0.0
            
        # Integer conversion (default to 0 on error)
        try:
            int_val = int(float_val)
        except (ValueError, TypeError):
            int_val = 0
            
        return str_val, float_val, int_val

#------------------------------------------------------------------------------
# Section 5: Main Processing Logic
#------------------------------------------------------------------------------
    def extract_parameter(
        self,
        input_text: str,
        separator: str,
        identifier: str,
        label: str,
        clear_parameters: bool
    ) -> Tuple[str, str, str, float, int]:
        """
        Main processing function to extract and convert parameters.
        
        Args:
            input_text: Text containing parameters
            separator: Parameter separator string
            identifier: Parameter identifier string
            label: Label to match
            clear_parameters: Whether to remove all parameters
            
        Returns:
            Tuple containing:
            - Processed text with parameter(s) removed
            - Matched label
            - Value as string
            - Value as float
            - Value as integer
        """
        try:
            # Find matching parameter and get all parameters
            param_text, found_label, value, all_params = self.find_parameter(
                input_text, identifier, separator, label
            )
            
            # Start with original text
            output_text = input_text
            
            # If clear_parameters is True, remove all parameters
            if clear_parameters:
                for param in all_params:
                    output_text = output_text.replace(param, "")
            # Otherwise, only remove matched parameter if found
            elif param_text:
                output_text = output_text.replace(param_text, "")
            
            # Clean up multiple whitespace
            output_text = " ".join(output_text.split())
            
            # If parameter found, process value outputs
            if param_text and found_label and value:
                # Convert value to all types
                str_val, float_val, int_val = self.convert_value(value)
                return (output_text, found_label, str_val, float_val, int_val)
            
            # If no parameter found, return processed text and defaults
            return (output_text, "", "", 0.0, 0)
            
        except Exception as e:
            print(f"Error in HTParameterExtractorNode: {str(e)}")
            return (input_text, "", "", 0.0, 0)  # Return safe defaults on error