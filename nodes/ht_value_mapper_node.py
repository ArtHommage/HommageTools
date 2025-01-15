"""
File: homage_tools/nodes/ht_value_mapper_node.py
Version: 1.2.0
Description: Node for mapping labels to values with flexible input types and boolean interpretation

Sections:
1. Imports and Type Definitions
2. Type Handling Classes
3. Helper Functions
4. Boolean Interpretation
5. Node Class Definition
6. Mapping Logic
7. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
from typing import Dict, Any, Tuple, Optional, Union, Set
import torch

#------------------------------------------------------------------------------
# Section 2: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False

class FlexibleOptionalInputType(dict):
    """Enables flexible/dynamic input types."""
    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type,)

    def __contains__(self, key):
        return True

# Define a universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 3: Helper Functions
#------------------------------------------------------------------------------
def parse_mapping_list(mapping_text: str) -> Dict[str, str]:
    """
    Parse a text list of label:value pairs into a dictionary.
    
    Args:
        mapping_text: Multi-line string with label:value pairs
        
    Returns:
        Dict[str, str]: Mapping of labels to values
    """
    mapping = {}
    for line in mapping_text.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            label, value = line.split(':', 1)
            mapping[label.strip().lower()] = value.strip()
    return mapping

def convert_value(value: str) -> Tuple[str, float, int]:
    """
    Convert a value string to multiple formats.
    
    Args:
        value: String value to convert
        
    Returns:
        Tuple[str, float, int]: Value in string, float, and int formats
    """
    # String is already handled
    str_val = value.strip()
    
    # Try float conversion
    try:
        float_val = float(str_val)
    except (ValueError, TypeError):
        float_val = 0.0
        
    # Try int conversion (from float to handle decimal strings)
    try:
        int_val = int(float_val)
    except (ValueError, TypeError):
        int_val = 0
        
    return str_val, float_val, int_val

def extract_label_from_input(input_value: Any) -> str:
    """
    Extract a label string from any input type.
    
    Args:
        input_value: Input of any type
        
    Returns:
        str: Extracted label string
    """
    try:
        if isinstance(input_value, torch.Tensor):
            # Handle tensor inputs
            if input_value.numel() == 1:
                # Single value tensor
                return str(input_value.item())
            else:
                # Get first value from tensor
                return str(input_value.flatten()[0].item())
        elif isinstance(input_value, (list, tuple)):
            # Handle list/tuple inputs
            return str(input_value[0]) if input_value else ""
        elif isinstance(input_value, dict):
            # Handle dictionary inputs
            return str(next(iter(input_value.values()))) if input_value else ""
        else:
            # Handle all other types
            return str(input_value)
    except Exception as e:
        print(f"Error extracting label from input: {str(e)}")
        return ""

#------------------------------------------------------------------------------
# Section 4: Boolean Interpretation
#------------------------------------------------------------------------------
class BooleanInterpreter:
    """Interprets strings as boolean values using common terms."""
    
    # Sets of words indicating true/false values
    POSITIVE_WORDS: Set[str] = {
        'yes', 'true', 'positive', 'on', 'enable', 'enabled',
        '1', 't', 'y', 'ok', 'right', 'correct', 'confirmed',
        'active', 'activated', 'valid', 'approved', 'accept',
        'allowed', 'permit', 'permitted', 'good'
    }
    
    NEGATIVE_WORDS: Set[str] = {
        'no', 'false', 'negative', 'off', 'disable', 'disabled',
        '0', 'f', 'n', 'wrong', 'incorrect', 'denied', 'reject',
        'inactive', 'deactivated', 'invalid', 'disapproved',
        'forbidden', 'denied', 'bad', 'none', 'null'
    }
    
    @classmethod
    def interpret(cls, value: str) -> Tuple[bool, int]:
        """
        Interpret a string value as boolean and corresponding integer.
        
        Args:
            value: String to interpret
            
        Returns:
            Tuple[bool, int]: Boolean interpretation and corresponding integer (1 or 0)
        """
        # Clean and normalize input
        cleaned = value.lower().strip()
        
        # First check if it's a direct number
        try:
            num_val = float(cleaned)
            return bool(num_val), 1 if num_val else 0
        except ValueError:
            pass
            
        # Check for positive words
        if cleaned in cls.POSITIVE_WORDS:
            return True, 1
            
        # Check for negative words
        if cleaned in cls.NEGATIVE_WORDS:
            return False, 0
            
        # Default case - treat non-empty strings as True
        return bool(cleaned), 1 if cleaned else 0

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
#------------------------------------------------------------------------------
class HTValueMapperNode:
    """
    Maps input labels to values using a configurable mapping list.
    Supports any input type and converts to multiple output formats including boolean.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "map_value"
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES = ("value_string", "value_float", "value_int", "bool_value", "bool_int")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "mapping_list": ("STRING", {
                    "multiline": True,
                    "default": "small: 512\nmedium: 768\nlarge: 1024",
                    "description": "List of label:value pairs"
                })
            },
            "optional": {
                "input_value": (any_type,)  # Accept any input type
            }
        }

#------------------------------------------------------------------------------
# Section 6: Mapping Logic
#------------------------------------------------------------------------------
    def map_value(
        self,
        mapping_list: str,
        input_value: Any = None
    ) -> Tuple[str, float, int, bool, int]:
        """
        Map an input value to its corresponding value using the mapping list
        and interpret as boolean when applicable.
        
        Args:
            mapping_list: Multi-line string containing label:value pairs
            input_value: Input of any type to look up in the mapping
            
        Returns:
            Tuple[str, float, int, bool, int]: Mapped values in multiple formats
        """
        try:
            # Parse mapping list
            mapping = parse_mapping_list(mapping_list)
            
            # Extract label from input value
            if input_value is not None:
                input_label = extract_label_from_input(input_value)
            else:
                input_label = ""
            
            # Look up value
            input_label = input_label.strip().lower()
            if input_label in mapping:
                value_str, value_float, value_int = convert_value(mapping[input_label])
            else:
                # If no mapping found, use input directly
                value_str = input_label
                value_float = 0.0
                value_int = 0
            
            # Interpret boolean value
            bool_value, bool_int = BooleanInterpreter.interpret(value_str)
            
            return (value_str, value_float, value_int, bool_value, bool_int)
            
        except Exception as e:
            print(f"Error in HTValueMapperNode: {str(e)}")
            return ("", 0.0, 0, False, 0)