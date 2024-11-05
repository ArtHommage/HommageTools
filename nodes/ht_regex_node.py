"""
Regex Parser Node for HommageTools
"""

import re

class HTRegexNode:
    """
    A node that performs regex parsing on input text based on a provided pattern.
    """
    
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parsed_text",)
    FUNCTION = "parse_text"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "regex_pattern": ("STRING", {"default": ".*"})
            }
        }
        
    def parse_text(self, input_text: str, regex_pattern: str) -> tuple:
        try:
            pattern = re.compile(regex_pattern)
            matches = pattern.findall(input_text)
            
            if len(matches) == 0:
                result = input_text
            elif len(matches) == 1:
                result = str(matches[0])
            else:
                result = "\n".join(str(match) for match in matches)
                
            return (result,)
            
        except re.error as e:
            print(f"RegEx Error in HTRegexNode: {str(e)}")
            return (input_text,)
        except Exception as e:
            print(f"Unexpected error in HTRegexNode: {str(e)}")
            return (input_text,)