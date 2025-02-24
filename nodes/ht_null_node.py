"""
File: homage_tools/nodes/ht_null_node.py
Version: 1.0.0
Description: Node for providing null/empty values to optional inputs
"""

#------------------------------------------------------------------------------
# Section 1: Type Handling Classes
#------------------------------------------------------------------------------
class AnyType(str):
    """Special class for handling any input type."""
    def __ne__(self, __value: object) -> bool:
        return False

# Define universal type
any_type = AnyType("*")

#------------------------------------------------------------------------------
# Section 2: Null Node Definition
#------------------------------------------------------------------------------
class HTNullNode:
    """Provides null/empty values for optional inputs."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "provide_null"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("null_value",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_type": (["IMAGE", "MASK", "MODEL", "VAE", "CLIP", "CONDITIONING"], {
                    "default": "IMAGE"
                })
            }
        }

    def provide_null(self, value_type: str):
        """Return None value with correct type handling."""
        return (None,)