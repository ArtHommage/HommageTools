"""
File: homage_tools/nodes/ht_seed_node.py
Version: 1.0.0
Description: Simple seed generator node with random seed capabilities
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import random
import torch
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.0.0"

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTSeedNode:
    """
    A seed generator node with random seed capabilities and seed forcing.
    Provides consistent seeding interface for ComfyUI workflows.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_seed"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "description": "Base seed value"
                }),
                "random_seed": ("BOOLEAN", {
                    "default": False, 
                    "label": "Random Seed",
                    "description": "Use a random seed instead of the provided value"
                }),
                "force_random_seed": ("BOOLEAN", {
                    "default": False, 
                    "label": "Force New Random",
                    "description": "Generate a new random seed on each execution"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 3: Processing Logic
#------------------------------------------------------------------------------
    def process_seed(self, seed: int, random_seed: bool, force_random_seed: bool) -> tuple:
        """
        Process seed value based on randomization settings.
        
        Args:
            seed: Base seed value
            random_seed: Whether to use a random seed
            force_random_seed: Whether to force a new random seed on each execution
            
        Returns:
            tuple: Tuple containing the processed seed value
        """
        try:
            # Generate random seed if requested
            if random_seed or force_random_seed:
                print(f"Generating random seed (previous: {seed})")
                generated_seed = random.randint(0, 0xffffffffffffffff)
                print(f"Generated seed: {generated_seed}")
                return (generated_seed,)
            
            # Use provided seed
            return (seed,)
            
        except Exception as e:
            logger.error(f"Error processing seed: {str(e)}")
            # Return original seed as fallback
            return (seed,)

#------------------------------------------------------------------------------
# Section 4: Execution Control
#------------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Control when the node should be re-evaluated.
        Only re-evaluate if force_random_seed is enabled.
        """
        # Re-evaluate if force_random_seed is True
        if kwargs.get("force_random_seed", False):
            return float("nan")  # Force re-evaluation
        return 0.0  # Otherwise, no change