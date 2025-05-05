"""
File: homage_tools/nodes/ht_seed_advanced_node.py
Version: 1.0.0
Description: Advanced seed generator with multiple modes and outputs
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import random
import torch
import hashlib
import logging
from server import PromptServer
from typing import Dict, Any, Tuple, List

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.0.0"

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTSeedAdvancedNode:
    """
    Advanced seed generator with multiple modes, derived seeds, and UI integration.
    Provides extended seeding control for ComfyUI workflows.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_seed"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("seed", "subseed", "iteration_seed", "seed_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
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
                }),
                "seed_mode": (["standard", "fixed", "iter_add", "iter_mult", "derived"], {
                    "default": "standard"
                }),
                "iter_value": ("INT", {
                    "default": 1,
                    "min": -999999,
                    "max": 999999
                }),
                "subseed_strength": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            },
            "optional": {
                "iteration": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999
                }),
                "text_input": ("STRING", {
                    "multiline": False
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

#------------------------------------------------------------------------------
# Section 3: Seed Generation Functions
#------------------------------------------------------------------------------
    def _generate_random_seed(self) -> int:
        """Generate a random seed value."""
        return random.randint(0, 0xffffffffffffffff)
    
    def _generate_derived_seed(self, seed: int, text_input: str = None) -> int:
        """Generate a derived seed based on text input or a default modifier."""
        if not text_input:
            return (seed * 773) % 0xffffffffffffffff
            
        # Create a hash of the text input combined with the seed
        hash_input = f"{seed}:{text_input}"
        hash_obj = hashlib.md5(hash_input.encode())
        # Convert first 16 chars of the hash to an integer
        derived = int(hash_obj.hexdigest()[:16], 16)
        return derived
    
    def _calculate_subseed(self, seed: int, strength: float) -> int:
        """Calculate a subseed with adjustable distance from the main seed."""
        if strength <= 0:
            return seed
            
        # Calculate a related seed with controlled deviation
        subseed_base = (seed * 11937) % 0xffffffffffffffff
        # Interpolate between original seed and subseed_base based on strength
        if strength >= 1.0:
            return subseed_base
            
        # Create a weighted blend of the two seeds
        blend_factor = int(0xffffffffffffffff * strength)
        subseed = (seed & ~blend_factor) | (subseed_base & blend_factor)
        return subseed
        
    def _calculate_iterated_seed(self, seed: int, iteration: int, mode: str, value: int) -> int:
        """Calculate an iterated seed based on iteration number and mode."""
        if mode == "fixed":
            return seed
        elif mode == "iter_add":
            return (seed + (iteration * value)) % 0xffffffffffffffff
        elif mode == "iter_mult":
            return (seed * (1 + (iteration * value / 1000))) % 0xffffffffffffffff
        else:
            # Standard mode - simple addition
            return (seed + iteration) % 0xffffffffffffffff

#------------------------------------------------------------------------------
# Section 4: Server Communication
#------------------------------------------------------------------------------
    def _send_seed_update(self, node_id: str, seed: int) -> None:
        """Send seed update to UI for display."""
        if not node_id:
            return
            
        try:
            PromptServer.instance.send_sync("ht-seed-update", {
                "node_id": node_id,
                "seed": seed
            })
        except Exception as e:
            logger.debug(f"Error sending seed update: {str(e)}")

#------------------------------------------------------------------------------
# Section 5: Processing Logic
#------------------------------------------------------------------------------
    def process_seed(
        self, 
        seed: int, 
        random_seed: bool, 
        force_random_seed: bool,
        seed_mode: str,
        iter_value: int,
        subseed_strength: float,
        iteration: int = 0,
        text_input: str = None,
        unique_id: str = None
    ) -> Tuple[int, int, int, str]:
        """
        Process seed based on selected mode and options.
        
        Args:
            seed: Base seed value
            random_seed: Whether to use a random seed
            force_random_seed: Whether to force a new random seed on each execution
            seed_mode: Seed processing mode
            iter_value: Value to use for iteration calculations
            subseed_strength: Strength of subseed variation (0-1)
            iteration: Current iteration number
            text_input: Optional text to derive seed from
            unique_id: Node ID for UI communication
            
        Returns:
            Tuple[int, int, int, str]: Main seed, subseed, iteration seed, and info string
        """
        try:
            # Step 1: Determine base seed
            main_seed = seed
            if random_seed or force_random_seed:
                main_seed = self._generate_random_seed()
                print(f"Generated random seed: {main_seed}")
            
            # Step 2: Handle special seed modes
            if seed_mode == "derived" and text_input:
                main_seed = self._generate_derived_seed(main_seed, text_input)
                print(f"Generated derived seed from text: {main_seed}")
            
            # Step 3: Calculate iteration seed
            iter_seed = self._calculate_iterated_seed(main_seed, iteration, seed_mode, iter_value)
            
            # Step 4: Calculate subseed
            subseed = self._calculate_subseed(main_seed, subseed_strength)
            
            # Step 5: Generate info string
            if seed_mode == "standard":
                mode_info = "Standard"
            elif seed_mode == "fixed":
                mode_info = "Fixed"
            elif seed_mode == "iter_add":
                mode_info = f"Iteration +{iter_value}"
            elif seed_mode == "iter_mult":
                mode_info = f"Iteration ×{iter_value/1000}"
            else:
                mode_info = "Derived"
                
            info = f"Mode: {mode_info}, Iteration: {iteration}"
            if subseed_strength > 0:
                info += f", Subseed: {subseed_strength:.2f}"
            
            # Step 6: Send update to UI
            self._send_seed_update(unique_id, main_seed)
            
            return (main_seed, subseed, iter_seed, info)
            
        except Exception as e:
            logger.error(f"Error processing seed: {str(e)}")
            # Return original values as fallback
            return (seed, seed, seed, f"Error: {str(e)}")

#------------------------------------------------------------------------------
# Section 6: Execution Control
#------------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Control when the node should be re-evaluated.
        Re-evaluate if force_random_seed is enabled or in iteration modes.
        """
        # Force re-evaluation for random seeds or iteration modes
        if kwargs.get("force_random_seed", False):
            return float("nan")  # Force re-evaluation
            
        # Also re-evaluate for certain seed modes
        seed_mode = kwargs.get("seed_mode", "standard")
        if seed_mode in ["iter_add", "iter_mult"]:
            return float("nan")
            
        return 0.0  # Otherwise, no change