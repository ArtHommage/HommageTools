"""
HommageTools Resolution Recommender Node
Version: 1.0.0
Description: A node that analyzes image dimensions and recommends optimal resolutions based on 
standard sizes or custom resolution lists.

Sections:
1. Imports and Setup
2. Node Class Definition
3. Resolution Analysis Methods
4. Main Processing Logic
"""

import math
import re
from typing import List, Tuple, Dict, Optional, Union
import torch

class HTResolutionNode:
    """
    Recommends optimal resolutions for images based on various quality priorities
    and standard or custom resolution lists.
    """
    
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("width", "height", "scale_factor", "crop_pad_values", "info")
    FUNCTION = "recommend_resolution"

    # Default resolution list (width, height)
    DEFAULT_RESOLUTIONS = [
        (1408, 1408), (1728, 1152), (1664, 1216), (1920, 1088),
        (2176, 960), (1024, 1024), (1216, 832), (1152, 896),
        (1344, 768), (1536, 640), (320, 320), (384, 256),
        (448, 320), (448, 256), (576, 256)
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """
        Define input types and their specifications.
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to analyze"}),
                "priority_mode": (["minimize_loss", "minimize_noise", "auto_decide"], {
                    "default": "auto_decide",
                    "tooltip": "Priority for resolution selection"
                }),
                "use_standard_list": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use standard resolution list or custom"
                }),
                "mode": (["crop", "pad"], {
                    "default": "crop",
                    "tooltip": "Method to handle dimension differences"
                }),
            },
            "optional": {
                "custom_resolutions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "1024x1024, 1920x1080...",
                    "tooltip": "Custom resolution list (comma or newline separated)"
                }),
            }
        }

    def parse_resolution_list(self, text: str) -> List[Tuple[int, int]]:
        """Parse a resolution list from string format."""
        resolutions = []
        
        # Replace any whitespace around commas with just comma
        text = re.sub(r'\s*,\s*', ',', text.strip())
        
        # Split on either commas or newlines
        items = text.split(',') if ',' in text else text.splitlines()
        
        for item in items:
            item = item.strip()
            if not item:  # Skip empty lines
                continue
                
            # Validate format
            if not re.match(r'^\d+x\d+$', item):
                print(f"Warning: Skipping invalid resolution format: {item}")
                continue
                
            try:
                w, h = map(int, item.split('x'))
                # Add both orientations
                resolutions.append((w, h))
                if w != h:  # Don't add duplicate for square resolutions
                    resolutions.append((h, w))
            except ValueError:
                print(f"Warning: Couldn't parse resolution values: {item}")
                continue
                
        return resolutions

    def get_aspect_ratio_str(self, width: int, height: int) -> str:
        """Calculate and return simplified aspect ratio as string."""
        gcd = math.gcd(width, height)
        simple_w = width // gcd
        simple_h = height // gcd
        return f"{simple_w}:{simple_h}"

    def calculate_quality_metrics(
        self,
        original_w: int,
        original_h: int,
        target_w: int,
        target_h: int,
        mode: str
    ) -> Dict[str, float]:
        """Calculate quality impact metrics for the transformation."""
        original_pixels = original_w * original_h
        target_pixels = target_w * target_h
        
        if mode == "crop":
            scale_factor = min(target_w / original_w, target_h / original_h)
            scaled_w = int(original_w * scale_factor)
            scaled_h = int(original_h * scale_factor)
            pixels_lost = (scaled_w * scaled_h) - (target_w * target_h)
            pixel_loss_percent = (pixels_lost / (scaled_w * scaled_h)) * 100
        else:  # pad
            scale_factor = max(target_w / original_w, target_h / original_h)
            scaled_w = int(original_w * scale_factor)
            scaled_h = int(original_h * scale_factor)
            pixel_loss_percent = 0
            
        return {
            "scale_factor": scale_factor,
            "pixel_loss_percent": pixel_loss_percent,
            "target_pixels": target_pixels
        }

    def calculate_score(
        self,
        original_w: int,
        original_h: int,
        target_w: int,
        target_h: int,
        priority_mode: str,
        mode: str
    ) -> float:
        """Calculate score for a potential resolution based on priority mode."""
        metrics = self.calculate_quality_metrics(
            original_w, original_h, target_w, target_h, mode
        )
        
        # Base weights
        PIXEL_LOSS_WEIGHT = 1.0
        SCALE_WEIGHT = 1.0
        
        # Adjust weights based on priority mode
        if priority_mode == "minimize_loss":
            PIXEL_LOSS_WEIGHT = 2.0
            SCALE_WEIGHT = 0.5
        elif priority_mode == "minimize_noise":
            PIXEL_LOSS_WEIGHT = 0.5
            SCALE_WEIGHT = 2.0
        # auto_decide uses base weights
        
        # Calculate individual scores
        scale_score = abs(1 - metrics["scale_factor"]) * SCALE_WEIGHT
        pixel_loss_score = (metrics["pixel_loss_percent"] / 100) * PIXEL_LOSS_WEIGHT
        
        # Penalties
        if metrics["scale_factor"] > 1.5:
            scale_score *= (metrics["scale_factor"] - 1.5) ** 2
        if metrics["pixel_loss_percent"] > 10:
            pixel_loss_score *= (metrics["pixel_loss_percent"] / 10)
            
        return scale_score + pixel_loss_score

    def recommend_resolution(
        self,
        image: torch.Tensor,
        priority_mode: str,
        use_standard_list: bool,
        mode: str,
        custom_resolutions: str = ""
    ) -> Tuple[int, int, float, str, str]:
        """
        Main function to recommend resolution based on inputs.
        
        Returns:
            Tuple[int, int, float, str, str]: (width, height, scale_factor, crop_pad_values, info_string)
        """
        try:
            # Get input image dimensions
            in_h, in_w = image.shape[-2:]
            
            # Get resolution list
            if use_standard_list or not custom_resolutions.strip():
                resolutions = self.DEFAULT_RESOLUTIONS
            else:
                resolutions = self.parse_resolution_list(custom_resolutions)
                if not resolutions:  # If parsing failed, use defaults
                    print("Warning: Using default resolutions due to parsing failure")
                    resolutions = self.DEFAULT_RESOLUTIONS
            
            # Find best match
            best_match = None
            best_score = float('inf')
            best_metrics = None
            
            for w, h in resolutions:
                score = self.calculate_score(
                    in_w, in_h, w, h,
                    priority_mode, mode
                )
                
                if score < best_score:
                    best_score = score
                    best_match = (w, h)
                    best_metrics = self.calculate_quality_metrics(
                        in_w, in_h, w, h, mode
                    )
            
            if not best_match:
                return (in_w, in_h, 1.0, "No change needed", "No changes required")
                
            # Calculate padding/cropping values
            scale_factor = best_metrics["scale_factor"]
            scaled_w = int(in_w * scale_factor)
            scaled_h = int(in_h * scale_factor)
            
            if mode == "crop":
                crop_w = max(0, scaled_w - best_match[0]) // 2
                crop_h = max(0, scaled_h - best_match[1]) // 2
                crop_pad_str = f"Crop: left/right={crop_w}, top/bottom={crop_h}"
            else:
                pad_w = max(0, best_match[0] - scaled_w) // 2
                pad_h = max(0, best_match[1] - scaled_h) // 2
                crop_pad_str = f"Pad: left/right={pad_w}, top/bottom={pad_h}"
            
            # Create info string
            info = (
                f"Input: {in_w}x{in_h} ({self.get_aspect_ratio_str(in_w, in_h)})\n"
                f"Output: {best_match[0]}x{best_match[1]} "
                f"({self.get_aspect_ratio_str(*best_match)})\n"
                f"Scale: {scale_factor:.3f}x\n"
                f"Quality: {100-best_metrics['pixel_loss_percent']:.1f}% preserved"
            )
            
            return (best_match[0], best_match[1], float(scale_factor), crop_pad_str, info)
            
        except Exception as e:
            print(f"Error in HTResolutionNode: {str(e)}")
            return (512, 512, 1.0, "Error occurred", str(e))