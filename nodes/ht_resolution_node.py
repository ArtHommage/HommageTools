"""
File: homage_tools/nodes/ht_resolution_node.py
Version: 1.0.0
Description: Node for recommending optimal image resolutions
"""

from typing import List, Tuple, Dict, Union, Optional
import torch

class HTResolutionNode:
    """Recommends optimal resolutions for images."""
    
    CATEGORY = "HommageTools"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("width", "height", "scale_factor", "crop_pad_values", "info", "pad_left_right", "pad_top_bottom")
    FUNCTION = "recommend_resolution"

    DEFAULT_RESOLUTIONS = [
        # Square
        (512, 512), (768, 768), (1024, 1024), (1408, 1408),
        # Landscape 16:9
        (1024, 576), (1152, 648), (1280, 720), (1408, 792),
        (1536, 864), (1664, 936), (1792, 1008), (1920, 1080),
        # Portrait 9:16
        (576, 1024), (648, 1152), (720, 1280), (792, 1408),
        (864, 1536), (936, 1664), (1008, 1792), (1080, 1920),
        # Landscape 2:1
        (1024, 512), (1152, 576), (1280, 640), (1408, 704),
        # Portrait 1:2
        (512, 1024), (576, 1152), (640, 1280), (704, 1408)
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "description": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "description": "Image height in pixels"
                }),
                "priority_mode": (["minimize_loss", "minimize_noise", "auto_decide"], {
                    "default": "auto_decide"
                }),
                "use_standard_list": ("BOOLEAN", {
                    "default": True
                }),
                "mode": (["crop", "pad"], {
                    "default": "crop"
                })
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_resolutions": ("STRING", {
                    "multiline": True,
                    "default": ""
                })
            }
        }

    def calculate_quality_metrics(
        self,
        original_w: int,
        original_h: int,
        target_w: int,
        target_h: int,
        mode: str
    ) -> Dict[str, float]:
        """Calculate quality impact metrics."""
        original_aspect = original_w / original_h
        target_aspect = target_w / target_h
        aspect_ratio_diff = abs(original_aspect - target_aspect)
        
        if mode == "crop":
            scale_factor = min(target_w / original_w, target_h / original_h)
            scaled_w = int(original_w * scale_factor)
            scaled_h = int(original_h * scale_factor)
            pixels_lost = (scaled_w * scaled_h) - (target_w * target_h)
            pixel_loss_percent = (pixels_lost / (scaled_w * scaled_h)) * 100
        else:
            scale_factor = max(target_w / original_w, target_h / original_h)
            pixel_loss_percent = 0
            
        return {
            "scale_factor": scale_factor,
            "pixel_loss_percent": pixel_loss_percent,
            "aspect_ratio_diff": aspect_ratio_diff
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
        """Calculate resolution score with aspect ratio preservation."""
        metrics = self.calculate_quality_metrics(
            original_w, original_h, target_w, target_h, mode
        )
        
        aspect_weight = 3.0  # High weight for aspect ratio preservation
        pixel_loss_weight = 2.0 if priority_mode == "minimize_loss" else 0.5
        scale_weight = 0.5 if priority_mode == "minimize_loss" else 2.0
        
        aspect_score = metrics["aspect_ratio_diff"] * aspect_weight
        scale_score = abs(1 - metrics["scale_factor"]) * scale_weight
        pixel_loss_score = (metrics["pixel_loss_percent"] / 100) * pixel_loss_weight
            
        return aspect_score + scale_score + pixel_loss_score

    def recommend_resolution(self, width: int, height: int, priority_mode: str, use_standard_list: bool, mode: str, image: Optional[torch.Tensor] = None, custom_resolutions: str = "") -> Tuple[Optional[int], Optional[int], float, str, str, int, int]:
        """Main resolution recommendation function."""

        input_width, input_height = width, height
        if image is not None:
            if image.ndim == 3:
                input_height, input_width, _ = image.shape
            elif image.ndim == 4:
                _, input_height, input_width, _ = image.shape
            else:
                raise ValueError(f"Unexpected tensor shape: {image.shape}")

        resolutions = self.DEFAULT_RESOLUTIONS

        if custom_resolutions:
            try:
                custom_res = [tuple(map(int, res.split('x'))) for res in custom_resolutions.strip().split('\n')]
                resolutions = list(set(resolutions + custom_res))
            except ValueError:
                print("Invalid custom resolutions format. Using default resolutions.")

        best_match = None
        best_score = float('inf')
        best_metrics = None

        for w, h in resolutions:
            score = self.calculate_score(input_width, input_height, w, h, priority_mode, mode)

            if score < best_score:
                best_score = score
                best_match = (w, h)
                best_metrics = self.calculate_quality_metrics(input_width, input_height, w, h, mode)

        if not best_match:
            return (None, None, 1.0, "No suitable resolution found", "No changes required", 0, 0)

        scale_factor = best_metrics["scale_factor"]

        if mode == "crop":
            scaled_width = int(input_width * scale_factor)
            scaled_height = int(input_height * scale_factor)
            crop_w = max(0, scaled_width - best_match[0]) // 2
            crop_h = max(0, scaled_height - best_match[1]) // 2
            mod_str = f"Crop: left/right={crop_w}, top/bottom={crop_h}"
            pad_w, pad_h = 0, 0
        else:  # Padding mode
            scale_to_fit = min(best_match[0] / input_width, best_match[1] / input_height)
            scaled_width = int(input_width * scale_to_fit)
            scaled_height = int(input_height * scale_to_fit)

            pad_w = best_match[0] - scaled_width
            pad_h = best_match[1] - scaled_height

            left_pad = pad_w // 2
            right_pad = pad_w - left_pad
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad

            mod_str = f"Pad: left/right={left_pad}/{right_pad}, top/bottom={top_pad}/{bottom_pad}"

        info = (
            f"Input: {input_width}x{input_height}\n"
            f"Output: {best_match[0]}x{best_match[1]}\n"
            f"Scale: {scale_factor:.3f}x\n"
            f"Quality: {100-best_metrics['pixel_loss_percent']:.1f}% preserved\n"
            f"Aspect Ratio Diff: {best_metrics['aspect_ratio_diff']:.3f}"
        )

        return (best_match[0], best_match[1], float(scale_factor), mod_str, info, pad_w, pad_h)
        
        return (best_match[0], best_match[1], float(scale_factor), mod_str, info, pad_w, pad_h)