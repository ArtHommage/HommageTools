"""
File: homage_tools/nodes/ht_tensor_info_node.py
Description: Node for displaying tensor shape information in BHWC format
Version: 1.0.3
"""

import torch
from typing import Dict, Any, Tuple
from server import PromptServer

VERSION = "1.0.3"

class HTTensorInfoNode:
    CATEGORY = "HommageTools/Debug"
    FUNCTION = "display_info"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "shape_info")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    def display_info(self, image: torch.Tensor, **kwargs):
        try:
            shape = image.shape
            if len(shape) == 3:
                batch, height, width, channels = 1, *shape
            else:
                batch, height, width, channels = shape

            info = (f"Shape: torch.Size({list(shape)})\n"
                   f"Format: {'BHWC' if len(shape) == 4 else 'HWC'}\n"
                   f"Dimensions: {height}x{width}\n"
                   f"Batch: {batch}, Channels: {channels}")

            if "unique_id" in kwargs:
                PromptServer.instance.send_sync("update_preview", {
                    "node": kwargs["unique_id"],
                    "content": info
                })

            return (image, info)

        except Exception as e:
            return (image, str(e))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")