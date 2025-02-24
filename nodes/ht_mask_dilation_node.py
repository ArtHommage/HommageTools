import torch
import numpy as np
import comfy

class ImageMaskResize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale_mode": (["Scale Closest", "Scale Up", "Scale Down", "Scale Max"],),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT",)
    RETURN_NAMES = ("dilated_mask", "cropped_image", "width", "height", "scale_factor",)
    FUNCTION = "resize_mask_image"
    CATEGORY = "image"

    def resize_mask_image(self, image, mask, scale_mode):
        mask_np = mask.cpu().numpy().squeeze(0)  # Convert to numpy and remove batch dim
        mask_binary = (mask_np > 0).astype(np.uint8)

        # Calculate bounding box
        rows, cols = np.where(mask_binary == 1)
        if len(rows) == 0 or len(cols) == 0:
            print("Warning: Empty mask. Returning original image and mask.")
            return (mask, image, image.shape[2], image.shape[1], 1.0)
        min_y, max_y = np.min(rows), np.max(rows)
        min_x, max_x = np.min(cols), np.max(cols)
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1

        # Calculate scale factor
        long_edge = max(bbox_width, bbox_height)
        buckets = [512, 768, 1024]

        if scale_mode == "Scale Closest":
            target_size = min(buckets, key=lambda x: abs(x - long_edge))
        elif scale_mode == "Scale Up":
            target_size = next((x for x in buckets if x >= long_edge), 1024)
        elif scale_mode == "Scale Down":
            target_size = max((x for x in buckets if x <= long_edge), 512)
        elif scale_mode == "Scale Max":
            target_size = 1024

        scale_factor = target_size / long_edge

        # Calculate scaled dimensions
        scaled_width = int(bbox_width * scale_factor)
        scaled_height = int(bbox_height * scale_factor)

        # Dilate mask and adjust short edge
        short_edge = min(scaled_width, scaled_height)
        padding = (64 - (short_edge % 64)) % 64
        padding_half = padding // 2

        scaled_width += (padding if scaled_width == short_edge else 0)
        scaled_height += (padding if scaled_height == short_edge else 0)

        # Calculate new bounding box coordinates (centered expansion)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        new_min_x = max(0, center_x - scaled_width // 2)
        new_min_y = max(0, center_y - scaled_height // 2)
        new_max_x = min(image.shape[2], new_min_x + scaled_width)
        new_max_y = min(image.shape[1], new_min_y + scaled_height)

        # Adjust for edge cases
        new_min_x = max(0, new_max_x - scaled_width)
        new_min_y = max(0, new_max_y - scaled_height)

        # Crop image and mask
        cropped_image = image[:, new_min_y:new_max_y, new_min_x:new_max_x, :]
        dilated_mask_np = np.zeros_like(mask_np)
        dilated_mask_np[new_min_y:new_max_y, new_min_x:new_max_x] = mask_np[new_min_y:new_max_y, new_min_x:new_max_x]

        dilated_mask = torch.from_numpy(dilated_mask_np.astype(np.float32)).unsqueeze(0)

        return (dilated_mask, cropped_image, cropped_image.shape[2], cropped_image.shape[1], scale_factor)

NODE_CLASS_MAPPINGS = {
    "ImageMaskResize": ImageMaskResize
}