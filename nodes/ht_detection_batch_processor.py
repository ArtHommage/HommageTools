#######################################################
# SECTION 1: CORE DEFINITIONS AND ENUMS
#######################################################

import math
import copy
import numpy as np
import torch
from PIL import Image
from enum import Enum

# Enum for scale mode
class ScaleMode(Enum):
    MAX = "max"  # Scale up to 1024 regardless
    UP = "up"    # Scale up to next closest bucket
    DOWN = "down"  # Scale down to next closest bucket
    CLOSEST = "closest"  # Scale to closest bucket

# Enum for short edge divisibility
class ShortEdgeDivisibility(Enum):
    DIV_BY_8 = 8
    DIV_BY_64 = 64

# Target bucket sizes
BUCKET_SIZES = [1024, 768, 512]
MAX_BUCKET_SIZE = 1024

# Constants
MAX_RESOLUTION = 8192
BLUR_KERNEL_SIZE = 15

# USDU Enums
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}

# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}
#######################################################
# SECTION 2: UTILITY AND HELPER FUNCTIONS
#######################################################

def tensor_to_pil(img_tensor, batch_index=0):
    """Convert a tensor to a PIL image."""
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

def pil_to_tensor(image):
    """Convert a PIL image to a tensor."""
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image

def get_crop_region(mask, pad=0):
    """Get crop region from a mask."""
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))

def fix_crop_region(region, image_size):
    """Fix crop region boundaries."""
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2

def expand_crop(region, width, height, target_width, target_height):
    """Expand a crop region to a target size."""
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # Try to expand region to the right
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)

def crop_tensor(tensor, region):
    """Crop a tensor to a specified region."""
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]

def resize_tensor(tensor, size, mode="nearest-exact"):
    """Resize a tensor to specified dimensions."""
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)

def resize_region(region, init_size, resize_size):
    """Resize a crop region based on image resize."""
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)

def region_intersection(region1, region2):
    """Find the intersection of two rectangular regions."""
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)

def pad_image2(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    """Pad an image with pixels on each side."""
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        if left_pad > 0:
            padded_image.paste(left_edge.resize((left_pad, new_height), resample=Image.Resampling.NEAREST), (0, 0))
        if right_pad > 0:
            padded_image.paste(right_edge.resize((right_pad, new_height),
                              resample=Image.Resampling.NEAREST), (new_width - right_pad, 0))
        if top_pad > 0:
            padded_image.paste(top_edge.resize((new_width, top_pad), resample=Image.Resampling.NEAREST), (0, 0))
        if bottom_pad > 0:
            padded_image.paste(bottom_edge.resize((new_width, bottom_pad),
                              resample=Image.Resampling.NEAREST), (0, new_height - bottom_pad))
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE))
            padded_image.paste(image, (left_pad, top_pad))
    return padded_image

def resize_and_pad_image(image, width, height, fill=False, blur=False):
    """Resize an image and pad it to specified dimensions."""
    width_ratio = width / image.width
    height_ratio = height / image.height
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_image2(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)
    return result, (horizontal_pad, vertical_pad)
#######################################################
# SECTION 3: BUCKET-RELATED CORE LOGIC
#######################################################

# Helper function to calculate target bucket dimensions
def calculate_bucket_dimensions(width, height, scale_mode, short_edge_div, mask_dilation=1.0):
    """
    Calculate the target dimensions based on bucket constraints.
    
    Args:
        width: Original width
        height: Original height
        scale_mode: ScaleMode enum value
        short_edge_div: ShortEdgeDivisibility enum value
        mask_dilation: Factor to dilate the mask by before calculating target dimensions
    
    Returns:
        tuple: (target_width, target_height, scale_type, bucket_size)
    """
    # Apply dilation factor to dimensions
    width = int(width * mask_dilation)
    height = int(height * mask_dilation)
    
    # Determine long and short edges
    is_landscape = width >= height
    long_edge = width if is_landscape else height
    short_edge = height if is_landscape else width
    
    # Cap at MAX_BUCKET_SIZE if larger
    if long_edge > MAX_BUCKET_SIZE:
        target_long_edge = MAX_BUCKET_SIZE
        scale_type = "down"
    else:
        # Select target bucket based on scale mode
        if scale_mode == ScaleMode.MAX:
            # Always scale up to MAX_BUCKET_SIZE
            target_long_edge = MAX_BUCKET_SIZE
            scale_type = "up" if long_edge < MAX_BUCKET_SIZE else "same"
        elif scale_mode == ScaleMode.UP:
            # Find the next bucket size up
            larger_buckets = [b for b in BUCKET_SIZES if b > long_edge]
            if larger_buckets:
                target_long_edge = min(larger_buckets)
                scale_type = "up"
            else:
                target_long_edge = MAX_BUCKET_SIZE
                scale_type = "max"
        elif scale_mode == ScaleMode.DOWN:
            # Find the next bucket size down
            smaller_buckets = [b for b in BUCKET_SIZES if b < long_edge]
            if smaller_buckets:
                target_long_edge = max(smaller_buckets)
                scale_type = "down"
            else:
                target_long_edge = min(BUCKET_SIZES)
                scale_type = "min"
        else:  # CLOSEST
            # Find the closest bucket size
            closest_bucket = min(BUCKET_SIZES, key=lambda b: abs(b - long_edge))
            target_long_edge = closest_bucket
            scale_type = "up" if closest_bucket > long_edge else "down" if closest_bucket < long_edge else "same"
    
    # Calculate scaling factor
    scale_factor = target_long_edge / long_edge
    
    # Calculate raw short edge size after scaling
    raw_short_edge = short_edge * scale_factor
    
    # Adjust short edge to be divisible by the specified divisor
    divisor = short_edge_div.value
    adjusted_short_edge = math.ceil(raw_short_edge / divisor) * divisor
    
    # Calculate final dimensions
    if is_landscape:
        target_width = target_long_edge
        target_height = adjusted_short_edge
    else:
        target_width = adjusted_short_edge
        target_height = target_long_edge
    
    return (target_width, target_height, scale_type, target_long_edge)

# Helper function to adjust crop regions to fit target bucket
def adjust_crop_to_bucket(crop_region, original_img_size, scale_mode, short_edge_div, mask_dilation=1.0):
    """
    Adjust a crop region to fit into the target bucket dimensions.
    
    Args:
        crop_region: Tuple (x1, y1, x2, y2) defining the crop
        original_img_size: Tuple (width, height) of the original image
        scale_mode: ScaleMode enum value
        short_edge_div: ShortEdgeDivisibility enum value
        mask_dilation: Factor to dilate the mask by before calculating target dimensions
    
    Returns:
        tuple: (adjusted_crop_region, target_dimensions, scale_type, bucket_size)
    """
    x1, y1, x2, y2 = crop_region
    img_width, img_height = original_img_size
    
    # Current crop dimensions
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Apply mask dilation
    if mask_dilation != 1.0:
        # Calculate expansion needed
        width_expansion = int(crop_width * (mask_dilation - 1))
        height_expansion = int(crop_height * (mask_dilation - 1))
        
        # Apply expansion evenly on all sides
        x1 = max(0, x1 - width_expansion // 2)
        y1 = max(0, y1 - height_expansion // 2)
        x2 = min(img_width, x2 + width_expansion // 2)
        y2 = min(img_height, y2 + height_expansion // 2)
        
        # Update dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
    
    # Calculate target dimensions for this crop
    target_width, target_height, scale_type, bucket_size = calculate_bucket_dimensions(
        crop_width, crop_height, scale_mode, short_edge_div
    )
    
    # Calculate required expansion to achieve target ratio
    target_ratio = target_width / target_height
    current_ratio = crop_width / crop_height
    
    # Adjust crop to match target ratio
    if current_ratio > target_ratio:
        # Width is the limiting factor, adjust height
        new_height = crop_width / target_ratio
        height_diff = new_height - crop_height
        y1 = max(0, y1 - height_diff / 2)
        y2 = min(img_height, y2 + height_diff / 2)
    else:
        # Height is the limiting factor, adjust width
        new_width = crop_height * target_ratio
        width_diff = new_width - crop_width
        x1 = max(0, x1 - width_diff / 2)
        x2 = min(img_width, x2 + width_diff / 2)
    
    # Ensure the crop values are integers
    adjusted_crop = (int(x1), int(y1), int(x2), int(y2))
    target_dims = (int(target_width), int(target_height))
    
    return adjusted_crop, target_dims, scale_type, bucket_size

# Function to process a batch of SEGS with bucket adjustments
def process_segs_with_buckets(segs, original_image, scale_mode, short_edge_div, mask_dilation=1.0):
    """
    Process SEGS to fit into specified buckets.
    
    Args:
        segs: SEGS object containing segmentation information
        original_image: The original image tensor
        scale_mode: ScaleMode enum value
        short_edge_div: ShortEdgeDivisibility enum value
        mask_dilation: Factor to dilate the mask before calculating target dimensions
    
    Returns:
        tuple: (image_batch, crop_regions, target_dimensions, scale_types, bucket_sizes)
    """
    if segs is None or len(segs[1]) == 0:
        # Create a 1x1 black pixel image tensor for empty SEGS
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return empty_tensor, [], [], [], []
    
    image_crops = []
    crop_regions = []
    target_dimensions = []
    scale_types = []
    bucket_sizes = []
    
    img_height, img_width = original_image.shape[1:3]
    original_img_size = (img_width, img_height)
    
    for seg in segs[1]:
        # Get crop region
        x1, y1, x2, y2 = seg.crop_region
        
        # Adjust crop region to fit target bucket
        adjusted_crop, target_dims, scale_type, bucket_size = adjust_crop_to_bucket(
            (x1, y1, x2, y2), original_img_size, scale_mode, short_edge_div, mask_dilation
        )
        
        # Update with adjusted crop
        ax1, ay1, ax2, ay2 = adjusted_crop
        
        # Store information
        crop_regions.append(adjusted_crop)
        target_dimensions.append(target_dims)
        scale_types.append(scale_type)
        bucket_sizes.append(bucket_size)
        
        # Crop the image with adjusted coordinates
        cropped_image = original_image[:, ay1:ay2, ax1:ax2, :]
        image_crops.append(cropped_image)
    
    # Stack the crops into a batch
    if len(image_crops) > 0:
        image_batch = torch.cat(image_crops, dim=0)
        return image_batch, crop_regions, target_dimensions, scale_types, bucket_sizes
    else:
        # Fallback 1x1 black pixel for safety
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return empty_tensor, [], [], [], []

# Add the standard segs_to_image_batch function for compatibility
def segs_to_image_batch(segs, original_image):
    """
    Convert SEGS to a batch of image crops for processing.
    
    Args:
        segs: SEGS object containing segmentation information
        original_image: The original image tensor
    
    Returns:
        Image batch tensor, crop_regions list
    """
    if segs is None or len(segs[1]) == 0:
        # Create a 1x1 black pixel image tensor for empty SEGS
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return empty_tensor, []
    
    image_crops = []
    crop_regions = []
    
    for seg in segs[1]:
        # Get crop region
        x1, y1, x2, y2 = seg.crop_region
        crop_regions.append(seg.crop_region)
        
        # Crop the image
        cropped_image = original_image[:, y1:y2, x1:x2, :]
        image_crops.append(cropped_image)
    
    # Stack the crops into a batch
    if len(image_crops) > 0:
        image_batch = torch.cat(image_crops, dim=0)
        return image_batch, crop_regions
    else:
        # Fallback 1x1 black pixel for safety
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return empty_tensor, []

# Helper function to reconstruct the full image from processed crops
def reconstruct_from_crops(original_image, processed_crops, crop_regions):
    """
    Reconstruct a full image from processed crops.
    
    Args:
        original_image: The original image tensor (used as base)
        processed_crops: The processed (upscaled) image crops
        crop_regions: List of crop regions (x1, y1, x2, y2)
    
    Returns:
        Reconstructed image tensor
    """
    # Create a copy of the original image to modify
    result_image = copy.deepcopy(original_image)
    
    # For each crop, place it back in the original image at the crop region
    for i, crop_region in enumerate(crop_regions):
        x1, y1, x2, y2 = crop_region
        
        # Handle differently sized crops (if upscaled)
        crop = processed_crops[i]
        
        # Resize crop to fit original region if needed
        if crop.shape[1] != y2-y1 or crop.shape[2] != x2-x1:
            from torchvision.transforms.functional import resize
            # Resize to match the original crop region
            crop = resize(crop, (y2-y1, x2-x1))
        
        # Place the crop back in the result image
        result_image[:, y1:y2, x1:x2, :] = crop
    
    return result_image
#######################################################
# SECTION 4: CONDITIONING-RELATED FUNCTIONS
#######################################################

def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    """
    Crop ControlNet conditioning to match the current region.
    
    Args:
        cond_dict: Conditioning dictionary 
        region: Tuple (x1, y1, x2, y2) defining the crop region
        init_size: Original image size
        canvas_size: Size of the canvas
        tile_size: Size of the tile
        w_pad: Width padding
        h_pad: Height padding
    """
    if "control" not in cond_dict:
        return
    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet
    while c is not None:
        # hint is shape (B, C, H, W)
        hint = controlnet.cond_hint_original
        resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])
        hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
        hint = resize_tensor(hint, tile_size[::-1])
        controlnet.cond_hint_original = hint
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet

def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop GLIGEN conditioning to match the current region.
    
    Args:
        cond_dict: Conditioning dictionary
        region: Tuple (x1, y1, x2, y2) defining the crop region
        init_size: Original image size
        canvas_size: Size of the canvas
        tile_size: Size of the tile
        w_pad: Width padding
        h_pad: Height padding
    """
    if "gligen" not in cond_dict:
        return
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn
        warn(f"Unknown gligen type {type}")
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        # Get the coordinates of the box in the upscaled image
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Calculate the intersection of the gligen box and the region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset the gligen box so that the origin is at the top left of the tile region
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Add the padding
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad

        # Set the new position params
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))

    cond_dict["gligen"] = (type, model, cropped)

def crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop area conditioning to match the current region.
    
    Args:
        cond_dict: Conditioning dictionary
        region: Tuple (x1, y1, x2, y2) defining the crop region
        init_size: Original image size
        canvas_size: Size of the canvas
        tile_size: Size of the tile
        w_pad: Width padding
        h_pad: Height padding
    """
    if "area" not in cond_dict:
        return

    # Resize the area conditioning to the canvas size and confine it to the tile region
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection

    # Offset origin to the top left of the tile
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]

    # Add the padding
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad

    # Set the params for tile
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8

    cond_dict["area"] = (h, w, y, x)

def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop mask conditioning to match the current region.
    
    Args:
        cond_dict: Conditioning dictionary
        region: Tuple (x1, y1, x2, y2) defining the crop region
        init_size: Original image size
        canvas_size: Size of the canvas
        tile_size: Size of the tile
        w_pad: Width padding
        h_pad: Height padding
    """
    if "mask" not in cond_dict:
        return
    mask_tensor = cond_dict["mask"]  # (B, H, W)
    masks = []
    for i in range(mask_tensor.shape[0]):
        # Convert to PIL image
        mask = tensor_to_pil(mask_tensor, i)  # W x H

        # Resize the mask to the canvas size
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)

        # Crop the mask to the region
        mask = mask.crop(region)

        # Add padding
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)

        # Resize the mask to the tile size
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)

        # Convert back to tensor
        mask = pil_to_tensor(mask)  # (1, H, W, 1)
        mask = mask.squeeze(-1)  # (1, H, W)
        masks.append(mask)

    cond_dict["mask"] = torch.cat(masks, dim=0)  # (B, H, W)

def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    """
    Crop all conditioning information to match the current region.
    
    Args:
        cond: List of conditioning tensors and dictionaries
        region: Tuple (x1, y1, x2, y2) defining the crop region
        init_size: Original image size
        canvas_size: Size of the canvas
        tile_size: Size of the tile
        w_pad: Width padding
        h_pad: Height padding
        
    Returns:
        Cropped conditioning information
    """
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        cropped.append(n)
    return cropped
#######################################################
# SECTION 5: OBJECT DETECTION AND SEGMENTATION
#######################################################

# Embedding SimpleDetectorForEach functionality directly in this node
class SimpleDetectorForEach:
    """
    Class providing detection and segmentation functionality.
    This embedded implementation avoids external dependencies on impact.detectors
    """
    
    @classmethod
    def detect(cls, detector, image, threshold, dilation, crop_factor, drop_size, 
               crop_min_size=0, crop_square=False, detailer_hook=None, bbox_fill=0.5, segment_threshold=0.5,
               sam_model_opt=None, segm_detector_opt=None):
        """
        Run detection on an image.
        
        Args:
            detector: Bbox detector model
            image: Image tensor
            threshold: Detection confidence threshold
            dilation: Amount to dilate detected regions
            crop_factor: Factor for cropping
            drop_size: Minimum size to keep
            crop_min_size: Minimum crop size
            crop_square: Whether to make crops square
            detailer_hook: Hook for detailed detection
            bbox_fill: Bounding box fill factor
            segment_threshold: Segmentation threshold
            sam_model_opt: Optional SAM model for segmentation
            segm_detector_opt: Optional segmentation detector
            
        Returns:
            SEGS object with detection results
        """
        # Run basic detection
        segs = detector.detect(image, threshold, dilation, crop_factor, drop_size, 
                             crop_min_size, crop_square, detailer_hook, bbox_fill)
                             
        # Run segmentation refinement if available
        if segm_detector_opt is not None and sam_model_opt is not None:
            refined_segs = cls.segment_refined(segs, sam_model_opt, segm_detector_opt, segment_threshold)
            return refined_segs
        
        return segs
    
    @classmethod
    def segment_refined(cls, segs, sam_model, segm_detector, threshold):
        """
        Refine detections with segmentation.
        
        Args:
            segs: SEGS object with bounding box detections
            sam_model: SAM model for segmentation
            segm_detector: Segmentation detector
            threshold: Segmentation threshold
            
        Returns:
            Refined SEGS object with segmentation
        """
        # Basic segmentation refinement logic
        # This is a simplified version of what would be in the impact.detectors module
        if segs is None or segs[1] is None or len(segs[1]) == 0:
            return segs
            
        try:
            # Use the segmentation detector to refine the bounding boxes
            refined_segs = segm_detector.detect(segs, sam_model=sam_model, threshold=threshold)
            return refined_segs
        except Exception as e:
            print(f"[USDU] Segmentation refinement error: {e}")
            return segs
#######################################################
# SECTION 6: DATA MODELS AND SHARED STATE
#######################################################

# Shared module implementation
class Options:
    """Class representing global options."""
    img2img_background_color = "#ffffff"  # Set to white for now

class State:
    """Class for tracking global state."""
    interrupted = False

    def begin(self):
        """Begin processing."""
        pass

    def end(self):
        """End processing."""
        pass

# Global variables (equivalent to shared.py)
opts = Options()
state = State()
sd_upscalers = [None]  # Will only ever hold the one upscaler
actual_upscaler = None  # The upscaler usable by ComfyUI nodes
batch = None  # Batch of images to upscale

# Upscaler implementation
class Upscaler:
    """Class providing upscaling functionality."""
    
    def _upscale(self, img: Image, scale):
        """
        Internal upscale method.
        
        Args:
            img: PIL image to upscale
            scale: Scale factor
        
        Returns:
            Upscaled PIL image
        """
        if scale == 1.0:
            return img
        if (actual_upscaler is None):
            return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        tensor = pil_to_tensor(img)
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        """
        Upscale all images in the batch.
        
        Args:
            img: PIL image (first in batch)
            scale: Scale factor
            selected_model: Optionally select a specific model
            
        Returns:
            First upscaled image
        """
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]

class UpscalerData:
    """Wrapper for upscaler data."""
    name = ""
    data_path = ""

    def __init__(self):
        """Initialize with an Upscaler instance."""
        self.scaler = Upscaler()
#######################################################
# SECTION 7: PROCESSING CORE
#######################################################

# StableDiffusionProcessing implementation
class StableDiffusionProcessing:
    """
    Core processing class for Stable Diffusion upscaling operations.
    """
    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height
        self.rows = math.ceil(self.height / tile_height)
        self.cols = math.ceil(self.width / tile_width)

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise
        self.upscale_by = upscale_by

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[USDU] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        
        # Import here to avoid circular imports
        from nodes import VAEEncode, VAEDecode, VAEDecodeTiled
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            print("[USDU] Using tiled decode")

        # Other required A1111 variables for the USDU script
        self.extra_generation_params = {}

        # Load config file for USDU
        import os
        import json
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        import comfy
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar = None
            # Creating the pbar here would cause an empty progress bar to be displayed

    def __del__(self):
        """Clean up when the object is destroyed."""
        # Undo changes to progress bar flag when node is done or cancelled
        import comfy
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True

class Processed:
    """
    Class to hold processed results.
    """
    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        """Get info text for a processed image."""
        return None

def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """
    Generate samples using the specified sampler and parameters.
    
    Args:
        model: Stable Diffusion model
        seed: Random seed
        steps: Number of sampling steps
        cfg: Classifier-free guidance scale
        sampler_name: Name of sampler to use
        scheduler: Scheduler to use
        positive: Positive conditioning
        negative: Negative conditioning
        latent: Latent input image
        denoise: Denoising strength
        custom_sampler: Optional custom sampler
        custom_sigmas: Optional custom sigmas
        
    Returns:
        Generated samples
    """
    # Choose way to sample based on given inputs

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        custom_sample = SamplerCustom()
        (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        return samples

    # Default
    from nodes import common_ksampler
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                               scheduler, positive, negative, latent, denoise=denoise)
    return samples

def process_images(p: StableDiffusionProcessing) -> Processed:
    """
    Main image generation function.
    
    Args:
        p: StableDiffusionProcessing object with all parameters
        
    Returns:
        Processed object with results
    """
    # Show the progress bar
    from tqdm import tqdm
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    # Create a full white mask for each image if not provided
    if p.image_mask is None:
        p.image_mask = Image.new('L', (p.init_images[0].width, p.init_images[0].height), 255)
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                           image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        from PIL import ImageFilter
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    global batch
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                   negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed

# USDU script stub - replaced by the actual implementation when running
class Script:
    """
    Script class that serves as an adapter to the USDU script.
    """
    def run(self, p, _, **kwargs):
        """Run the script either using the original USDU or the fallback."""
        # Import the actual USDU implementation
        try:
            from usdu_patch import usdu
            # Use the original script if available
            return usdu.Script().run(p=p, _=_, **kwargs)
        except ImportError:
            print("[USDU] Warning: usdu_patch module not found, using built-in implementation")
            # Fallback to our implementation
            return process_images(p)
#######################################################
# SECTION 8: MAIN NODE IMPLEMENTATION
#######################################################

# The UltimateSDUpscale node with buckets support
class HTDetectionBatchProcessor:
    """
    ComfyUI node that implements Ultimate SD Upscale with bucket support.
    Detects objects in images and upscales them to standardized dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        """Define inputs for the node."""
        required = {
            "image": ("IMAGE",),  # Will accept batch of images
            # Sampling Params
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            # Upscale Params
            "upscale_model": ("UPSCALE_MODEL",),
            "mode_type": (list(MODES.keys()),),
            "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            "tile_padding": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            # Seam fix params
            "seam_fix_mode": (list(SEAM_FIX_MODES.keys()),),
            "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            # Detection Params
            "detection_model": ("BBOX_DETECTOR",),
            "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "detection_dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
            "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
            "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
            # Bucket Params (NEW)
            "scale_mode": ([mode.value for mode in ScaleMode], {"default": "max"}),
            "short_edge_div": ([div.name for div in ShortEdgeDivisibility], {"default": "DIV_BY_8"}),
            "mask_dilation": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05}),
            "use_buckets": ("BOOLEAN", {"default": True, "label": "Use target buckets"}),
            # Misc
            "force_uniform_tiles": ("BOOLEAN", {"default": True}),
            "tiled_decode": ("BOOLEAN", {"default": False}),
        }
        
        optional = {
            "segs": ("SEGS", {"default": None}),
            "use_provided_segs": ("BOOLEAN", {"default": False, "label": "Use provided SEGS instead of detecting"}),
            "segs_upscale_separately": ("BOOLEAN", {"default": True, "label": "Upscale SEGS separately"}),
            "detection_upscale_model": ("UPSCALE_MODEL", {"default": None}),
            "use_detection_upscaler": ("BOOLEAN", {"default": False, "label": "Use separate upscaler for SEGS"}),
            "sam_model_opt": ("SAM_MODEL", {"default": None}),
            "segm_detector_opt": ("SEGM_DETECTOR", {"default": None})
        }

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("upscaled_image", "cropped_batch", "scale_info")
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, detection_model, detection_threshold,
                detection_dilation, crop_factor, drop_size, scale_mode, short_edge_div,
                mask_dilation, use_buckets, force_uniform_tiles, tiled_decode,
                segs=None, use_provided_segs=False, segs_upscale_separately=True,
                detection_upscale_model=None, use_detection_upscaler=False,
                sam_model_opt=None, segm_detector_opt=None):
        """
        Main upscaling function that processes either segments or whole images.
        """
        # Store params
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.mask_blur = mask_blur
        self.tile_padding = tile_padding
        self.seam_fix_width = seam_fix_width
        self.seam_fix_denoise = seam_fix_denoise
        self.seam_fix_padding = seam_fix_padding
        self.seam_fix_mode = seam_fix_mode
        self.mode_type = mode_type
        self.seam_fix_mask_blur = seam_fix_mask_blur
        
        # Convert string enum choices to actual enum values
        scale_mode_enum = ScaleMode(scale_mode)
        short_edge_div_enum = ShortEdgeDivisibility[short_edge_div]

        # IMPORTANT: All global declarations must appear here at the top
        # before any usage of these variables
        global sd_upscalers, actual_upscaler, batch
        
        # Set up required globals
        sd_upscalers[0] = UpscalerData()
        
        # For storing scale info
        scale_info = []
        
        # Get SEGS - either from provided input or by running detection
        working_segs = None
        if use_provided_segs and segs is not None:
            working_segs = segs
            print(f"[USDU] Using provided SEGS with {len(segs[1])} items")
        else:
            # Run detection using provided detector
            print(f"[USDU] Running detection with threshold {detection_threshold}")
            if segm_detector_opt is not None:
                # Use SimpleDetectorForEach to get SEGS with the provided detectors
                working_segs = SimpleDetectorForEach.detect(
                    detection_model, image, detection_threshold, detection_dilation, 
                    crop_factor, drop_size, 0.5, 0, 0, 0.7, 0,
                    sam_model_opt=sam_model_opt, segm_detector_opt=segm_detector_opt)
            else:
                # Fallback to basic detection without segmentation refinement
                working_segs = detection_model.detect(image, detection_threshold, detection_dilation, crop_factor, drop_size)
            
            print(f"[USDU] Detection completed, found {len(working_segs[1])} regions")
        
        # Process SEGS if available
        if working_segs is not None:
            print(f"[USDU] Processing detected regions")
            
            # If using buckets, process SEGS with bucket adjustment
            if use_buckets:
                image_batch, crop_regions, target_dimensions, scale_types, bucket_sizes = process_segs_with_buckets(
                    working_segs, image, scale_mode_enum, short_edge_div_enum, mask_dilation
                )
                
                # Generate scale info for output
                for i, (region, target_dim, scale_type, bucket_size) in enumerate(zip(crop_regions, target_dimensions, scale_types, bucket_sizes)):
                    x1, y1, x2, y2 = region
                    width, height = target_dim
                    scale_info.append(f"Region {i+1}: {x2-x1}x{y2-y1} → {width}x{height} ({scale_type}, bucket: {bucket_size})")
            else:
                # Original SEGS processing without buckets
                image_batch, crop_regions = segs_to_image_batch(working_segs, image)
                scale_info.append("Bucket sizing disabled")
            
            # Empty SEGS check - no crops detected
            if len(crop_regions) == 0:
                print("[USDU] No valid regions detected")
                # Return 1x1 black pixel for outputs
                empty_tensor = torch.zeros(1, 1, 1, 3)
                return (empty_tensor, empty_tensor, "No regions detected")
            
            # Store the raw cropped batch for output
            raw_cropped_batch = image_batch.clone()
            
            # Determine which upscale model to use for SEGS regions
            current_upscale_model = detection_upscale_model if use_detection_upscaler and detection_upscale_model is not None else upscale_model
            
            # Set the upscaler for SEGS regions
            actual_upscaler = current_upscale_model
            print(f"[USDU] Using {'detection' if use_detection_upscaler and detection_upscale_model is not None else 'primary'} upscaler for regions")
            
            if segs_upscale_separately:
                # Process each segmented region separately
                result_images = []
                
                for i in range(len(image_batch)):
                    # Process single crop
                    batch = [tensor_to_pil(image_batch[i:i+1])]
                    
                    # If using buckets, use the target dimensions to determine upscale factor
                    if use_buckets:
                        original_width, original_height = batch[0].size
                        target_width, target_height = target_dimensions[i]
                        
                        # Calculate effective upscale factor based on target dimensions
                        width_factor = target_width / original_width
                        height_factor = target_height / original_height
                        effective_upscale = max(width_factor, height_factor)
                        
                        # Override upscale_by for this segment
                        current_upscale_by = effective_upscale
                    else:
                        current_upscale_by = 2.0  # Default value
                    
                    # Create processing object for this crop
                    sdprocessing = StableDiffusionProcessing(
                        batch[0], model, positive, negative, vae,
                        seed, steps, cfg, sampler_name, scheduler, denoise, current_upscale_by, 
                        force_uniform_tiles, tiled_decode, tile_width, tile_height, 
                        MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode]
                    )
                    
                    # Set the image mask for this batch item
                    sdprocessing.image_mask = Image.new('L', (batch[0].width, batch[0].height), 255)
                    sdprocessing.init_images = [batch[0]]
                    
                    # If using buckets, override the target custom scale
                    custom_scale = effective_upscale if use_buckets else 2.0
                    
                    # Run upscaling process
                    script = Script()
                    script.run(
                        p=sdprocessing, 
                        _=None, 
                        tile_width=self.tile_width, 
                        tile_height=self.tile_height,
                        mask_blur=self.mask_blur, 
                        padding=self.tile_padding, 
                        seams_fix_width=self.seam_fix_width,
                        seams_fix_denoise=self.seam_fix_denoise, 
                        seams_fix_padding=self.seam_fix_padding,
                        upscaler_index=0, 
                        save_upscaled_image=False, 
                        redraw_mode=MODES[self.mode_type],
                        save_seams_fix_image=False, 
                        seams_fix_mask_blur=self.seam_fix_mask_blur,
                        seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], 
                        target_size_type=2,
                        custom_width=None, 
                        custom_height=None, 
                        custom_scale=custom_scale
                    )
                    
                    # If using buckets, resize to exact target dimensions
                    if use_buckets:
                        processed_img = batch[0]
                        target_width, target_height = target_dimensions[i]
                        if processed_img.width != target_width or processed_img.height != target_height:
                            processed_img = processed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            batch[0] = processed_img
                    
                    # Get the processed image
                    result_images.append(pil_to_tensor(batch[0]))
                    print(f"[USDU] Processed SEGS {i+1}/{len(image_batch)}")
                
                # Stack the results into a tensor
                processed_batch = torch.cat(result_images, dim=0)
                
                # Reconstruct the full image by placing crops back
                result_tensor = reconstruct_from_crops(image, processed_batch, crop_regions)
                return (result_tensor, raw_cropped_batch, "\n".join(scale_info))
            
            else:
                # Process all crops as a single batch
                batch = [tensor_to_pil(image_batch[i:i+1]) for i in range(len(image_batch))]
                
                # If using buckets, we need to process each crop separately anyway
                # since each may have different dimensions/upscale factors
                if use_buckets:
                    # Process separately and collect results
                    processed_crops = []
                    
                    for i, img in enumerate(batch):
                        current_batch = [img]
                        
                        # Get original and target dimensions
                        original_width, original_height = img.size
                        target_width, target_height = target_dimensions[i]
                        
                        # Calculate effective upscale factor
                        width_factor = target_width / original_width
                        height_factor = target_height / original_height
                        effective_upscale = max(width_factor, height_factor)
                        
                        # Create processing object for this crop
                        sdprocessing = StableDiffusionProcessing(
                            img, model, positive, negative, vae,
                            seed, steps, cfg, sampler_name, scheduler, denoise, effective_upscale,
                            force_uniform_tiles, tiled_decode, tile_width, tile_height,
                            MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode]
                        )
                        
                        # Set the image mask for this batch item
                        sdprocessing.image_mask = Image.new('L', img.size, 255)
                        sdprocessing.init_images = [img]
                        
                        # Process this crop
                        batch = current_batch
                        
                        script = Script()
                        script.run(
                            p=sdprocessing, 
                            _=None, 
                            tile_width=self.tile_width, 
                            tile_height=self.tile_height,
                            mask_blur=self.mask_blur, 
                            padding=self.tile_padding, 
                            seams_fix_width=self.seam_fix_width,
                            seams_fix_denoise=self.seam_fix_denoise, 
                            seams_fix_padding=self.seam_fix_padding,
                            upscaler_index=0, 
                            save_upscaled_image=False, 
                            redraw_mode=MODES[self.mode_type],
                            save_seams_fix_image=False, 
                            seams_fix_mask_blur=self.seam_fix_mask_blur,
                            seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], 
                            target_size_type=2,
                            custom_width=None, 
                            custom_height=None, 
                            custom_scale=effective_upscale
                        )
                        
                        # Resize to exact target dimensions
                        processed_img = batch[0]
                        if processed_img.width != target_width or processed_img.height != target_height:
                            processed_img = processed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                        
                        processed_crops.append(processed_img)
                    
                    # Update batch with processed crops
                    batch = processed_crops
                
                else:
                    # Original approach - process as single batch with uniform upscale factor
                    # Create processing object
                    sdprocessing = StableDiffusionProcessing(
                        batch[0], model, positive, negative, vae,
                        seed, steps, cfg, sampler_name, scheduler, denoise, 2.0, force_uniform_tiles, tiled_decode,
                        tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode]
                    )
                    
                    # Set image mask for all batch items
                    sdprocessing.image_mask = Image.new('L', (batch[0].width, batch[0].height), 255)
                    sdprocessing.init_images = batch
                    
                    # Process the entire batch
                    script = Script()
                    script.run(
                        p=sdprocessing, 
                        _=None, 
                        tile_width=self.tile_width, 
                        tile_height=self.tile_height,
                        mask_blur=self.mask_blur, 
                        padding=self.tile_padding, 
                        seams_fix_width=self.seam_fix_width,
                        seams_fix_denoise=self.seam_fix_denoise, 
                        seams_fix_padding=self.seam_fix_padding,
                        upscaler_index=0, 
                        save_upscaled_image=False, 
                        redraw_mode=MODES[self.mode_type],
                        save_seams_fix_image=False, 
                        seams_fix_mask_blur=self.seam_fix_mask_blur,
                        seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], 
                        target_size_type=2,
                        custom_width=None, 
                        custom_height=None, 
                        custom_scale=2.0
                    )
                
                # Convert all PIL images back to tensors
                tensors = [pil_to_tensor(img) for img in batch]
                processed_batch = torch.cat(tensors, dim=0)
                
                # Reconstruct the full image
                result_tensor = reconstruct_from_crops(image, processed_batch, crop_regions)
                return (result_tensor, raw_cropped_batch, "\n".join(scale_info))
        
        # Original processing path (no SEGS)
        else:
            # Reset upscaler to primary for non-SEGS processing
            actual_upscaler = upscale_model
            
            # Convert all input images to PIL for batch processing
            batch = [tensor_to_pil(image, i) for i in range(len(image))]
            
            # Get the first image to initialize processing
            # Each image will be processed individually in a batch
            first_image = tensor_to_pil(image, 0)
            
            # For non-SEGS mode with buckets, calculate dimensions for the whole image
            if use_buckets:
                for i, img in enumerate(batch):
                    width, height = img.size
                    target_width, target_height, scale_type, bucket_size = calculate_bucket_dimensions(
                        width, height, scale_mode_enum, short_edge_div_enum, mask_dilation
                    )
                    scale_info.append(f"Image {i+1}: {width}x{height} → {target_width}x{target_height} ({scale_type}, bucket: {bucket_size})")
                    
                    # Calculate effective upscale
                    width_factor = target_width / width
                    height_factor = target_height / height
                    effective_upscale = max(width_factor, height_factor)
                    current_upscale_by = effective_upscale
            else:
                scale_info.append("Bucket sizing disabled")
                current_upscale_by = 2.0  # Default upscale factor
            
            # Create processing object
            sdprocessing = StableDiffusionProcessing(
                first_image, model, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise, current_upscale_by, force_uniform_tiles, tiled_decode,
                tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode]
            )

            # Disable logging during processing
            import logging
            logger = logging.getLogger()
            old_level = logger.getEffectiveLevel()
            logger.setLevel(logging.CRITICAL + 1)
            try:
                # Get access to USDU script
                try:
                    from usdu_patch import usdu
                    script = usdu.Script()
                except ImportError:
                    # If USDU script is not available, use our implementation
                    print("[USDU] Warning: usdu_patch module not found, using built-in implementation")
                    script = Script()
                    
                # Process each image in the batch
                result_images = []
                
                # Store original batch
                original_batch = batch.copy()
                
                # Process each image separately
                for i in range(len(original_batch)):
                    # Set the current image as the only one in the batch
                    batch = [original_batch[i]]
                    
                    # If using buckets, calculate dimensions for this specific image
                    if use_buckets:
                        width, height = batch[0].size
                        target_width, target_height, scale_type, bucket_size = calculate_bucket_dimensions(
                            width, height, scale_mode_enum, short_edge_div_enum, mask_dilation
                        )
                        
                        # Calculate effective upscale factor
                        width_factor = target_width / width
                        height_factor = target_height / height
                        effective_upscale = max(width_factor, height_factor)
                        
                        # Override upscale_by for this image
                        sdprocessing.upscale_by = effective_upscale
                    
                    # Set the image mask for this batch item
                    sdprocessing.image_mask = Image.new('L', (batch[0].width, batch[0].height), 255)
                    sdprocessing.init_images = [batch[0]]
                    
                    # If using buckets, override the target custom scale
                    custom_scale = effective_upscale if use_buckets else 2.0
                    
                    # Run upscaling process
                    processed = script.run(
                        p=sdprocessing, 
                        _=None, 
                        tile_width=self.tile_width, 
                        tile_height=self.tile_height,
                        mask_blur=self.mask_blur, 
                        padding=self.tile_padding, 
                        seams_fix_width=self.seam_fix_width,
                        seams_fix_denoise=self.seam_fix_denoise, 
                        seams_fix_padding=self.seam_fix_padding,
                        upscaler_index=0, 
                        save_upscaled_image=False, 
                        redraw_mode=MODES[self.mode_type],
                        save_seams_fix_image=False, 
                        seams_fix_mask_blur=self.seam_fix_mask_blur,
                        seams_fix_type=SEAM_FIX_MODES[self.seam_fix_mode], 
                        target_size_type=2,
                        custom_width=None, 
                        custom_height=None, 
                        custom_scale=custom_scale
                    )
                    
                    # If using buckets, resize the final image to target dimensions
                    if use_buckets:
                        processed_img = batch[0]
                        if processed_img.width != target_width or processed_img.height != target_height:
                            processed_img = processed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            batch[0] = processed_img
                    
                    # Store the processed image
                    result_images.append(batch[0])
                    
                    # Print progress
                    print(f"[USDU] Processed image {i+1}/{len(original_batch)}")
                
                # Convert all PIL images back to tensors and combine into a batch
                tensors = [pil_to_tensor(img) for img in result_images]
                combined_tensor = torch.cat(tensors, dim=0)
                return (combined_tensor, image, "\n".join(scale_info))
            finally:
                # Restore the original logging level
                logger.setLevel(old_level)

# Export the node
NODE_CLASS_MAPPINGS = {
    "ht_detection_batch_processor_v2": HTDetectionBatchProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ht_detection_batch_processor_v2": "Ultimate SD Upscale with Buckets",
}