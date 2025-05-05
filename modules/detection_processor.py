"""
Region detection and processing module for HTDetectionBatchProcessor
"""
import torch
import math
from enum import Enum
import numpy as np
import time
from PIL import Image, ImageFilter

from ..modules.debug_utils import save_debug_image

# Re-export needed enums and constants
class ScaleMode(Enum):
    MAX = "max"  # Scale up to 1024 regardless
    UP = "up"    # Scale up to next closest bucket
    DOWN = "down"  # Scale down to next closest bucket
    CLOSEST = "closest"  # Scale to closest bucket

class ShortEdgeDivisibility(Enum):
    DIV_BY_8 = 8
    DIV_BY_64 = 64

# Target bucket sizes for standardized dimensions
BUCKET_SIZES = [1024, 768, 512]
MAX_BUCKET_SIZE = 1024
MAX_RESOLUTION = 8192

def force_crops_output(original_image, working_segs):
    """
    Create direct visualization of crops from SEGS
    """
    print("[DEBUG] Creating visualization of detected crops")
    
    if working_segs is None or len(working_segs[1]) == 0:
        print("[DEBUG] No segments found for visualization")
        return original_image
    
    # Create a new empty batch to hold the crops
    crops_list = []
    
    # Extract each segment as a separate image
    for i, seg in enumerate(working_segs[1]):
        x1, y1, x2, y2 = seg.crop_region
        print(f"[DEBUG] Extracting region {i}: {x1},{y1} to {x2},{y2}")
        
        try:
            # Extract crop - handling potential dimension issues
            if len(original_image.shape) == 4:  # With batch dimension
                crop = original_image[0, y1:y2, x1:x2, :]
            else:  # Without batch dimension
                crop = original_image[y1:y2, x1:x2, :]
            
            # Add batch dimension if needed
            if len(crop.shape) == 3:
                crop = crop.unsqueeze(0)
                
            print(f"[DEBUG] Crop shape: {crop.shape}")
            crops_list.append(crop)
        except Exception as e:
            print(f"[DEBUG] Crop extraction error: {e}")
    
    if len(crops_list) == 0:
        print("[DEBUG] No valid crops created")
        return original_image
    
    # Stack crops horizontally if possible
    try:
        # Resize all crops to same height
        from torchvision.transforms.functional import resize
        
        # Use first crop's height as reference
        target_height = crops_list[0].shape[1]
        
        # Resize each crop to match reference height
        resized_crops = []
        for crop in crops_list:
            # Move channels to position expected by resize
            crop_ch_first = crop.permute(0, 3, 1, 2)
            # Calculate new width maintaining aspect ratio
            new_width = int(crop.shape[2] * (target_height / crop.shape[1]))
            # Resize
            resized = resize(crop_ch_first, [target_height, new_width])
            # Move channels back
            resized = resized.permute(0, 2, 3, 1)
            resized_crops.append(resized)
        
        # Pad each crop with a border
        padded_crops = []
        for crop in resized_crops:
            # Create slightly larger tensor with black border
            padded = torch.zeros(1, crop.shape[1]+4, crop.shape[2]+4, 3)
            # Place crop in center
            padded[0, 2:-2, 2:-2, :] = crop[0]
            padded_crops.append(padded)
        
        # Concatenate horizontally - we need same height for this
        concat_width = sum(crop.shape[2] for crop in padded_crops)
        concat_crop = torch.zeros(1, padded_crops[0].shape[1], concat_width, 3)
        
        current_x = 0
        for crop in padded_crops:
            width = crop.shape[2]
            concat_crop[0, :, current_x:current_x+width, :] = crop[0]
            current_x += width
        
        print(f"[DEBUG] Concatenated visualization shape: {concat_crop.shape}")
        save_debug_image(concat_crop, "concatenated_crops_visualization")
        return concat_crop
        
    except Exception as e:
        print(f"[DEBUG] Concatenation error: {e}")
        # Return first crop as fallback
        return crops_list[0]

def prepare_crop_batch(image_batch, crop_regions, original_image):
    """
    Properly prepare cropped batch for output, ensuring correct dimensions.
    """
    # If no crops were found, return a modified version of the original image
    if len(crop_regions) == 0:
        print("[DEBUG] No crop regions found, returning marker image")
        # Create a red-tinted version of the original image to indicate no crops
        marker = original_image.clone()
        if marker.shape[-1] == 3:  # If channels are in the last dimension
            marker[..., 0] = 1.0  # Set red channel to max
            marker[..., 1] = 0.3  # Reduce green
            marker[..., 2] = 0.3  # Reduce blue
        return marker
    
    # Ensure the image_batch has the right dimensions
    if len(image_batch.shape) != 4:
        print(f"[DEBUG] Unusual image batch shape: {image_batch.shape}")
        
        # If we have a single image with [H, W, C] format, add batch dimension
        if len(image_batch.shape) == 3 and image_batch.shape[-1] == 3:
            image_batch = image_batch.unsqueeze(0)
        
        # If we have [C, H, W] format, convert to [1, H, W, C]
        elif len(image_batch.shape) == 3 and image_batch.shape[0] == 3:
            image_batch = image_batch.permute(1, 2, 0).unsqueeze(0)
            
        # If we have [B, C, H, W] format, convert to [B, H, W, C]
        elif len(image_batch.shape) == 4 and image_batch.shape[1] == 3:
            image_batch = image_batch.permute(0, 2, 3, 1)
            
        print(f"[DEBUG] Reshaped to: {image_batch.shape}")
    
    # Ensure we have proper [B, H, W, C] format with batch size matching crop_regions
    if image_batch.shape[0] != len(crop_regions):
        print(f"[DEBUG] Warning: Batch size {image_batch.shape[0]} doesn't match crop count {len(crop_regions)}")
        
        # If we have more crops than batch items, repeat the batch
        if image_batch.shape[0] < len(crop_regions):
            repeats = math.ceil(len(crop_regions) / image_batch.shape[0])
            image_batch = image_batch.repeat(repeats, 1, 1, 1)[:len(crop_regions)]
            
        # If we have more batch items than crops, take only what we need
        elif image_batch.shape[0] > len(crop_regions):
            image_batch = image_batch[:len(crop_regions)]
    
    save_debug_image(image_batch, "prepared_crop_batch")
    return image_batch

def calculate_bucket_dimensions(width, height, scale_mode, short_edge_div, mask_dilation=1.0, bucket_scale_factor=1.0):
    """
    Calculate the target dimensions based on bucket constraints.
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
    target_long_edge = int(target_long_edge * bucket_scale_factor)
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

def adjust_crop_to_bucket(crop_region, original_img_size, scale_mode, short_edge_div, mask_dilation=1.0, bucket_scale_factor=1.0):
    """
    Adjust a crop region to fit into the target bucket dimensions.
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
        crop_width, crop_height, scale_mode, short_edge_div, mask_dilation, bucket_scale_factor
    )
    
    # Track if we need to downscale
    needs_downscale = False
    if target_width < crop_width or target_height < crop_height:
        needs_downscale = True
    
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
    
    return adjusted_crop, target_dims, scale_type, bucket_size, needs_downscale

def process_segs_with_buckets(segs, original_image, scale_mode, short_edge_div, mask_dilation=1.0, bucket_scale_factor=1.0):
    """
    Process SEGS to fit into specified buckets.
    Returns individual crop tensors (not stacked) to handle different dimensions.
    """
    
    start_time = time.time()
    print(f"[USDU_DEBUG] Starting bucket processing for {len(segs[1]) if segs and len(segs) > 1 else 0} segments")
    
    if segs is None or len(segs[1]) == 0:
        # Create a 1x1 black pixel image tensor for empty SEGS
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return [empty_tensor], [], [], [], []
    
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
        adjusted_crop, target_dims, scale_type, bucket_size, needs_downscale = adjust_crop_to_bucket(
            (x1, y1, x2, y2), original_img_size, scale_mode, short_edge_div, mask_dilation, bucket_scale_factor
        )
        
        # Updated logic
        ax1, ay1, ax2, ay2 = adjusted_crop
        
        # Crop the image with adjusted coordinates
        cropped_image = original_image[:, ay1:ay2, ax1:ax2, :]
        
        # If needs downscaling, do it immediately with Lanczos
        if needs_downscale and scale_type == "down":
            target_w, target_h = target_dims
            from torchvision.transforms.functional import resize
            # Reshape for resize ([B,H,W,C] -> [B,C,H,W])
            reshaped = cropped_image.permute(0, 3, 1, 2)
            # Resize
            resized = resize(reshaped, [target_h, target_w], antialias=True)
            # Back to original shape
            cropped_image = resized.permute(0, 2, 3, 1)
            print(f"[USDU] Pre-downscaled region from {ay2-ay1}x{ax2-ax1} to {target_h}x{target_w}")
        
        # Store information
        crop_regions.append(adjusted_crop)
        target_dimensions.append(target_dims)
        scale_types.append(scale_type)
        bucket_sizes.append(bucket_size)
        image_crops.append(cropped_image)
    
    # Return the list of crops - DO NOT stack them since they have different dimensions
    if len(image_crops) > 0:
        # Save a sample crop for debugging
        save_debug_image(image_crops[0], "bucket_processed_crop_sample")
        print(f"[USDU_DEBUG] Bucket processing completed in {time.time() - start_time:.2f}s, found {len(crop_regions)} regions")
        return image_crops, crop_regions, target_dimensions, scale_types, bucket_sizes
    else:
        # Fallback 1x1 black pixel for safety
        empty_tensor = torch.zeros(1, 1, 1, 3)
        print(f"[USDU_DEBUG] Bucket processing failed in {time.time() - start_time:.2f}s, found no regions")
        return [empty_tensor], [], [], [], []

def segs_to_image_batch(segs, original_image):
    """
    Convert SEGS to a list of individual image crops for processing.
    Returns a list of image tensors rather than a batched tensor.
    """
    print(f"[DEBUG] Original image shape: {original_image.shape}")
    
    if segs is None or len(segs[1]) == 0:
        print("[DEBUG] No segments found in segs!")
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return [empty_tensor], []
    
    image_crops = []
    crop_regions = []
    
    print(f"[DEBUG] Creating crops from {len(segs[1])} segments")
    for i, seg in enumerate(segs[1]):
        # Get crop region
        x1, y1, x2, y2 = seg.crop_region
        print(f"[DEBUG] Segment {i} - Crop region: {x1},{y1} to {x2},{y2} size: {x2-x1}x{y2-y1}")
        
        # Validate crop region
        if x1 >= x2 or y1 >= y2:
            print(f"[DEBUG] WARNING: Invalid crop region {x1},{y1} to {x2},{y2}")
            continue
            
        if x2 > original_image.shape[2] or y2 > original_image.shape[1]:
            print(f"[DEBUG] WARNING: Crop region outside image bounds. Image: {original_image.shape[1:3]}, Crop: {y2}x{x2}")
            # Clamp values
            x2 = min(x2, original_image.shape[2])
            y2 = min(y2, original_image.shape[1])
            
        crop_regions.append((x1, y1, x2, y2))  # Store validated region
        
        # Explicitly crop the image and ensure proper dimensions
        try:
            cropped_image = original_image[:, y1:y2, x1:x2, :]
            print(f"[DEBUG] Cropped shape: {cropped_image.shape}")
            
            # Additional validation
            if cropped_image.numel() == 0:
                print(f"[DEBUG] WARNING: Empty crop for segment {i}")
                continue
                
            image_crops.append(cropped_image)
        except Exception as e:
            print(f"[DEBUG] Error cropping segment {i}: {e}")
    
    # Return the list of crops without stacking
    if len(image_crops) > 0:
        print(f"[DEBUG] Created {len(image_crops)} individual crops")
        save_debug_image(image_crops[0], "segs_extracted_crop_example") 
        return image_crops, crop_regions
    else:
        print("[DEBUG] No valid crops created!")
        empty_tensor = torch.zeros(1, 1, 1, 3)
        return [empty_tensor], []

# Export the SimpleDetectorForEach class for segmentation
class SimpleDetectorForEach:
    """
    Class providing detection and segmentation functionality.
    This embedded implementation avoids external dependencies.
    """
    
    @classmethod
    def detect(cls, detector, image, threshold, dilation, crop_factor, drop_size, 
               crop_min_size=0, crop_square=False, detailer_hook=None, bbox_fill=0.5, segment_threshold=0.5,
               sam_model_opt=None, segm_detector_opt=None):
        """
        Run detection on an image.
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
        """
        # Basic segmentation refinement logic
        if segs is None or segs[1] is None or len(segs[1]) == 0:
            return segs
            
        try:
            # Use the segmentation detector to refine the bounding boxes
            refined_segs = segm_detector.detect(segs, sam_model=sam_model, threshold=threshold)
            return refined_segs
        except Exception as e:
            print(f"[DEBUG] Segmentation refinement error: {e}")
            return segs