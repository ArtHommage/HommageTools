"""
File: homage_tools/nodes/ht_mask_dilation_node.py
Version: 2.1.1
Description: Node for cropping images to mask content and calculating scaling factors with batch support
"""

import torch
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "2.1.1"
STANDARD_BUCKETS = [512, 768, 1024]

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def find_mask_bounds(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Find the bounding box of non-zero mask content."""
    # Debug mask information
    print(f"DEBUG: Mask shape: {mask.shape}")
    print(f"DEBUG: Mask min/max values: {mask.min().item():.6f}/{mask.max().item():.6f}")
    
    # Extract first channel for calculation (assuming HWC format)
    if len(mask.shape) == 3:  # HWC format
        if mask.shape[-1] > 1:  # Multiple channels
            mask_2d = mask[..., 0]  # First channel
        else:
            mask_2d = mask[..., 0]  # Single channel
    else:
        mask_2d = mask
        
    print(f"DEBUG: Extracted 2D mask shape: {mask_2d.shape}")
    
    # Find non-zero indices
    indices = torch.nonzero(mask_2d > 0)
    print(f"DEBUG: Non-zero indices count: {len(indices)}")
    
    if len(indices) == 0:
        print("DEBUG: Empty mask detected")
        return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
    
    # Get bounds
    min_y = indices[:, 0].min().item()
    max_y = indices[:, 0].max().item() + 1
    min_x = indices[:, 1].min().item()
    max_x = indices[:, 1].max().item() + 1
    
    print(f"DEBUG: Bounds - Y: {min_y} to {max_y}, X: {min_x} to {max_x}")
    return min_y, max_y, min_x, max_x

def calculate_target_size(bbox_size: int, scale_mode: str) -> int:
    """Calculate target size based on scale mode."""
    print(f"DEBUG: Original size: {bbox_size}, Mode: {scale_mode}")
    
    if scale_mode == "Scale Closest":
        target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
    elif scale_mode == "Scale Up":
        target = next((x for x in STANDARD_BUCKETS if x >= bbox_size), STANDARD_BUCKETS[-1])
    elif scale_mode == "Scale Down":
        target = next((x for x in reversed(STANDARD_BUCKETS) if x <= bbox_size), STANDARD_BUCKETS[0])
    elif scale_mode == "Scale Max":
        target = STANDARD_BUCKETS[-1]
    else:
        target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
    
    print(f"DEBUG: Target size: {target}")
    return target

#------------------------------------------------------------------------------
# Section 3: Batch Processing Functions
#------------------------------------------------------------------------------
def detect_tensor_format(tensor: torch.Tensor) -> str:
    """
    Detect the format of a tensor based on its shape.
    
    Returns:
        str: Format name - 'BHWC', 'BCHW', 'HWC', 'CHW', or 'UNKNOWN'
    """
    shape = tensor.shape
    
    if len(shape) == 4:
        # Could be BHWC or BCHW
        if shape[1] <= 4 and shape[2] > 4 and shape[3] > 4:
            # BCHW - batch, channels, height, width
            return 'BCHW'
        elif shape[3] <= 4 and shape[1] > 4 and shape[2] > 4:
            # BHWC - batch, height, width, channels
            return 'BHWC'
    elif len(shape) == 3:
        # Could be HWC or CHW
        if shape[2] <= 4 and shape[0] > 4 and shape[1] > 4:
            # HWC - height, width, channels
            return 'HWC'
        elif shape[0] <= 4 and shape[1] > 4 and shape[2] > 4:
            # CHW - channels, height, width
            return 'CHW'
        
    # If we can't determine the format with confidence
    return 'UNKNOWN'

def normalize_to_bhwc(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Convert tensor to BHWC format regardless of initial format."""
    format_name = detect_tensor_format(tensor)
    print(f"DEBUG: Detected {name} format: {format_name}")
    
    if format_name == 'BCHW':
        # BCHW -> BHWC
        tensor = tensor.permute(0, 2, 3, 1)
        print(f"DEBUG: Converted {name} from BCHW to BHWC: {tensor.shape}")
    elif format_name == 'HWC':
        # HWC -> BHWC
        tensor = tensor.unsqueeze(0)
        print(f"DEBUG: Converted {name} from HWC to BHWC: {tensor.shape}")
    elif format_name == 'CHW':
        # CHW -> BHWC
        tensor = tensor.unsqueeze(0).permute(0, 2, 3, 1)
        print(f"DEBUG: Converted {name} from CHW to BHWC: {tensor.shape}")
    elif format_name == 'UNKNOWN':
        print(f"WARNING: Could not determine {name} format: {tensor.shape}")
        # Try to make a best guess based on shape
        if len(tensor.shape) == 4:
            # Already 4D, assume it needs permutation
            tensor = tensor.permute(0, 2, 3, 1)
            print(f"DEBUG: Attempting to convert unknown format to BHWC: {tensor.shape}")
        elif len(tensor.shape) == 3:
            # 3D tensor, add batch dimension
            tensor = tensor.unsqueeze(0)
            print(f"DEBUG: Added batch dimension to unknown format: {tensor.shape}")
            
    # Final check for batch size
    print(f"DEBUG: Final {name} shape: {tensor.shape} (BHWC format)")
    print(f"DEBUG: {name} batch size: {tensor.shape[0]}")
    
    return tensor

def detect_mask_batch_structure(mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Detect if what appears to be channels might actually be batch items.
    
    Returns:
        Tuple[torch.Tensor, int]: Restructured tensor and actual batch size
    """
    # Check if we have a tensor that might be misinterpreted as single batch with channels
    if len(mask.shape) == 4 and mask.shape[0] == 1 and mask.shape[3] > 1:
        # This might be a batch incorrectly formatted
        original_shape = mask.shape
        
        # Try to interpret the channels as batch items
        print(f"DEBUG: Possible batch structure detected in mask: {mask.shape}")
        # Reshape to separate batch items
        batch_size = mask.shape[3]
        height = mask.shape[1]
        width = mask.shape[2]
        
        # Reshape mask to have proper batch dimension
        mask = mask.permute(3, 1, 2, 0)  # Move channels to batch position
        
        print(f"DEBUG: Restructured mask from {original_shape} to {mask.shape}")
        return mask, batch_size
    
    # No restructuring needed
    return mask, mask.shape[0]

def process_single_mask(
    image_single: torch.Tensor,
    mask_single: torch.Tensor,
    scale_mode: str,
    padding: int,
    mask_index: int
) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """Process a single mask-image pair."""
    print(f"\n==== Processing Mask #{mask_index+1} ====")
    
    # Find mask bounds
    min_y, max_y, min_x, max_x = find_mask_bounds(mask_single)
    
    # Verify bounds are sensible
    print(f"DEBUG: Image dimensions: {image_single.shape[0]}x{image_single.shape[1]}")
    print(f"DEBUG: Bounds check - X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
    
    # Calculate bounding box dimensions
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    long_edge = max(bbox_width, bbox_height)
    
    print(f"DEBUG: Bounding box size: {bbox_width}x{bbox_height}")
    print(f"DEBUG: Long edge: {long_edge}")
    
    # Empty mask check
    if bbox_width <= 1 or bbox_height <= 1:
        print(f"WARNING: Mask #{mask_index+1} is nearly empty. Using full image.")
        return (mask_single, image_single, image_single.shape[1], image_single.shape[0], 1.0)
    
    # Calculate target size and scale factor
    target_size = calculate_target_size(long_edge, scale_mode)
    scale_factor = target_size / long_edge
    
    print(f"DEBUG: Scale factor for mask #{mask_index+1}: {scale_factor:.4f}")
    
    # Crop image and mask to the mask content bounds
    cropped_image = image_single[min_y:max_y, min_x:max_x, :]
    cropped_mask = mask_single[min_y:max_y, min_x:max_x, :]
    
    print(f"DEBUG: Cropped image shape: {cropped_image.shape}")
    print(f"DEBUG: Cropped mask shape: {cropped_mask.shape}")
    
    return cropped_mask, cropped_image, bbox_width, bbox_height, scale_factor

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTMaskDilationNode:
    """Node for cropping images to mask content and calculating scaling factors."""
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_mask"
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("dilated_mask", "cropped_image", "width", "height", "scale_factor")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale_mode": (["Scale Closest", "Scale Up", "Scale Down", "Scale Max"], {
                    "default": "Scale Closest"
                }),
                "padding": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 8
                })
            }
        }

    def process_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        scale_mode: str,
        padding: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]:
        """Process mask and image based on mask content."""
        print(f"\n===== HTMaskDilationNode v{VERSION} - Processing =====")
        
        try:
            # Print original shapes
            print(f"DEBUG: Original image shape: {image.shape}")
            print(f"DEBUG: Original mask shape: {mask.shape}")
            
            # First normalize to BHWC format
            image = normalize_to_bhwc(image, "image")
            
            # Check for special case with mask - it might have batch items disguised as channels
            mask = normalize_to_bhwc(mask, "mask")
            mask, actual_batch_size = detect_mask_batch_structure(mask)
            
            # Get actual batch sizes
            mask_batch_size = mask.shape[0]
            image_batch_size = image.shape[0]
            
            print(f"\n==== BATCH INFORMATION ====")
            print(f"Mask batch size: {mask_batch_size}")
            print(f"Image batch size: {image_batch_size}")
            
            # Make sure batch sizes match or handle mismatch
            if mask_batch_size != image_batch_size:
                print(f"WARNING: Batch size mismatch - Mask: {mask_batch_size}, Image: {image_batch_size}")
                if mask_batch_size > image_batch_size:
                    # Repeat image to match mask batch size
                    image = image.repeat(mask_batch_size, 1, 1, 1)
                    batch_size = mask_batch_size
                    print(f"Repeated image to match mask batch size: {image.shape}")
                else:
                    # Use the mask batch size, repeat as needed
                    batch_size = mask_batch_size
                    print(f"Using mask batch size: {batch_size}")
            else:
                batch_size = mask_batch_size
            
            # Ensure each mask has only one channel (but preserving batch dimension)
            if mask.shape[-1] > 1:
                print(f"DEBUG: Each mask has {mask.shape[-1]} channels, converting to single channel")
                mask = mask.mean(dim=-1, keepdim=True)
                print(f"DEBUG: New mask shape: {mask.shape}")
            
            # Process each mask-image pair
            cropped_masks = []
            cropped_images = []
            widths = []
            heights = []
            scale_factors = []
            
            for i in range(batch_size):
                print(f"\nProcessing batch item {i+1}/{batch_size}")
                
                # Extract single image and mask (preserve HWC format)
                img_single = image[i]  # HWC format
                mask_single = mask[i]  # HWC format
                
                # Process single mask-image pair
                c_mask, c_image, width, height, scale_factor = process_single_mask(
                    img_single, mask_single, scale_mode, padding, i
                )
                
                # Store results
                cropped_masks.append(c_mask)
                cropped_images.append(c_image)
                widths.append(width)
                heights.append(height)
                scale_factors.append(scale_factor)
            
            # Find maximum dimensions
            max_h = max([m.shape[0] for m in cropped_masks])
            max_w = max([m.shape[1] for m in cropped_masks])
            
            print(f"\n==== BATCH RESULTS ====")
            print(f"Maximum dimensions across batch: {max_w}x{max_h}")
            
            # Pad each mask and image to the maximum size
            padded_masks = []
            padded_images = []
            
            for i in range(batch_size):
                mask_h, mask_w = cropped_masks[i].shape[:2]
                pad_h = max_h - mask_h
                pad_w = max_w - mask_w
                
                print(f"Mask #{i+1}: Original size={mask_w}x{mask_h}, Padding needed={pad_w}x{pad_h}")
                
                # Pad dimensions
                if pad_h > 0 or pad_w > 0:
                    # For HWC format, padding is (left, right, top, bottom, channel_front, channel_back)
                    # In PyTorch, padding starts from the last dimension and moves backward
                    pad_config = (0, 0, 0, pad_w, 0, pad_h)
                    
                    # Pad mask
                    padded_mask = torch.nn.functional.pad(
                        cropped_masks[i], pad_config, mode='constant', value=0
                    )
                    padded_masks.append(padded_mask)
                    
                    # Pad image
                    padded_image = torch.nn.functional.pad(
                        cropped_images[i], pad_config, mode='constant', value=0
                    )
                    padded_images.append(padded_image)
                else:
                    # No padding needed
                    padded_masks.append(cropped_masks[i])
                    padded_images.append(cropped_images[i])
            
            # Stack masks and images into batches
            result_masks = torch.stack(padded_masks, dim=0)
            result_images = torch.stack(padded_images, dim=0)
            
            # Verify output shapes are correct
            print(f"\n==== FINAL OUTPUT ====")
            print(f"Output batch shape - Masks: {result_masks.shape}, Images: {result_images.shape}")
            print(f"Using width={widths[0]}, height={heights[0]}, scale_factor={scale_factors[0]:.4f}")
            
            # Final verification and fixing
            if len(result_masks.shape) != 4 or result_masks.shape[-1] != 1:
                print(f"WARNING: Fixing mask format: {result_masks.shape}")
                if len(result_masks.shape) == 3:
                    result_masks = result_masks.unsqueeze(-1)
                
            if len(result_images.shape) != 4 or result_images.shape[-1] != 3:
                print(f"WARNING: Fixing image format: {result_images.shape}")
                if len(result_images.shape) == 3:
                    # Add channel dimension
                    if result_images.shape[-1] != 3:
                        result_images = result_images.unsqueeze(-1).repeat(1, 1, 1, 3)
                    else:
                        result_images = result_images.unsqueeze(0)
            
            print(f"Final verification - Masks: {result_masks.shape}, Images: {result_images.shape}")
            print(f"============================")
            
            # Return values
            return (result_masks, result_images, widths[0], heights[0], scale_factors[0])
            
        except Exception as e:
            logger.error(f"Error in mask dilation: {str(e)}")
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to return in expected format
            try:
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(0)
                return (mask, image, image.shape[2], image.shape[1], 1.0)
            except:
                return (mask, image, 0, 0, 1.0)