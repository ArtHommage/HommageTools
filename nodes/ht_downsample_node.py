"""
File: homage_tools/nodes/ht_resolution_downsample_node.py
Version: 1.3.0
Description: Node for downsampling images to a target resolution with improved tensor handling
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union, List
import math
import logging

# Set up logger
logger = logging.getLogger('HommageTools')
# Set minimum level to show all messages
logger.setLevel(logging.DEBUG)
# Create console handler to ensure output is visible
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Create a formatter with timestamps
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(ch)

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.3.0"
MIN_ALLOWED_SIZE = 8  # Minimum allowed dimension

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_dimensions(tensor: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """Verify and extract dimensions from BHWC tensor."""
    shape = tensor.shape
    logger.debug(f"{context} - Tensor shape: {shape}")
    logger.debug(f"{context} - Tensor dtype: {tensor.dtype}")
    
    # Add NaN and infinity checks
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        logger.warning(f"{context} - WARNING: Tensor contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Infinity' if has_inf else ''} values!")
    
    if len(shape) == 3:  # HWC format
        height, width, channels = shape
        batch = 1
        logger.debug(f"{context} - HWC format detected")
        
        # Validate channel dimension
        if channels not in [1, 3, 4]:
            logger.warning(f"{context} - WARNING: Unusual channel count: {channels} (expected 1, 3, or 4)")
            
    elif len(shape) == 4:  # BHWC format
        batch, height, width, channels = shape
        logger.debug(f"{context} - BHWC format detected")
        
        # Validate channel dimension
        if channels not in [1, 3, 4]:
            logger.warning(f"{context} - WARNING: Unusual channel count: {channels} (expected 1, 3, or 4)")
            
    else:
        logger.error(f"{context} - ERROR: Invalid tensor shape: {shape}")
        logger.info(f"{context} - Attempting to recover...")
        
        # Try to recover based on dimension sizes
        if len(shape) == 2:  # Might be a single-channel image without batch
            height, width = shape
            channels = 1
            batch = 1
            logger.info(f"{context} - Recovered as HW format: {batch}x{height}x{width}x{channels}")
        elif len(shape) == 3:
            if shape[0] in [1, 3, 4]:  # Might be CHW format
                channels, height, width = shape
                batch = 1
                logger.info(f"{context} - Recovered as CHW format: {batch}x{height}x{width}x{channels}")
            else:
                batch, height, width = shape
                channels = 1
                logger.info(f"{context} - Recovered as BHW format: {batch}x{height}x{width}x{channels}")
        else:
            raise ValueError(f"Invalid tensor shape: {shape}, cannot recover")
        
    logger.debug(f"{context} - Dimensions: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def verify_mask_dimensions(mask: torch.Tensor, context: str) -> Tuple[int, int, int]:
    """Verify and extract dimensions from mask tensor."""
    shape = mask.shape
    logger.debug(f"{context} - Mask shape: {shape}")
    
    if len(shape) == 2:  # HW format
        height, width = shape
        batch = 1
        logger.debug(f"{context} - HW format detected")
    elif len(shape) == 3:  # BHW format
        batch, height, width = shape
        logger.debug(f"{context} - BHW format detected")
    else:
        raise ValueError(f"Invalid mask shape: {shape}")
        
    logger.debug(f"{context} - Mask dimensions: {batch}x{height}x{width}")
    return batch, height, width

def calculate_target_dimensions(
    current_height: int,
    current_width: int,
    target_long_edge: int
) -> Tuple[int, int, float]:
    """Calculate target dimensions maintaining aspect ratio."""
    current_long_edge = max(current_height, current_width)
    aspect_ratio = current_width / current_height
    
    scale_factor = target_long_edge / current_long_edge
    logger.debug(f"Scale factor: {scale_factor:.3f}")
    
    if current_width >= current_height:
        new_width = target_long_edge
        new_height = int(round(new_width / aspect_ratio))
    else:
        new_height = target_long_edge
        new_width = int(round(new_height * aspect_ratio))
    
    # Enforce minimum dimensions
    new_height = max(MIN_ALLOWED_SIZE, new_height)
    new_width = max(MIN_ALLOWED_SIZE, new_width)
        
    logger.debug(f"Target dimensions: {new_width}x{new_height}")
    return new_height, new_width, scale_factor

#------------------------------------------------------------------------------
# Section 3: Input Normalization
#------------------------------------------------------------------------------
def normalize_input(image_input: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
    """
    Normalize input to a standard tensor format, handling both tensor and list formats.
    
    Args:
        image_input: Either a tensor or a list of tensors
        
    Returns:
        torch.Tensor: Normalized tensor in BHWC format
    """
    # Detailed input reporting
    logger.info("\n" + "="*50)
    logger.info("INPUT ANALYSIS:")
    logger.info(f"Input type: {type(image_input).__name__}")
    
    # Handle list of tensors
    if isinstance(image_input, list):
        logger.info(f"List input detected with {len(image_input)} elements")
        if not image_input:
            logger.warning("WARNING: Empty list input, creating placeholder tensor")
            # Empty list - create a valid tensor with minimum dimensions
            return torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, 3)
        
        # Report on list elements
        for i, tensor in enumerate(image_input[:3]):  # Show up to first 3 elements
            logger.debug(f"  List element {i}: type={type(tensor).__name__}, shape={tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")
        if len(image_input) > 3:
            logger.debug(f"  ... and {len(image_input) - 3} more elements")
            
        # Try to stack tensors if they have consistent shapes
        first_shape = image_input[0].shape
        if all(t.shape == first_shape for t in image_input):
            stacked = torch.cat(image_input, dim=0)
            logger.info(f"Stacked {len(image_input)} tensors into shape {stacked.shape}")
            return stacked
        else:
            # Report shape inconsistency
            logger.warning("WARNING: Inconsistent tensor shapes in list:")
            shapes = [t.shape for t in image_input]
            unique_shapes = set(str(s) for s in shapes)
            for shape in unique_shapes:
                count = sum(1 for s in shapes if str(s) == shape)
                logger.debug(f"  Shape {shape}: {count} tensors")
            logger.info(f"Using first tensor of shape {first_shape}")
            return image_input[0]
    
    # Regular tensor - ensure it has proper dimensions
    if isinstance(image_input, torch.Tensor):
        logger.info(f"Tensor input detected with shape {image_input.shape} and dtype {image_input.dtype}")
        
        # Ensure tensor has 4 dimensions (BHWC)
        if len(image_input.shape) == 3:  # HWC format
            logger.debug("HWC format detected, adding batch dimension")
            return image_input.unsqueeze(0)  # Add batch dimension
        elif len(image_input.shape) == 4:  # Already BHWC
            logger.debug("BHWC format confirmed")
            return image_input
        else:
            # Try to recover from unusual format
            logger.warning(f"WARNING: Unusual tensor shape {image_input.shape}")
            try:
                if len(image_input.shape) == 2:  # HW format (grayscale)
                    logger.debug("HW format detected, adding batch and channel dimensions")
                    return image_input.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimensions
                # For other formats, create a valid tensor with minimum dimensions
                logger.info(f"Creating valid tensor as fallback")
                return torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, 3)
            except Exception as e:
                logger.error(f"ERROR: Failed to normalize tensor: {e}")
                return torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, 3)
    
    # Unknown input type - return valid tensor with minimum dimensions
    logger.warning(f"WARNING: Unknown input type {type(image_input)}, creating valid tensor")
    return torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, 3)

#------------------------------------------------------------------------------
# Section 4: Processing Functions
#------------------------------------------------------------------------------
def process_downsample(
    image: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """Process downsampling operation maintaining BHWC format."""
    logger.info(f"Processing downsample: shape={image.shape} (BHWC)")
    logger.info(f"Target size: {target_width}x{target_height}")
    
    # Check for unusual tensor shapes or values
    if len(image.shape) != 4:
        logger.warning(f"[WARNING] Expected 4D tensor (BHWC) but got {len(image.shape)}D")
        # Try to reshape if needed
        if len(image.shape) == 3:  # HWC format
            logger.debug(f"[DEBUG] Converting HWC to BHWC format")
            image = image.unsqueeze(0)
    
    # Check for NaN or infinity
    has_nan = torch.isnan(image).any().item()
    has_inf = torch.isinf(image).any().item()
    if has_nan or has_inf:
        logger.warning(f"[WARNING] Image tensor contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Infinity' if has_inf else ''} values!")
        # Try to fix by clipping values
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        logger.debug(f"[DEBUG] Applied nan_to_num to fix invalid values")
    
    # CRITICAL: Enforce minimum dimensions for target
    if target_height < MIN_ALLOWED_SIZE or target_width < MIN_ALLOWED_SIZE:
        logger.warning(f"[WARNING] Target dimensions too small: {target_width}x{target_height}, increasing to minimum size of {MIN_ALLOWED_SIZE}")
        target_height = max(MIN_ALLOWED_SIZE, target_height)
        target_width = max(MIN_ALLOWED_SIZE, target_width)
    
    # Ensure tensor has proper shape and values
    if image.shape[1] == 0 or image.shape[2] == 0 or image.shape[1] < MIN_ALLOWED_SIZE or image.shape[2] < MIN_ALLOWED_SIZE:
        logger.warning(f"[WARNING] Invalid or too small image dimensions: {image.shape}")
        # Create a valid tensor with minimum dimensions
        return torch.zeros(image.shape[0], max(MIN_ALLOWED_SIZE, target_height), max(MIN_ALLOWED_SIZE, target_width), image.shape[3], device=device)
    
    # Move to processing device
    image = image.to(device)
    
    try:
        # Convert BHWC to BCHW for interpolation
        x = image.permute(0, 3, 1, 2)
        logger.debug(f"Converted to BCHW: shape={x.shape}")
        
        # Ensure no extreme values
        if x.max().item() > 10.0 or x.min().item() < -10.0:
            logger.warning(f"[WARNING] Image has extreme values: min={x.min().item():.3f}, max={x.max().item():.3f}")
            logger.debug(f"[DEBUG] Clipping values to [0, 1] range")
            x = torch.clamp(x, 0.0, 1.0)
        
        # Determine antialiasing based on interpolation mode
        use_antialias = interpolation in ['bilinear', 'bicubic', 'lanczos']
        
        # Handle lanczos mode
        if interpolation == 'lanczos':
            interpolation = 'bicubic'
        
        # CRITICAL: Double-check target dimensions one more time
        if target_height < MIN_ALLOWED_SIZE or target_width < MIN_ALLOWED_SIZE:
            target_height = max(MIN_ALLOWED_SIZE, target_height)
            target_width = max(MIN_ALLOWED_SIZE, target_width)
            logger.debug(f"[DEBUG] Final target size adjusted to: {target_width}x{target_height}")
        
        # Process
        result = F.interpolate(
            x,
            size=(target_height, target_width),
            mode=interpolation,
            antialias=use_antialias if use_antialias else None,
            align_corners=None if interpolation == 'nearest' else False
        )
        
        # Convert back to BHWC
        result = result.permute(0, 2, 3, 1)
        logger.debug(f"Converted back to BHWC: shape={result.shape}")
        
        # CRITICAL: Verify output dimensions
        if result.shape[1] < MIN_ALLOWED_SIZE or result.shape[2] < MIN_ALLOWED_SIZE:
            logger.error(f"[ERROR] Output dimensions still too small: {result.shape}")
            # Create a valid tensor with minimum dimensions
            result = torch.zeros(result.shape[0], MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, result.shape[3], device=device)
            logger.debug(f"[DEBUG] Created fallback tensor: {result.shape}")
        
        return result
    except Exception as e:
        logger.error(f"[ERROR] Downsample processing error: {str(e)}")
        logger.debug(f"[DEBUG] Error details: {e.__class__.__name__}")
        # Return fallback tensor with valid dimensions
        logger.debug(f"[DEBUG] Creating fallback tensor due to error")
        fallback = torch.zeros(image.shape[0], max(MIN_ALLOWED_SIZE, target_height), max(MIN_ALLOWED_SIZE, target_width), image.shape[3], device=device)
        logger.debug(f"[DEBUG] Fallback tensor shape: {fallback.shape}")
        return fallback

def process_mask_downsample(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    interpolation: str,
    device: torch.device
) -> torch.Tensor:
    """Process mask downsampling operation."""
    logger.info(f"Processing mask downsample: shape={mask.shape}")
    logger.info(f"Target size: {target_width}x{target_height}")
    
    # Move to processing device
    mask = mask.to(device)
    
    # Ensure mask has batch dimension (BHW format)
    if len(mask.shape) == 2:  # HW format
        mask = mask.unsqueeze(0)
    
    # CRITICAL: Enforce minimum dimensions
    if target_height < MIN_ALLOWED_SIZE or target_width < MIN_ALLOWED_SIZE:
        logger.warning(f"[WARNING] Target dimensions too small: {target_width}x{target_height}, increasing to minimum size")
        target_height = max(MIN_ALLOWED_SIZE, target_height)
        target_width = max(MIN_ALLOWED_SIZE, target_width)
    
    # Add channel dimension for interpolation (BCHW format)
    x = mask.unsqueeze(1)
    logger.debug(f"Converted to BCHW: shape={x.shape}")
    
    # Determine antialiasing based on interpolation mode
    use_antialias = interpolation in ['bilinear', 'bicubic', 'lanczos']
    
    # Handle lanczos mode
    if interpolation == 'lanczos':
        interpolation = 'bicubic'
    
    # Process
    try:
        result = F.interpolate(
            x,
            size=(target_height, target_width),
            mode=interpolation,
            antialias=use_antialias if use_antialias else None,
            align_corners=None if interpolation == 'nearest' else False
        )
        
        # Remove channel dimension (back to BHW)
        result = result.squeeze(1)
        logger.debug(f"Converted back to BHW: shape={result.shape}")
        
        # CRITICAL: Verify output dimensions
        if result.shape[1] < MIN_ALLOWED_SIZE or result.shape[2] < MIN_ALLOWED_SIZE:
            logger.error(f"[ERROR] Mask output dimensions too small: {result.shape}")
            # Create a valid tensor with minimum dimensions
            result = torch.zeros(result.shape[0], MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, device=device)
            logger.debug(f"[DEBUG] Created fallback mask: {result.shape}")
        
        return result
    except Exception as e:
        logger.error(f"[ERROR] Mask downsample processing error: {str(e)}")
        # Create a valid mask tensor as fallback
        fallback = torch.zeros(mask.shape[0], max(MIN_ALLOWED_SIZE, target_height), max(MIN_ALLOWED_SIZE, target_width), device=device)
        logger.debug(f"[DEBUG] Fallback mask shape: {fallback.shape}")
        return fallback

def create_empty_mask(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create empty mask tensor with correct dimensions."""
    # Ensure minimum dimensions
    height = max(MIN_ALLOWED_SIZE, height)
    width = max(MIN_ALLOWED_SIZE, width)
    return torch.zeros((batch_size, height, width), device=device)

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
#------------------------------------------------------------------------------
class HTResolutionDownsampleNode:
    """Downsamples images to a target resolution with mask support."""
    
    CATEGORY = "HommageTools/Image"
    FUNCTION = "downsample_to_resolution"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("downsampled_image", "downsampled_mask")
    
    INTERPOLATION_MODES = ["nearest", "bilinear", "bicubic", "area", "lanczos"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "target_long_edge": ("INT", {
                    "default": 1024,
                    "min": MIN_ALLOWED_SIZE,
                    "max": 8192,
                    "step": 8,
                    "description": "Target size for longest edge"
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "bicubic",
                    "description": "Interpolation method for images"
                }),
                "crop_to_mask": ("BOOLEAN", {
                    "default": False,
                    "description": "Crop to mask content before scaling"
                })
            },
            "optional": {
                "mask": ("MASK", {
                    "description": "Optional mask to process"
                }),
                "mask_interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "nearest",
                    "description": "Interpolation method for masks"
                })
            }
        }

    def downsample_to_resolution(
        self,
        image: Union[torch.Tensor, List[torch.Tensor]],
        target_long_edge: int,
        interpolation: str,
        crop_to_mask: bool,
        mask: Optional[torch.Tensor] = None,
        mask_interpolation: Optional[str] = "nearest"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample image and mask to target resolution."""
        logger.info(f"\n{'='*50}")
        logger.info(f"HTResolutionDownsampleNode v{VERSION} - Processing")
        logger.info(f"{'='*50}")
        
        # Input analysis section
        logger.info(f"\n{'='*50}")
        logger.info("INPUT PARAMETERS:")
        logger.info(f"Target long edge: {target_long_edge}")
        logger.info(f"Interpolation: {interpolation}")
        logger.info(f"Crop to mask: {crop_to_mask}")
        logger.info(f"Mask interpolation: {mask_interpolation}")
        
        # CRITICAL: Enforce minimum target size
        if target_long_edge < MIN_ALLOWED_SIZE:
            logger.warning(f"[WARNING] Target long edge too small: {target_long_edge}, increasing to {MIN_ALLOWED_SIZE}")
            target_long_edge = MIN_ALLOWED_SIZE
        
        if mask is not None:
            logger.info(f"Mask provided: shape={mask.shape}, dtype={mask.dtype}")
            value_range = f"min={mask.min().item():.3f}, max={mask.max().item():.3f}"
            logger.debug(f"Mask value range: {value_range}")
        else:
            logger.info("No mask provided")
        
        try:
            # Normalize input to a standard format with detailed reporting
            image = normalize_input(image)
            
            # Verify input image tensor
            batch, height, width, channels = verify_tensor_dimensions(image, "Input image")
            value_range = f"min={image.min().item():.3f}, max={image.max().item():.3f}"
            logger.debug(f"Image value range: {value_range}")
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Processing device: {device}")
            
            # Special case for minimal inputs (can happen from detection nodes)
            if height <= MIN_ALLOWED_SIZE or width <= MIN_ALLOWED_SIZE:
                logger.warning(f"[WARNING] Input dimensions too small ({width}x{height}), creating valid output")
                # Create reasonable minimal output
                result_image = torch.zeros(batch, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, channels, device=device)
                result_mask = torch.zeros(batch, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, device=device)
                
                # Output reporting
                logger.info(f"\n{'='*50}")
                logger.info("OUTPUT SUMMARY (MINIMAL VALID OUTPUT):")
                logger.info(f"Image: shape={result_image.shape}, dtype={result_image.dtype}")
                logger.info(f"Mask: shape={result_mask.shape}, dtype={result_mask.dtype}")
                logger.info(f"{'='*50}")
                
                return (result_image, result_mask)
            
            # Handle mask - create empty mask if none provided
            if mask is None:
                logger.info("No mask provided, creating empty mask")
                mask = create_empty_mask(batch, height, width, device)
            else:
                # Verify mask dimensions
                mask_batch, mask_height, mask_width = verify_mask_dimensions(mask, "Input mask")
                logger.debug(f"Mask value range: min={mask.min():.3f}, max={mask.max():.3f}")
                
                # Check mask dimensions match image
                if mask_height != height or mask_width != width:
                    logger.warning(f"Warning: Mask dimensions ({mask_height}x{mask_width}) don't match image ({height}x{width})")
                    # Resize mask to match image
                    mask = process_mask_downsample(mask, height, width, "nearest", device)
            
            # Calculate target dimensions
            target_height, target_width, scale_factor = calculate_target_dimensions(
                height, width, target_long_edge
            )
            
            # Skip processing if no change needed
            if target_height == height and target_width == width:
                logger.info("No resizing needed - dimensions match target")
                
                # Output reporting
                logger.info(f"\n{'='*50}")
                logger.info("OUTPUT SUMMARY (UNCHANGED):")
                logger.info(f"Image: shape={image.shape}, dtype={image.dtype}")
                logger.info(f"Mask: shape={mask.shape}, dtype={mask.dtype}")
                logger.info(f"{'='*50}")
                
                return (image, mask)
            
            # Process image
            result_image = process_downsample(
                image, target_height, target_width, interpolation, device
            )
            
            # Process mask
            result_mask = process_mask_downsample(
                mask, target_height, target_width, mask_interpolation, device
            )
            
            # Verify output
            logger.info(f"\n{'='*50}")
            logger.info("OUTPUT SUMMARY:")
            logger.info(f"Image: shape={result_image.shape} (BHWC), dtype={result_image.dtype}")
            logger.info(f"Mask: shape={result_mask.shape} (BHW), dtype={result_mask.dtype}")
            
            # CRITICAL: Final verification of output dimensions
            if result_image.shape[1] < MIN_ALLOWED_SIZE or result_image.shape[2] < MIN_ALLOWED_SIZE:
                logger.warning(f"[WARNING] Final output dimensions too small: {result_image.shape}, creating valid tensor")
                valid_image = torch.zeros(batch, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, channels, device=device)
                valid_mask = torch.zeros(batch, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, device=device)
                
                # Revised output reporting
                logger.info(f"REVISED OUTPUT (minimum size):")
                logger.info(f"Image: shape={valid_image.shape}, dtype={valid_image.dtype}")
                logger.info(f"Mask: shape={valid_mask.shape}, dtype={valid_mask.dtype}")
                logger.info(f"{'='*50}")
                
                return (valid_image, valid_mask)
                
            # Ensure values are in valid range  
            result_image = torch.clamp(result_image, 0.0, 1.0)
            result_mask = torch.clamp(result_mask, 0.0, 1.0)
            
            # Final value range reporting
            img_range = f"min={result_image.min().item():.3f}, max={result_image.max().item():.3f}"
            mask_range = f"min={result_mask.min().item():.3f}, max={result_mask.max().item():.3f}"
            logger.debug(f"Image value range: {img_range}")
            logger.debug(f"Mask value range: {mask_range}")
            logger.info(f"{'='*50}")
            
            # Clean up
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            return (result_image, result_mask)
            
        except Exception as e:
            logger.error(f"Error in resolution downsample: {str(e)}")
            logger.debug(f"[DEBUG] Error type: {e.__class__.__name__}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a valid tensor with minimum dimensions
            safe_image = torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE, 3)
            safe_mask = torch.zeros(1, MIN_ALLOWED_SIZE, MIN_ALLOWED_SIZE)
            
            # Error output reporting
            logger.info(f"\n{'='*50}")
            logger.info("ERROR OUTPUT (fallback):")
            logger.info(f"Image: shape={safe_image.shape}, dtype={safe_image.dtype}")
            logger.info(f"Mask: shape={safe_mask.shape}, dtype={safe_mask.dtype}")
            logger.info(f"{'='*50}")
            
            return (safe_image, safe_mask)