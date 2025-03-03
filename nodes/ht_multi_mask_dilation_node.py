"""
File: homage_tools/nodes/ht_multi_mask_dilation_node.py
Version: 1.0.0
Description: Node for detecting and cropping multiple regions from a mask with BHWC tensor handling

Sections:
1. Imports and Type Definitions
2. Constants and Configuration
3. Region Detection Functions
4. Bounding Box Processing
5. Node Class Definition
6. Main Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.0.0"
STANDARD_BUCKETS = [512, 768, 1024]

#------------------------------------------------------------------------------
# Section 3: Region Detection Functions
#------------------------------------------------------------------------------
def find_connected_components(mask: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """
    Find connected components in a mask using GPU if available.
    
    Args:
        mask: Input mask tensor (BHWC format)
        connectivity: Connectivity type (4 or 8)
        
    Returns:
        torch.Tensor: Labeled components tensor
    """
    # Extract first batch and channel for processing
    mask_2d = mask[0, ..., 0].cpu().numpy()
    
    # Use scipy for connected component labeling
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(mask_2d > 0, 
                                              structure=ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2))
    
    print(f"Found {num_features} connected components")
    
    # Convert back to tensor
    labeled_tensor = torch.from_numpy(labeled_array).to(mask.device)
    
    return labeled_tensor, num_features

def get_component_bounds(labeled_components: torch.Tensor, component_idx: int) -> Tuple[int, int, int, int]:
    """
    Get the bounding box for a specific component.
    
    Args:
        labeled_components: Tensor with labeled components
        component_idx: Component index to find bounds for
        
    Returns:
        Tuple[int, int, int, int]: (min_y, max_y, min_x, max_x)
    """
    # Find pixels belonging to the component
    indices = torch.nonzero(labeled_components == component_idx)
    
    if len(indices) == 0:
        print(f"Warning: Empty component {component_idx}")
        return 0, 0, 0, 0
        
    # Get bounds
    min_y = indices[:, 0].min().item()
    max_y = indices[:, 0].max().item() + 1
    min_x = indices[:, 1].min().item()
    max_x = indices[:, 1].max().item() + 1
    
    print(f"Component {component_idx} bounds: Y: {min_y} to {max_y}, X: {min_x} to {max_x}")
    return min_y, max_y, min_x, max_x

#------------------------------------------------------------------------------
# Section 4: Bounding Box Processing
#------------------------------------------------------------------------------
def calculate_target_size(bbox_size: int, scale_mode: str) -> int:
    """
    Calculate target size based on scale mode.
    
    Args:
        bbox_size: Original bounding box size
        scale_mode: Scaling mode selection
        
    Returns:
        int: Target size
    """
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
    
    return target

def apply_padding_to_bounds(
    min_y: int, 
    max_y: int, 
    min_x: int, 
    max_x: int, 
    padding: int, 
    height: int, 
    width: int
) -> Tuple[int, int, int, int]:
    """
    Apply padding to bounding box coordinates with bounds checking.
    
    Args:
        min_y, max_y, min_x, max_x: Original bounds
        padding: Padding to add
        height, width: Image dimensions
        
    Returns:
        Tuple[int, int, int, int]: Padded bounds
    """
    min_y = max(0, min_y - padding)
    max_y = min(height, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(width, max_x + padding)
    
    return min_y, max_y, min_x, max_x

#------------------------------------------------------------------------------
# Section 5: Node Class Definition
#------------------------------------------------------------------------------
class HTMultiMaskDilationNode:
    """
    Node for detecting and processing multiple regions in a mask.
    Outputs batches of cropped masks and images along with metadata.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_multi_mask"
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "FLOAT", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("cropped_masks", "cropped_images", "widths", "heights", "scale_factors", "is_multi_region", "region_count", "region_indices")
    
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
                }),
                "connectivity": (["4-connected", "8-connected"], {
                    "default": "8-connected"
                }),
                "max_regions": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                })
            }
        }

#------------------------------------------------------------------------------
# Section 6: Main Processing Logic
#------------------------------------------------------------------------------
    def process_multi_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        scale_mode: str,
        padding: int,
        connectivity: str,
        max_regions: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, int, torch.Tensor]:
        """
        Process mask to detect and crop multiple regions.
        
        Args:
            image: Input image tensor (BHWC format)
            mask: Input mask tensor (BHWC format)
            scale_mode: Scaling mode for output size
            padding: Padding to add around regions
            connectivity: Connectivity type for region detection
            max_regions: Maximum number of regions to process
            
        Returns:
            Tuple of processed tensors and metadata
        """
        print(f"\nHTMultiMaskDilationNode v{VERSION} - Processing")
        
        try:
            # Ensure BHWC format
            if len(image.shape) == 3:  # HWC format
                image = image.unsqueeze(0)  # Add batch -> BHWC
                
            if len(mask.shape) == 3:  # HWC format
                mask = mask.unsqueeze(0)  # Add batch -> BHWC
            
            # Print detailed tensor info
            print(f"Image: shape={image.shape}, dtype={image.dtype}")
            print(f"Mask: shape={mask.shape}, dtype={mask.dtype}")
            
            # Handle mask with multiple channels
            if mask.shape[-1] > 1:
                print("Converting multi-channel mask to single channel")
                mask = mask.mean(dim=-1, keepdim=True)
            
            # Get image dimensions
            batch, height, width, channels = image.shape
            
            # Detect connected components
            conn_value = 8 if connectivity == "8-connected" else 4
            labeled_components, num_regions = find_connected_components(mask, conn_value)
            
            # Limit number of regions
            num_regions = min(num_regions, max_regions)
            
            if num_regions == 0:
                print("Warning: No regions detected in mask")
                # Return original image/mask with default values
                return (
                    mask,
                    image,
                    torch.tensor(width, dtype=torch.int32),
                    torch.tensor(height, dtype=torch.int32),
                    torch.tensor(1.0, dtype=torch.float32),
                    False,
                    0,
                    torch.tensor([0], dtype=torch.int32)  # Default region index
                )
            
            # Initialize result tensors
            cropped_masks = []
            cropped_images = []
            widths = []
            heights = []
            scale_factors = []
            region_indices = []  # Track region indices
            
            # Process each region
            for i in range(1, num_regions + 1):  # Component indices start from 1
                # Get bounds for this component
                min_y, max_y, min_x, max_x = get_component_bounds(labeled_components, i)
                
                # Apply padding
                min_y, max_y, min_x, max_x = apply_padding_to_bounds(
                    min_y, max_y, min_x, max_x, padding, height, width
                )
                
                # Calculate bounding box dimensions
                bbox_width = max_x - min_x
                bbox_height = max_y - min_y
                long_edge = max(bbox_width, bbox_height)
                
                # Skip if box is too small
                if bbox_width <= 1 or bbox_height <= 1:
                    print(f"Skipping region {i}: too small ({bbox_width}x{bbox_height})")
                    continue
                
                # Calculate target size and scale factor
                target_size = calculate_target_size(long_edge, scale_mode)
                scale_factor = target_size / long_edge
                
                # Create component mask (isolate this component)
                component_mask = torch.zeros_like(mask)
                component_indices = (labeled_components == i).unsqueeze(0).unsqueeze(-1)
                component_mask = component_mask.masked_fill(component_indices, 1.0)
                
                # Crop image and mask to the mask content bounds
                cropped_component_mask = component_mask[:, min_y:max_y, min_x:max_x, :]
                cropped_component_image = image[:, min_y:max_y, min_x:max_x, :]
                
                # Add to result lists
                cropped_masks.append(cropped_component_mask)
                cropped_images.append(cropped_component_image)
                widths.append(bbox_width)
                heights.append(bbox_height)
                scale_factors.append(scale_factor)
                region_indices.append(i)  # Store the region index
            
            # Check if we found any valid regions
            if not cropped_masks:
                print("Warning: No valid regions after filtering")
                return (
                    mask,
                    image,
                    torch.tensor(width, dtype=torch.int32),
                    torch.tensor(height, dtype=torch.int32),
                    torch.tensor(1.0, dtype=torch.float32),
                    False,
                    0,
                    torch.tensor([0], dtype=torch.int32)  # Default region index
                )
            
            # Stack results into batch tensors
            batched_masks = torch.cat(cropped_masks, dim=0)
            batched_images = torch.cat(cropped_images, dim=0)
            
            # Convert metadata to tensors for ComfyUI
            widths_tensor = torch.tensor(widths, dtype=torch.int32)
            heights_tensor = torch.tensor(heights, dtype=torch.int32)
            scales_tensor = torch.tensor(scale_factors, dtype=torch.float32)
            indices_tensor = torch.tensor(region_indices, dtype=torch.int32)  # Region indices as tensor
            
            # Set multi-region flag and count
            is_multi_region = len(cropped_masks) > 1
            region_count = len(cropped_masks)
            
            print(f"Processed {region_count} regions from mask")
            return (
                batched_masks,
                batched_images,
                widths_tensor,
                heights_tensor,
                scales_tensor,
                is_multi_region,
                region_count,
                indices_tensor
            )
            
        except Exception as e:
            logger.error(f"Error in multi-mask dilation: {str(e)}")
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return (
                mask,
                image,
                torch.tensor(width, dtype=torch.int32),
                torch.tensor(height, dtype=torch.int32),
                torch.tensor(1.0, dtype=torch.float32),
                False,
                0,
                torch.tensor([0], dtype=torch.int32)  # Default region index
            )