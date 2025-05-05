"""
File: homage_tools/nodes/ht_moire_removal_node.py
Version: 1.5.0
Description: Node for removing moiré patterns from images using median, Butterworth, and notch filters
with improved filter ordering, parameter selection, consistent cross-tile normalization, and histogram matching
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import math
from scipy import fftpack
import logging
import gc

logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.5.0"

# Reference resolution for parameter scaling
REFERENCE_RESOLUTION = 1024

#------------------------------------------------------------------------------
# Section 1: Memory Management Functions
#------------------------------------------------------------------------------
def estimate_memory_requirement(
    height: int,
    width: int,
    channels: int,
    dtype: torch.dtype
) -> int:
    """
    Estimate memory requirement for FFT processing.
    """
    # Calculate sizes considering FFT operations need more memory
    input_size = height * width * channels
    fft_size = height * width * channels * 2  # Complex numbers need twice the space
    
    # Account for intermediate tensors (3x for FFT operations)
    total_elements = (input_size + fft_size) * 3
    
    # Calculate bytes per element
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    return total_elements * bytes_per_element

def get_optimal_tile_size(
    height: int,
    width: int,
    channels: int,
    dtype: torch.dtype,
    tile_size_mode: str
) -> int:
    """
    Calculate optimal tile size based on available memory and selected mode.
    """
    # Handle fixed tile sizes
    if tile_size_mode in ["1024", "768", "512"]:
        return int(tile_size_mode)
    
    # Memory-based tile size calculation
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # Different memory allocation based on mode
        if tile_size_mode == "Aggressive":
            available_memory = int(total_memory * 0.85)  # Use 85% of available memory
            gc.collect()
            torch.cuda.empty_cache()
        else:  # Conservative
            available_memory = int(total_memory * 0.6)  # Use 60% of available memory
    else:
        available_memory = 4 * (1024 ** 3)  # Assume 4GB for CPU
    
    # Target memory per tile
    if tile_size_mode == "Aggressive":
        target_memory = available_memory // 2  # Use 50% of available memory per tile
        memory_factor = 3  # Tighter margin
    else:  # Conservative
        target_memory = available_memory // 4  # Use 25% of available memory per tile
        memory_factor = 4  # More safety margin
    
    # Calculate base tile size
    bytes_per_pixel = channels * {
        torch.float32: 4,
        torch.float16: 2,
        torch.float64: 8
    }.get(dtype, 4)
    
    pixels_per_tile = target_memory / (bytes_per_pixel * memory_factor)
    tile_size = int(math.sqrt(pixels_per_tile))
    
    # Ensure tile size is a power of 2 for efficient FFT
    tile_size = 2 ** int(math.log2(max(64, tile_size)))
    
    # Cap tile size based on mode
    if tile_size_mode == "Aggressive":
        return min(2048, tile_size)
    else:
        return min(1024, tile_size)

def verify_tensor_format(tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Verify and normalize tensor to BHWC format."""
    if len(tensor.shape) == 3:  # HWC
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    if len(tensor.shape) != 4:
        raise ValueError(f"Invalid tensor shape: {tensor.shape}")
        
    b, h, w, c = tensor.shape
    return tensor, {
        "batch_size": b,
        "height": h,
        "width": w,
        "channels": c
    }

#------------------------------------------------------------------------------
# Section 2: Parameter Scaling Functions
#------------------------------------------------------------------------------
def calculate_resolution_scale(height: int, width: int) -> float:
    """
    Calculate resolution scaling factor based on image dimensions.
    """
    # Use longest edge for scaling calculation
    longest_edge = max(height, width)
    
    # Calculate scale relative to reference resolution
    return longest_edge / REFERENCE_RESOLUTION

def scale_spatial_parameters(
    median_kernel_size: int,
    butterworth_cutoff: int,
    notch_size: int,
    notch_h_spacing: int,
    notch_v_spacing: int,
    resolution_scale: float,
    scaling_factor: float
) -> Tuple[int, int, int, int, int]:
    """
    Scale spatial parameters based on resolution scale.
    """
    # Apply combined scale factor
    combined_scale = resolution_scale * scaling_factor
    
    # Ensure scale is reasonable
    combined_scale = max(0.25, min(4.0, combined_scale))
    
    # Scale parameters, ensuring they stay within their valid ranges
    # Median kernel must be odd
    scaled_kernel = max(3, min(21, int(median_kernel_size * combined_scale)))
    if scaled_kernel % 2 == 0:
        scaled_kernel += 1
    
    # Scale frequency domain parameters
    scaled_cutoff = max(5, min(100, int(butterworth_cutoff * combined_scale)))
    scaled_notch_size = max(5, min(30, int(notch_size * combined_scale)))
    scaled_h_spacing = max(50, min(200, int(notch_h_spacing * combined_scale)))
    scaled_v_spacing = max(50, min(300, int(notch_v_spacing * combined_scale)))
    
    return (
        scaled_kernel,
        scaled_cutoff,
        scaled_notch_size,
        scaled_h_spacing,
        scaled_v_spacing
    )

#------------------------------------------------------------------------------
# Section 3: Filter Implementations
#------------------------------------------------------------------------------
def apply_median_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply median filter to tensor.
    """
    device = image.device
    
    # Process each channel separately
    result = torch.zeros_like(image)
    for b in range(image.shape[0]):
        for c in range(image.shape[3]):
            # Extract channel and move to CPU for processing
            channel = image[b, :, :, c].cpu().numpy()
            
            # Convert to uint8 for OpenCV (normalized to 0-255)
            if channel.max() <= 1.0:
                channel_uint8 = (channel * 255).astype(np.uint8)
            else:
                channel_uint8 = channel.astype(np.uint8)
            
            # Apply median filter
            try:
                import cv2
                filtered = cv2.medianBlur(channel_uint8, kernel_size)
            except ImportError:
                # Fallback to scipy if OpenCV is not available
                from scipy import ndimage
                filtered = ndimage.median_filter(channel_uint8, size=kernel_size)
            
            # Convert back to float and normalize
            if image.max() <= 1.0:
                filtered = filtered.astype(np.float32) / 255.0
            else:
                filtered = filtered.astype(np.float32)
            
            # Place back in result tensor
            result[b, :, :, c] = torch.from_numpy(filtered).to(device)
    
    return result

class ButterworthFilter:
    """Butterworth filter implementation."""
    
    def __init__(self, a: float = 0.75, b: float = 1.25):
        self.a = float(a)
        self.b = float(b)
    
    def __butterworth_filter(self, I_shape, cutoff, order):
        """Create Butterworth filter mask."""
        P = I_shape[0] // 2
        Q = I_shape[1] // 2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2 + (V-Q)**2)).astype(float)
        H = 1 / (1 + (Duv / cutoff**2)**order)
        return (1 - H)
    
    def filter(self, I, cutoff, order):
        """Apply Butterworth filter to image."""
        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float") + 1e-8)
        
        I_fft = np.fft.fft2(I_log)
        
        # Create filter
        H = self.__butterworth_filter(I_shape=I_fft.shape, cutoff=cutoff, order=order)
        
        # Apply filter
        H = np.fft.fftshift(H)
        I_fft_filt = (self.a + self.b * H) * I_fft
        
        # Transform back to spatial domain
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        
        # Handle invalid values
        I[np.isnan(I)] = 0
        I[np.isinf(I)] = 0
        
        return I

def simple_butterworth_filter(image: np.ndarray, cutoff: int, order: int) -> np.ndarray:
    """Simplified Butterworth high-pass filter implementation."""
    # Get image dimensions
    h, w = image.shape
    
    # Create frequency domain coordinates
    u = np.fft.fftfreq(h)[:, np.newaxis]
    v = np.fft.fftfreq(w)
    
    # Calculate distances from center
    d_squared = u**2 + v**2
    
    # Create Butterworth filter
    cutoff_squared = (cutoff / min(h, w))**2
    high_pass = 1 - 1/(1 + (d_squared/cutoff_squared)**order)
    
    # Apply filter
    F = np.fft.fft2(image)
    F_filtered = F * high_pass
    filtered = np.fft.ifft2(F_filtered).real
    
    # Normalize to original range
    min_val, max_val = np.min(image), np.max(image)
    if max_val > min_val:
        filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
        filtered = filtered * (max_val - min_val) + min_val
    
    return filtered

def apply_butterworth_filter(
    image: np.ndarray, 
    a: float, 
    b: float, 
    cutoff: int, 
    order: int
) -> np.ndarray:
    """Apply Butterworth filter with proper value normalization."""
    print(f"Butterworth input stats: shape={image.shape}, min={np.min(image):.4f}, max={np.max(image):.4f}")
    
    # Create filter
    filter = ButterworthFilter(a=a, b=b)
    
    # Store original min/max for normalization
    orig_min = np.min(image)
    orig_max = np.max(image)
    
    try:
        # Apply filter
        filtered = filter.filter(image, cutoff=cutoff, order=order)
        
        print(f"Butterworth raw output stats: min={np.min(filtered):.4f}, max={np.max(filtered):.4f}")
        
        # Handle invalid values if any
        filtered[np.isnan(filtered)] = orig_min
        filtered[np.isinf(filtered)] = orig_min
        
        # Normalize back to original range
        if np.max(filtered) > np.min(filtered):
            filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
            filtered = filtered * (orig_max - orig_min) + orig_min
            
        print(f"Butterworth normalized output stats: min={np.min(filtered):.4f}, max={np.max(filtered):.4f}")
        
        return filtered
    except Exception as e:
        print(f"Error in Butterworth filter: {e}, using backup implementation")
        return simple_butterworth_filter(image, cutoff, order)

def apply_notch_filter(
    image: np.ndarray, 
    h_spacing: int, 
    v_spacing: int, 
    notch_size: int
) -> np.ndarray:
    """
    Apply notch filter to image based on the original notebook implementation.
    """
    # Compute FFT
    F1 = fftpack.fft2(image.astype(float))
    F2 = fftpack.fftshift(F1)
    
    # Get image dimensions
    h, w = image.shape
    
    # Get dimensions for the notch filter
    magnitude_spectrum = 20*np.log10(np.abs(F2) + 0.1)
    
    # Apply notch filter - first pattern
    for i in range(60, h, h_spacing):
        for j in range(100, w, v_spacing):
            if not (i == h//2 and j == w//2):  # Preserve DC component
                F2[max(0, i-notch_size):min(h, i+notch_size), 
                   max(0, j-notch_size):min(w, j+notch_size)] = 0
    
    # Apply notch filter - second pattern (offset)
    for i in range(0, h, h_spacing):
        for j in range(v_spacing//2, w, v_spacing):
            if not (i == h//2 and j == w//2):  # Preserve DC component
                F2[max(0, i-notch_size):min(h, i+notch_size), 
                   max(0, j-notch_size):min(w, j+notch_size)] = 0
    
    # Transform back to spatial domain
    img_filtered = fftpack.ifft2(fftpack.ifftshift(F2)).real
    
    return img_filtered

#------------------------------------------------------------------------------
# Section 4: Processing Implementation
#------------------------------------------------------------------------------
def process_tile(
    tile: torch.Tensor,
    median_enabled: bool,
    median_kernel_size: int,
    butterworth_enabled: bool,
    butterworth_a: float,
    butterworth_b: float,
    butterworth_cutoff: int,
    butterworth_order: int,
    notch_enabled: bool,
    notch_h_spacing: int,
    notch_v_spacing: int,
    notch_size: int,
    device: torch.device,
    global_min_max: Optional[Dict[str, Dict[float, float]]] = None,
    get_stats_only: bool = False
) -> Union[torch.Tensor, Dict[int, Dict[float, float]]]:
    """
    Process a single tile to remove moiré patterns.
    This function can either process the tile or just return stats for global normalization.
    
    Args:
        tile: Input tile tensor
        median_enabled: Whether to apply median filter
        median_kernel_size: Size of median filter kernel
        butterworth_enabled: Whether to apply Butterworth filter
        butterworth_a: Low frequency amplification
        butterworth_b: High frequency amplification
        butterworth_cutoff: Cutoff frequency
        butterworth_order: Filter order
        notch_enabled: Whether to apply notch filter
        notch_h_spacing: Horizontal spacing between notches
        notch_v_spacing: Vertical spacing between notches
        notch_size: Size of each frequency notch
        device: Processing device
        global_min_max: Optional dict with global min/max values for each channel
        get_stats_only: If True, only calculate stats without processing
        
    Returns:
        Either the processed tile tensor or a dict with min/max stats for each channel
    """
    # Move tile to specified device
    tile = tile.to(device)
    
    # Ensure BHWC format
    tile, dims = verify_tensor_format(tile)
    
    # If we only need stats, return them now
    if get_stats_only:
        stats = {}
        for c in range(dims['channels']):
            channel = tile[..., c]
            stats[c] = {
                'min': float(channel.min().item()),
                'max': float(channel.max().item())
            }
        return stats
    
    # Process the input tile
    
    # FIRST: Apply median filter if enabled
    if median_enabled:
        tile = apply_median_filter(tile, median_kernel_size)
    
    # Skip frequency domain processing if neither filter is enabled
    if not (butterworth_enabled or notch_enabled):
        return tile
    
    # Process each channel separately for frequency domain filters
    result = torch.zeros_like(tile)
    
    for b in range(dims['batch_size']):
        for c in range(dims['channels']):
            # Extract channel
            channel = tile[b, :, :, c].cpu().numpy()
            
            # BUTTERWORTH FIRST (if enabled)
            if butterworth_enabled:
                try:
                    # Try original implementation
                    channel_filtered = apply_butterworth_filter(
                        channel, 
                        butterworth_a, 
                        butterworth_b, 
                        butterworth_cutoff, 
                        butterworth_order
                    )
                    
                    # Check if output has valid values
                    if np.isnan(channel_filtered).any() or np.isinf(channel_filtered).any() or np.max(channel_filtered) == np.min(channel_filtered):
                        print("Warning: Primary Butterworth filter produced invalid results, using backup implementation")
                        channel = simple_butterworth_filter(channel, butterworth_cutoff, butterworth_order)
                    else:
                        channel = channel_filtered
                except Exception as e:
                    print(f"Error in Butterworth filter: {e}, using backup implementation")
                    channel = simple_butterworth_filter(channel, butterworth_cutoff, butterworth_order)
                

            
            # NOTCH FILTER SECOND (if enabled)
            if notch_enabled:
                channel = apply_notch_filter(
                    channel, 
                    notch_h_spacing, 
                    notch_v_spacing, 
                    notch_size
                )
                

            
            # Place back in result tensor
            channel_tensor = torch.from_numpy(channel.astype(np.float32)).to(device)
            
            # Apply global normalization if provided
            if global_min_max is not None and c in global_min_max:
                # Use global min/max values for consistent normalization
                gmin, gmax = global_min_max[c]['min'], global_min_max[c]['max']
                if gmin != gmax:  # Avoid division by zero
                    # Normalize using global stats
                    channel_min = channel_tensor.min()
                    channel_max = channel_tensor.max()
                    if channel_min != channel_max:  # Only normalize if there's a range
                        # First normalize to 0-1 range based on this channel's min/max
                        normalized = (channel_tensor - channel_min) / (channel_max - channel_min)
                        # Then scale to global range
                        channel_tensor = normalized * (gmax - gmin) + gmin
            
            print(f"Channel {c} tensor stats: min={channel_tensor.min().item():.4f}, max={channel_tensor.max().item():.4f}")
            result[b, :, :, c] = channel_tensor
    
    # Normalize to 0-1 range if original was in that range
    if tile.max() <= 1.0 and global_min_max is None:
        # Only apply local normalization if no global normalization was provided
        # Preserve original value range
        if result.min() != result.max():
            result = (result - result.min()) / (result.max() - result.min()) 
            result = result * (tile.max() - tile.min()) + tile.min()
    
    # Ensure output has valid range
    if result.max() == 0 and result.min() == 0:
        print("WARNING: Result is all zeros, returning original image")
        return tile
    
    # Final verification
    print(f"Final tile stats: min={result.min().item():.4f}, max={result.max().item():.4f}")
    

    
    return result

#------------------------------------------------------------------------------
# Section 5: Histogram Matching
#------------------------------------------------------------------------------
def match_histograms(source: torch.Tensor, reference: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    Match histograms of source image to reference image with adjustable strength.
    
    Args:
        source: Source image tensor to modify (BHWC format)
        reference: Reference image tensor (BHWC format)
        strength: Blending strength from 0.0 (no change) to 1.0 (full matching)
        
    Returns:
        torch.Tensor: Image with matched histogram
    """
    if strength <= 0.0:
        return source  # No matching needed
    
    # Ensure both tensors have batch dimension and are BHWC format
    if len(source.shape) == 3:
        source = source.unsqueeze(0)
    if len(reference.shape) == 3:
        reference = reference.unsqueeze(0)
        
    device = source.device
    batch_size = source.shape[0]
    channels = source.shape[3]
    result = torch.zeros_like(source)
    
    # Process each batch and channel separately
    for b in range(batch_size):
        for c in range(channels):
            # Get source and reference channels
            src_channel = source[b, :, :, c].flatten()
            ref_channel = reference[b, :, :, c].flatten()
            
            # Skip empty channels
            if torch.allclose(src_channel.max(), src_channel.min()):
                result[b, :, :, c] = source[b, :, :, c]
                continue
                
            # Compute histograms and CDFs
            bins = 256
            src_hist = torch.histc(src_channel, bins=bins, min=0, max=1)
            ref_hist = torch.histc(ref_channel, bins=bins, min=0, max=1)
            
            # Calculate CDFs
            src_cdf = torch.cumsum(src_hist, dim=0)
            ref_cdf = torch.cumsum(ref_hist, dim=0)
            
            # Normalize CDFs
            src_cdf = src_cdf / src_cdf[-1].clamp(min=1e-5)
            ref_cdf = ref_cdf / ref_cdf[-1].clamp(min=1e-5)
            
            # Create lookup table for histogram matching
            lookup = torch.zeros(bins, device=device)
            for i in range(bins):
                # Find the closest value in ref_cdf to the src_cdf value
                src_value = src_cdf[i]
                idx = torch.abs(ref_cdf - src_value).argmin()
                lookup[i] = idx / (bins - 1)
            
            # Apply lookup table to source image
            src_values = (source[b, :, :, c] * (bins - 1)).long().clamp(0, bins - 1)
            matched = lookup[src_values]
            
            # Blend between original and matched based on strength
            if strength < 1.0:
                matched = source[b, :, :, c] * (1 - strength) + matched * strength
            
            result[b, :, :, c] = matched
    
    return result

#------------------------------------------------------------------------------
# Section 6: Overlapping Tile Processing
#------------------------------------------------------------------------------
def create_tile_mask(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    """
    Create a blending mask for overlapping tiles.
    The mask has a smoother falloff in the overlap regions to prevent vignetting.
    
    Args:
        tile_size: Size of the tile
        overlap: Overlap amount
        device: Device to create the mask on
        
    Returns:
        torch.Tensor: A 2D mask with smooth blending in overlap areas
    """
    # Create base mask of ones
    mask = torch.ones((tile_size, tile_size), device=device)
    
    # Create falloff in overlap areas
    if overlap > 0:
        # Use cosine falloff for smoother transitions
        # Left edge falloff
        for x in range(overlap):
            # Cosine falloff: 0.5 * (1 - cos(π * x / overlap))
            mask[:, x] = 0.5 * (1 - math.cos(math.pi * x / overlap))
            
        # Right edge falloff
        for x in range(tile_size - overlap, tile_size):
            # Cosine falloff: 0.5 * (1 + cos(π * (x - (tile_size - overlap)) / overlap))
            mask[:, x] = 0.5 * (1 + math.cos(math.pi * (x - (tile_size - overlap)) / overlap))
            
        # Top edge falloff
        for y in range(overlap):
            # Cosine falloff: 0.5 * (1 - cos(π * y / overlap))
            mask[y, :] *= 0.5 * (1 - math.cos(math.pi * y / overlap))
            
        # Bottom edge falloff
        for y in range(tile_size - overlap, tile_size):
            # Cosine falloff: 0.5 * (1 + cos(π * (y - (tile_size - overlap)) / overlap))
            mask[y, :] *= 0.5 * (1 + math.cos(math.pi * (y - (tile_size - overlap)) / overlap))
    
    return mask

#------------------------------------------------------------------------------
# Section 6: Node Class Definition
#------------------------------------------------------------------------------
class HTMoireRemovalNode:
    """
    Node for removing moiré patterns using median, Butterworth, and notch filters.
    Improved with consistent cross-tile normalization and optional tile overlap.
    """
    
    CATEGORY = "HommageTools/Filters"
    FUNCTION = "remove_moire"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    
    TILE_SIZE_MODES = ["1024", "768", "512", "Conservative", "Aggressive"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image": ("IMAGE",),
                
                # Resolution scaling factor
                "resolution_scaling_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "Scale for resolution-dependent parameters (1.0 = 1024px reference)"
                }),
                
                # Tile size and consistency control
                "tile_size_mode": (cls.TILE_SIZE_MODES, {
                    "default": "1024",
                    "description": "Tile size mode for memory management"
                }),
                "global_normalization": ("BOOLEAN", {
                    "default": True,
                    "description": "Apply consistent normalization across all tiles"
                }),
                "tile_overlap": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 256,
                    "step": 16,
                    "description": "Overlap between adjacent tiles in pixels"
                }),
                "edge_padding": ("INT", {
                    "default": 64,
                    "min": 0, 
                    "max": 256,
                    "step": 16,
                    "description": "Padding around image edges to prevent vignetting"
                }),
                
                # Median filter options (APPLIED FIRST)
                "median_enabled": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable median filtering"
                }),
                "median_kernel_size": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                    "description": "Base size of median filter kernel (auto-scaled with resolution)"
                }),
                
                # Butterworth filter options (APPLIED SECOND)
                "butterworth_enabled": ("BOOLEAN", {
                    "default": True,
                    "description": "Enable Butterworth filtering"
                }),
                "butterworth_a": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "description": "Low frequency amplification"
                }),
                "butterworth_b": ("FLOAT", {
                    "default": 1.25,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.05,
                    "description": "High frequency amplification"
                }),
                "butterworth_cutoff": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "description": "Base cutoff frequency (auto-scaled with resolution)"
                }),
                "butterworth_order": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "description": "Filter order (steepness)"
                }),
                
                # Notch filter options (APPLIED THIRD)
                "notch_enabled": ("BOOLEAN", {
                    "default": False,
                    "description": "Enable notch filtering"
                }),
                "notch_h_spacing": ("INT", {
                    "default": 135,
                    "min": 50,
                    "max": 200,
                    "step": 5,
                    "description": "Base horizontal spacing between notches (auto-scaled with resolution)"
                }),
                "notch_v_spacing": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 300,
                    "step": 5,
                    "description": "Base vertical spacing between notches (auto-scaled with resolution)"
                }),
                "notch_size": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 30,
                    "step": 1,
                    "description": "Base size of each frequency notch (auto-scaled with resolution)"
                })
            }
        }

    def remove_moire(
        self,
        image: torch.Tensor,
        resolution_scaling_factor: float,
        tile_size_mode: str,
        global_normalization: bool,
        tile_overlap: int,
        edge_padding: int,
        median_enabled: bool,
        median_kernel_size: int,
        butterworth_enabled: bool,
        butterworth_a: float,
        butterworth_b: float,
        butterworth_cutoff: int,
        butterworth_order: int,
        notch_enabled: bool,
        notch_h_spacing: int,
        notch_v_spacing: int,
        notch_size: int
    ) -> Tuple[torch.Tensor]:
        """
        Remove moiré patterns from image with improved filter order and consistent normalization.
        """
        logger.info(f"HTMoireRemovalNode v{VERSION} - Processing")
        
        try:
            # Process the input image
            
            # Ensure BHWC format
            image, dims = verify_tensor_format(image)
            
            # Simple case - if no filters are enabled, return original
            if not any([median_enabled, butterworth_enabled, notch_enabled]):
                return (image,)
            
            # Set processing device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Processing device: {device}")
            
            # Calculate resolution scale
            resolution_scale = calculate_resolution_scale(dims['height'], dims['width'])
            logger.info(f"Image resolution scale: {resolution_scale:.2f} (relative to {REFERENCE_RESOLUTION}px)")
            
            # Scale spatial parameters based on resolution
            (scaled_kernel_size, 
             scaled_cutoff, 
             scaled_notch_size, 
             scaled_h_spacing, 
             scaled_v_spacing) = scale_spatial_parameters(
                median_kernel_size,
                butterworth_cutoff,
                notch_size,
                notch_h_spacing,
                notch_v_spacing,
                resolution_scale,
                resolution_scaling_factor
            )
            
            logger.info(f"Scaled parameters: kernel={scaled_kernel_size}, cutoff={scaled_cutoff}, "
                       f"notch_size={scaled_notch_size}, spacing={scaled_h_spacing}x{scaled_v_spacing}")
            
            # Prepare for aggressive memory management if needed
            if tile_size_mode == "Aggressive" and device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
            
            # Memory usage estimation and tiling
            memory_req = estimate_memory_requirement(
                dims['height'], 
                dims['width'], 
                dims['channels'], 
                image.dtype
            )
            logger.info(f"Estimated memory requirement: {memory_req / (1024**3):.2f} GB")
            
            # Calculate optimal tile size
            if dims['height'] * dims['width'] > 512 * 512:  # Only tile larger images
                tile_size = get_optimal_tile_size(
                    dims['height'], 
                    dims['width'], 
                    dims['channels'], 
                    image.dtype,
                    tile_size_mode
                )
                logger.info(f"Using tile size: {tile_size}x{tile_size} (Mode: {tile_size_mode})")
                
                # Apply edge padding to avoid vignetting at boundaries
                if edge_padding > 0:
                    print(f"Adding {edge_padding}px edge padding to prevent vignetting")
                    
                    # Create padded image
                    padded_image = torch.zeros(
                        (dims['batch_size'], 
                         dims['height'] + 2 * edge_padding, 
                         dims['width'] + 2 * edge_padding, 
                         dims['channels']), 
                        dtype=image.dtype, 
                        device=image.device
                    )
                    
                    # Fill center with original image
                    padded_image[:, 
                                 edge_padding:edge_padding + dims['height'], 
                                 edge_padding:edge_padding + dims['width'], 
                                 :] = image
                    
                    # Fill padding - extend edge pixels
                    # Top edge
                    padded_image[:, :edge_padding, edge_padding:edge_padding + dims['width'], :] = image[:, 0:1, :, :].expand(-1, edge_padding, -1, -1)
                    # Bottom edge
                    padded_image[:, edge_padding + dims['height']:, edge_padding:edge_padding + dims['width'], :] = image[:, -1:, :, :].expand(-1, edge_padding, -1, -1)
                    # Left edge (including corners)
                    padded_image[:, :, :edge_padding, :] = padded_image[:, :, edge_padding:edge_padding+1, :].expand(-1, -1, edge_padding, -1)
                    # Right edge (including corners)
                    padded_image[:, :, edge_padding + dims['width']:, :] = padded_image[:, :, edge_padding + dims['width'] - 1:edge_padding + dims['width'], :].expand(-1, -1, edge_padding, -1)
                    
                    # Update dimensions
                    padded_dims = {
                        'batch_size': dims['batch_size'],
                        'height': dims['height'] + 2 * edge_padding,
                        'width': dims['width'] + 2 * edge_padding,
                        'channels': dims['channels']
                    }
                    
                    # Set the padded image as our working image
                    working_image = padded_image
                    working_dims = padded_dims
                    

                else:
                    working_image = image
                    working_dims = dims
                
                # Step 1: Calculate global min/max for consistent normalization if enabled
                global_min_max = None
                if global_normalization:
                    print("Calculating global statistics for consistent normalization...")
                    # First pass: collect statistics without processing
                    channel_stats = {c: {'min': float('inf'), 'max': float('-inf')} 
                                    for c in range(working_dims['channels'])}
                    
                    # Process non-overlapping tiles first for stats
                    for y in range(0, working_dims['height'], tile_size):
                        for x in range(0, working_dims['width'], tile_size):
                            # Calculate tile bounds
                            y_end = min(y + tile_size, working_dims['height'])
                            x_end = min(x + tile_size, working_dims['width'])
                            
                            # Extract tile
                            tile = working_image[:, y:y_end, x:x_end, :]
                            
                            # Get stats only for this tile
                            tile_stats = process_tile(
                                tile=tile,
                                median_enabled=median_enabled,
                                median_kernel_size=scaled_kernel_size,
                                butterworth_enabled=butterworth_enabled,
                                butterworth_a=butterworth_a,
                                butterworth_b=butterworth_b,
                                butterworth_cutoff=scaled_cutoff,
                                butterworth_order=butterworth_order,
                                notch_enabled=notch_enabled,
                                notch_h_spacing=scaled_h_spacing,
                                notch_v_spacing=scaled_v_spacing,
                                notch_size=scaled_notch_size,
                                device=device,
                                get_stats_only=True
                            )
                            
                            # Update global stats
                            for c in range(working_dims['channels']):
                                if c in tile_stats:
                                    channel_stats[c]['min'] = min(channel_stats[c]['min'], tile_stats[c]['min'])
                                    channel_stats[c]['max'] = max(channel_stats[c]['max'], tile_stats[c]['max'])
                    
                    global_min_max = channel_stats
                    print(f"Global statistics: {global_min_max}")
                
                # Step 2: Create accumulation buffer with all zeros
                accumulator = torch.zeros_like(working_image)
                weight_map = torch.zeros((working_dims['batch_size'], working_dims['height'], working_dims['width'], 1), device=device)
                
                # Create blend mask for overlapping tiles
                blend_mask = None
                if tile_overlap > 0:
                    blend_mask = create_tile_mask(tile_size, tile_overlap, device)
                    print(f"Created blending mask for tile overlap of {tile_overlap} pixels")
                
                # Step 3: Process tiles with overlap
                effective_step = max(1, tile_size - tile_overlap)
                for y in range(0, working_dims['height'], effective_step):
                    y_end = min(y + tile_size, working_dims['height'])
                    y_start = max(0, y_end - tile_size)  # Work backwards to ensure we cover the full image
                    
                    for x in range(0, working_dims['width'], effective_step):
                        x_end = min(x + tile_size, working_dims['width'])
                        x_start = max(0, x_end - tile_size)  # Work backwards to ensure we cover the full image
                        
                        # Extract tile
                        tile = working_image[:, y_start:y_end, x_start:x_end, :]
                        
                        # Create local blend mask if tile is smaller than full size
                        local_blend = None
                        if blend_mask is not None:
                            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                                # Create custom size mask for edge tiles
                                local_blend = create_tile_mask(
                                    max(tile.shape[1], tile.shape[2]), 
                                    tile_overlap, 
                                    device
                                )
                                # Crop to actual tile size
                                local_blend = local_blend[:tile.shape[1], :tile.shape[2]]
                            else:
                                local_blend = blend_mask
                            
                            # Expand dimensions to match tile
                            local_blend = local_blend.view(1, local_blend.shape[0], local_blend.shape[1], 1)
                            local_blend = local_blend.expand(tile.shape[0], -1, -1, tile.shape[3])
                        
                        # Process tile
                        processed_tile = process_tile(
                            tile=tile,
                            median_enabled=median_enabled,
                            median_kernel_size=scaled_kernel_size,
                            butterworth_enabled=butterworth_enabled,
                            butterworth_a=butterworth_a,
                            butterworth_b=butterworth_b,
                            butterworth_cutoff=scaled_cutoff,
                            butterworth_order=butterworth_order,
                            notch_enabled=notch_enabled,
                            notch_h_spacing=scaled_h_spacing,
                            notch_v_spacing=scaled_v_spacing,
                            notch_size=scaled_notch_size,
                            device=device,
                            global_min_max=global_min_max
                        )
                        
                        # Apply blending mask
                        if local_blend is not None:
                            processed_tile = processed_tile * local_blend
                            
                            # Accumulate the weight mask
                            weight_map[:, y_start:y_end, x_start:x_end, :] += local_blend[:, :, :, :1]
                        else:
                            # Without blending, just add 1 to the weight map for this tile
                            weight_map[:, y_start:y_end, x_start:x_end, :] += 1
                        
                        # Accumulate the processed tile
                        accumulator[:, y_start:y_end, x_start:x_end, :] += processed_tile
                        
                        # Clear GPU cache after each tile if in aggressive mode
                        if tile_size_mode == "Aggressive" and device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                # Step 4: Normalize by accumulated weights
                epsilon = 1e-8  # Small value to prevent division by zero
                result = accumulator / (weight_map + epsilon)
                
                # If we used edge padding, extract the central part
                if edge_padding > 0:
                    print(f"Removing edge padding from result")
                    result = result[:, 
                                   edge_padding:edge_padding + dims['height'], 
                                   edge_padding:edge_padding + dims['width'], 
                                   :]
                

                
                return (result,)
            else:
                # Process entire image at once for small images
                result = process_tile(
                    tile=image,
                    median_enabled=median_enabled,
                    median_kernel_size=scaled_kernel_size,
                    butterworth_enabled=butterworth_enabled,
                    butterworth_a=butterworth_a,
                    butterworth_b=butterworth_b,
                    butterworth_cutoff=scaled_cutoff,
                    butterworth_order=butterworth_order,
                    notch_enabled=notch_enabled,
                    notch_h_spacing=scaled_h_spacing,
                    notch_v_spacing=scaled_v_spacing,
                    notch_size=scaled_notch_size,
                    device=device
                )
                

                
                return (result,)
                
        except Exception as e:
            logger.error(f"Error in moire removal: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return (image,)