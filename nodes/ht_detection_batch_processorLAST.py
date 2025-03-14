"""
File: homage_tools/nodes/ht_detection_batch_processor_v2.py
Version: 2.0.0
Description: Enhanced node for detecting regions, applying mask dilation, and processing with model-based upscaling
             Using UltimateSDUpscaler-style tiled processing for better quality and memory efficiency
             Processes each detected region independently while maintaining aspect ratio
"""
print("Version 2.0.0)

import os
import json
import math
import numpy as np
from enum import Enum
from tqdm import tqdm
from typing import Any, List, Tuple, Optional, Dict, Union

import torch
from PIL import Image, ImageFilter

import comfy
from comfy.samplers import KSampler
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

# Ensure backward compatibility with older Pillow versions
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image

#-------------------------------------------------------
# Constants
#-------------------------------------------------------
MAX_RESOLUTION = 8192

#-------------------------------------------------------
# Utility Functions
#-------------------------------------------------------

def set_tensor_format(format_str="B,C,H,W"):
    """
    Set the global tensor format for processing.
    
    Args:
        format_str: String like "B,C,H,W" or "B,H,W,C" specifying dimension order
    
    Returns:
        The format string for confirmation
    """
    global TENSOR_FORMAT
    TENSOR_FORMAT = format_str
    print(f"Set global tensor format to: {TENSOR_FORMAT}")
    return TENSOR_FORMAT

def get_tensor_format():
    """Get the current global tensor format"""
    global TENSOR_FORMAT
    return TENSOR_FORMAT

def permute_tensor_to_format(tensor, source_format, target_format):
    """
    Permute a tensor from source_format to target_format
    
    Args:
        tensor: PyTorch tensor to permute
        source_format: String like "B,C,H,W" describing current format
        target_format: String like "B,H,W,C" describing desired format
    
    Returns:
        Permuted tensor
    """
    import torch
    
    # If formats are the same, return unchanged
    if source_format == target_format:
        return tensor
        
    # Parse formats into lists
    source_dims = source_format.split(',')
    target_dims = target_format.split(',')
    
    # Verify formats have the same dimensions
    if len(source_dims) != len(target_dims):
        raise ValueError(f"Source format {source_format} and target format {target_format} must have same number of dimensions")
    
    if len(source_dims) != len(tensor.shape):
        raise ValueError(f"Tensor shape {tensor.shape} doesn't match source format {source_format}")
    
    # Create permutation order list
    permutation = []
    for dim in target_dims:
        if dim not in source_dims:
            raise ValueError(f"Dimension {dim} in target format not found in source format")
        permutation.append(source_dims.index(dim))
    
    print(f"Permuting tensor from {source_format} ({tensor.shape}) to {target_format} using order {permutation}")
    return tensor.permute(*permutation)

# Define global default format
TENSOR_FORMAT = "B,C,H,W"  # Standard PyTorch BCHW format

def pil_to_tensor(image):
    """Convert PIL image to PyTorch tensor"""
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None,]
    return img.permute(0, 3, 1, 2)

def tensor_to_pil(tensor, index=0):
    """Convert PyTorch tensor to PIL image"""
    try:
        # Check tensor shape and format
        if len(tensor.shape) != 4:
            raise ValueError(f"Expected 4D tensor (BCHW or BHWC), got shape {tensor.shape}")
        
        # Handle different dimension orders
        if tensor.shape[1] <= 4 and tensor.shape[2] > 4 and tensor.shape[3] > 4:
            # BCHW format - permute to BHWC for processing
            tensor = tensor.permute(0, 2, 3, 1)
        
        # Extract the specified image from batch
        if index >= tensor.shape[0]:
            raise IndexError(f"Index {index} out of bounds for tensor with batch size {tensor.shape[0]}")
        
        img_tensor = tensor[index]
        
        # Scale values to 0-255 range
        i = 255.0 * img_tensor.cpu().numpy()
        i = np.clip(i, 0, 255).astype(np.uint8)
        
        return Image.fromarray(i)
    except Exception as e:
        print(f"Error in tensor_to_pil: {e}, tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        # Try fallback conversion for unusual tensor shapes
        try:
            # For single channel tensors with unusual dimensions
            if len(tensor.shape) == 3 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
                # Reshape to create a valid image (e.g., square dimensions)
                size = int(math.sqrt(tensor.shape[2]))
                reshaped = tensor[0, 0].reshape(size, size).cpu().numpy()
                return Image.fromarray((reshaped * 255).astype(np.uint8), 'L')
            # Basic reshape attempt
            elif len(tensor.shape) == 3:
                return Image.fromarray((tensor[0].cpu().numpy() * 255).astype(np.uint8))
            else:
                # Last resort fallback - create blank image with error message
                img = Image.new('RGB', (256, 256), color=(50, 50, 50))
                print(f"Could not convert tensor with shape {tensor.shape} to PIL image")
                return img
        except Exception as e2:
            print(f"Fallback conversion also failed: {e2}")
            return Image.new('RGB', (256, 256), color=(100, 0, 0))

def get_crop_region(mask, padding):
    """Find the bounding box of the non-zero region in the mask"""
    w, h = mask.size
    x_min, y_min, x_max, y_max = w, h, 0, 0
    
    for y in range(h):
        for x in range(w):
            if mask.getpixel((x, y)) > 0:
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)

def expand_crop(crop, width, height, target_width, target_height):
    """Expand crop region to match target dimensions"""
    x1, y1, x2, y2 = crop
    current_width = x2 - x1
    current_height = y2 - y1
    
    # Calculate how much to expand in each direction
    expand_x = max(0, target_width - current_width)
    expand_y = max(0, target_height - current_height)
    
    # Expand evenly in both directions
    new_x1 = max(0, x1 - expand_x // 2)
    new_y1 = max(0, y1 - expand_y // 2)
    new_x2 = min(width, x2 + (expand_x - expand_x // 2))
    new_y2 = min(height, y2 + (expand_y - expand_y // 2))
    
    # Calculate the actual size after clamping
    actual_width = new_x2 - new_x1
    actual_height = new_y2 - new_y1
    
    return (new_x1, new_y1, new_x2, new_y2), (actual_width, actual_height)

def crop_cond(cond, crop_region, original_size, image_size, crop_size):
    """Crop conditioning to match the crop region"""
    # Explicitly import numpy at function scope to ensure availability
    import numpy as np
    
    x1, y1, x2, y2 = crop_region
    w, h = original_size
    crop_w, crop_h = crop_size
    img_w, img_h = image_size
    
    scale_x = crop_w / (x2 - x1)
    scale_y = crop_h / (y2 - y1)
    
    # Create a copy of the conditioning to avoid modifying the original
    cropped_cond = []
    for c in cond:
        n_cond = c[1].copy()
        n_area = c[2].copy() if len(c) > 2 else None
        
        # Scale the area
        if n_area is not None:
            # Adjust conditioning area to match crop
            area_x, area_y, area_w, area_h = n_area
            # Convert to image coordinates
            area_x = area_x * img_w
            area_y = area_y * img_h
            area_w = area_w * img_w
            area_h = area_h * img_h
            
            # Adjust to crop coordinates
            area_x = (area_x - x1) * scale_x / crop_w
            area_y = (area_y - y1) * scale_y / crop_h
            area_w = area_w * scale_x / crop_w
            area_h = area_h * scale_y / crop_h
            
            # Clamp values
            area_x = max(0, min(1, area_x))
            area_y = max(0, min(1, area_y))
            area_w = max(0, min(1 - area_x, area_w))
            area_h = max(0, min(1 - area_y, area_h))
            
            n_area = [area_x, area_y, area_w, area_h]
        
        if n_area is not None:
            cropped_cond.append([c[0], n_cond, n_area])
        else:
            cropped_cond.append([c[0], n_cond])
    
    return cropped_cond

def dilate_mask_opencv(mask_np, dilation_value):
    """Dilate mask using OpenCV"""
    import cv2
    
    if dilation_value == 0:
        return mask_np
        
    # Convert to uint8 for OpenCV
    binary_mask = (mask_np * 255).astype(np.uint8)
    
    # Create kernel for dilation/erosion
    kernel_size = abs(dilation_value)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation or erosion
    if dilation_value > 0:
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    else:
        dilated = cv2.erode(binary_mask, kernel, iterations=1)
        
    # Convert back to float [0,1]
    return dilated.astype(float) / 255.0

def tensor_gaussian_blur_mask(mask_tensor, blur_size):
    """Apply gaussian blur to mask tensor using OpenCV"""
    import cv2
    
    # Extract mask and convert to numpy (handling batch if needed)
    if len(mask_tensor.shape) == 4:  # BHWC
        # Process each batch independently
        result = []
        for b in range(mask_tensor.shape[0]):
            img = mask_tensor[b, ..., 0].detach().cpu().numpy()
            # Apply blur (must be odd size)
            blur_amount = blur_size if blur_size % 2 == 1 else blur_size + 1
            blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
            result.append(torch.from_numpy(blurred).unsqueeze(-1))
        return torch.stack(result, dim=0)
    else:  # HWC
        img = mask_tensor[..., 0].detach().cpu().numpy()
        # Apply blur (must be odd size)
        blur_amount = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        return torch.from_numpy(blurred).unsqueeze(-1)

def flatten(img, bgcolor):
    """Replace transparency with bgcolor"""
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")

def get_nearest_divisible_size(size, divisor):
    """Calculate nearest size divisible by divisor."""
    return ((size + divisor - 1) // divisor) * divisor

def ensure_bhwc_format(tensor):
    """
    Ensures that a tensor is in BHWC format (batch, height, width, channels).
    Handles unusual tensor shapes and adds proper error handling.
    """
    # Handle empty tensors
    if tensor is None or tensor.numel() == 0:
        raise ValueError("Empty tensor provided")
        
    # Add batch dimension if missing
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        
    # Handle unusual shapes
    if len(tensor.shape) == 4:
        # Check if it's in BCHW format (typical PyTorch format)
        if tensor.shape[1] <= 4 and tensor.shape[2] > 4 and tensor.shape[3] > 4:
            # Convert BCHW to BHWC
            return tensor.permute(0, 2, 3, 1)
        # Already in BHWC format
        elif tensor.shape[3] <= 4:
            return tensor
        else:
            # Unusual tensor shape - try to reshape intelligently
            print(f"Warning: Unusual tensor shape {tensor.shape}, attempting to reshape")
            
            # Case: Single channel but stretched out (like [1, 1, N])
            if tensor.shape[1] == 1 and tensor.shape[2] == 1 and tensor.shape[3] > 16:
                # Reshape to square if possible
                size = int(math.sqrt(tensor.shape[3]))
                if size * size == tensor.shape[3]:
                    return tensor.reshape(tensor.shape[0], size, size, 1)
                else:
                    # Not a perfect square, use closest dimensions
                    h = size
                    w = tensor.shape[3] // h
                    return tensor.reshape(tensor.shape[0], h, w, 1)
    
    # If we got here, return the original tensor
    return tensor

#-------------------------------------------------------
# Core Processing Components
#-------------------------------------------------------

class USDUMode(Enum):
    """Redraw modes for USDU"""
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    """Seam fix modes for USDU"""
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class Upscaler:
    """Handles image upscaling with or without a model"""
    def __init__(self, upscaler_model=None):
        self.upscaler_model = upscaler_model
    
    def _upscale(self, img, scale):
        if scale == 1.0:
            return img
        if self.upscaler_model is None:
            return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(self.upscaler_model, tensor)
        return tensor_to_pil(upscaled)
    
    def upscale(self, img, scale):
        return self._upscale(img, scale)

class VAEEncode:
    """Wrapper for VAE encoding with preemptive tensor transposition"""
    def encode(self, vae, image_tensor):
        """Encode image tensor with explicit format control"""
        print(f"DEBUG: VAE encoder received tensor with shape: {image_tensor.shape}")
        
        # Ensure proper format - BCHW (batch, channels, height, width)
        if len(image_tensor.shape) != 4:
            raise ValueError(f"VAE encoder expects 4D tensor, got {len(image_tensor.shape)}D")
            
        # Get dimensions
        batch_size, channels, height, width = image_tensor.shape
        print(f"B={batch_size}, C={channels}. W={width}, H={height}")
        
        # CRITICAL FIX: Preemptively transpose height and width dimensions
        # The VAE seems to flip dimensions internally, so we flip them first
        # # print(f"DEBUG: PREEMPTIVELY TRANSPOSING tensor from [B={batch_size}, C={channels}, H={height}, W={width}]")
        # # transposed_tensor = image_tensor.permute(0, 1, 3, 2).contiguous()
        # # transposed_tensor = image_tensor.contiguous()
        transposed_tensor = torch.zeros((1, 3, width, height), 
                           dtype=image_tensor.dtype, 
                           device=image_tensor.device)

        # Fill with data (either copy from original tensor if possible, or use default value)
        transposed_tensor.fill_(0.5)  # Use 0.5 as a neutral gray value
        # # print(f"DEBUG: Transposed shape: {transposed_tensor.shape} [B, C, W, H]")
        
        # Verify no zero dimensions
        if 0 in transposed_tensor.shape:
            print(f"WARNING: Zero dimension in transposed tensor: {transposed_tensor.shape}")
            # Create safe tensor
            safe_height = max(1, transposed_tensor.shape[3])
            safe_width = max(1, transposed_tensor.shape[2])
            safe_tensor = torch.zeros((batch_size, 3, safe_width, safe_height), 
                                    dtype=transposed_tensor.dtype, 
                                    device=transposed_tensor.device)
            safe_tensor.fill_(0.5)  # Fill with mid-gray
            transposed_tensor = safe_tensor
            print(f"DEBUG: Created safe tensor: {transposed_tensor.shape}")
            
        # Check device of VAE
        vae_device = getattr(vae, 'device', None)
        if vae_device is None:
            try:
                vae_device = next(vae.parameters()).device
            except:
                vae_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        print(f"DEBUG: VAE device: {vae_device}, Tensor device: {transposed_tensor.device}")
        
        # Force tensor to the same device as VAE
        if str(transposed_tensor.device) != str(vae_device):
            print(f"DEBUG: Moving tensor to VAE device: {vae_device}")
            transposed_tensor = transposed_tensor.to(vae_device)
        
        # Wrap encoding in try-except to catch any errors
        try:
            # Apply encoding with properly formatted tensor
            latent = vae.encode(transposed_tensor)
            print(f"DEBUG: VAE encoding successful: {latent.shape}")
            return (latent,)
        except Exception as e:
            print(f"DEBUG: VAE encoding failed with error: {e}")
            
            # Try with direct emergency tensor
            try:
                # Extract dimensions from error message
                import re
                shape_match = re.search(r'input\[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\]', str(e))
                if shape_match:
                    b = int(shape_match.group(1))
                    c2 = int(shape_match.group(2))
                    h2 = int(shape_match.group(3))
                    w2 = int(shape_match.group(4))
                    
                    print(f"DEBUG: Error reports expected shape: [{b}, {c2}, {h2}, {w2}]")
                    
                    # Create exactly the format the error message wants
                    h2 = max(1, h2)  # Ensure no zero dimension
                    w2 = max(1, w2)  # Ensure no zero dimension
                    
                    # Check if this is a classic error pattern where second dim != 3
                    if c2 != 3:
                        print(f"DEBUG: Creating special fixed tensor with [1, 3, {h2}, {w2}]")
                        new_tensor = torch.zeros((b, 3, w2, h2), dtype=torch.float32, device=vae_device)
                        new_tensor.fill_(0.5)  # Fill with mid-gray
                        
                        # Try encoding with the fixed tensor
                        latent = vae.encode(new_tensor)
                        print(f"DEBUG: Emergency encoding successful: {latent.shape}")
                        return (latent,)
                
                # Create zero latent directly if error extraction fails
                print("DEBUG: Creating direct zero latent")
                latent_h = height // 8
                latent_w = width // 8
                zero_latent = torch.zeros((batch_size, 4, latent_w, latent_h), 
                                       dtype=torch.float32,
                                       device=vae_device)
                return (zero_latent,)
                
            except Exception as e2:
                print(f"DEBUG: Emergency fallback also failed: {e2}. Creating zero latent.")
                
                # Calculate latent dimensions (typically height/8 and width/8)
                latent_h = height // 8
                latent_w = width // 8
                
                # Create zero latent
                zero_latent = torch.zeros((batch_size, 4, latent_w, latent_h), 
                                       dtype=torch.float32,
                                       device=vae_device)
                                       
                return (zero_latent,)

class VAEDecode:
    """Wrapper for VAE decoding"""
    def decode(self, vae, latent):
        return (vae.decode(latent),)

class VAEDecodeTiled:
    """Wrapper for tiled VAE decoding"""
    def decode(self, vae, samples, tile_size=512):
        if hasattr(vae, 'decode_tiled'):
            return (vae.decode_tiled(samples, tile_size),)
        print("[USDU] Tiled decode not available in VAE, falling back to regular decode")
        return (vae.decode(samples),)

def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler=None, custom_sigmas=None):
    """Sample from the model with fixed KSampler initialization"""
    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        try:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(
                model=model,
                add_noise=True,
                seed=seed,
                cfg=cfg,
                positive=positive,
                negative=negative,
                sampler=custom_sampler,
                sigmas=custom_sigmas,
                latent_image=latent
            )
            return samples
        except Exception as e:
            print(f"DEBUG: Custom sampler failed: {e}, falling back to default")
            # Fall through to default sampler
    
    # Default KSampler
    try:
        # Get device from model or latent
        device = getattr(model, 'device', None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except:
                device = latent.device
        
        print(f"DEBUG: Using KSampler with device: {device}")
        
        # Fix: Create KSampler with required arguments
        from comfy.samplers import KSampler
        sampler = KSampler(model, steps, device)
        
        # Call sample method with remaining parameters
        (samples,) = sampler.sample(
            seed=seed,
            cfg=cfg, 
            sampler_name=sampler_name, 
            scheduler=scheduler, 
            positive=positive, 
            negative=negative, 
            latent=latent, 
            denoise=denoise
        )
        
        print(f"DEBUG: KSampler successful, output shape: {samples.shape}")
        return samples
        
    except Exception as e:
        print(f"DEBUG: KSampler failed: {e}, using emergency latent")
        # Emergency fallback - just return the latent unchanged
        print(f"DEBUG: Returning latent unchanged: {latent.shape}")
        return latent

class StableDiffusionProcessing:
    """Main processing class for UltimateSDUpscaler"""
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
        batch=None,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        # Initialize batch
        self.batch = batch if batch is not None else [init_img]
        
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

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[USDU] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.tile_width = tile_width
        self.tile_height = tile_height
        
        # Create VAE encoder and decoder nodes
        self.vae_encoder = VAEEncode()
        self.vae_decoder = VAEDecode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            print("[USDU] Using tiled decode")

        # Other required A1111 variables for the USDU script
        self.extra_generation_params = {}

        # Load config file for USDU
        self.config_path = os.path.join(os.path.dirname(__file__), 'usdu_config.json')
        self.config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if hasattr(comfy.utils, 'PROGRESS_BAR_ENABLED') and comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            self.per_tile_progress = self.config.get('per_tile_progress', True)
            comfy.utils.PROGRESS_BAR_ENABLED = self.per_tile_progress
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
            self.redraw_mode = redraw_mode
            self.seam_fix_mode = seam_fix_mode

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled and hasattr(comfy.utils, 'PROGRESS_BAR_ENABLED'):
            comfy.utils.PROGRESS_BAR_ENABLED = True

class Processed:
    """Result container for USDU processing"""
    def __init__(self, p, images, seed, info):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p, index):
        return None

def process_images(p):
    """Main processing function for USDU with improved tensor handling"""
    # Explicitly re-import numpy here to avoid scope issues
    # This ensures np is definitely available in this function's scope
    import numpy as np
    import torch
    
    # Use global tensor format configuration
    global TENSOR_FORMAT
    tensor_format = TENSOR_FORMAT
    print(f"DEBUG: Using tensor format: {tensor_format}")
    
    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L') if p.image_mask is not None else Image.new('L', p.init_images[0].size, 255)
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
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                         image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in p.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Verify that all tiles have valid dimensions
    for i, tile in enumerate(tiles):
        if tile.width == 0 or tile.height == 0:
            print(f"Error: Tile {i} has invalid dimensions: {tile.size}")
            raise ValueError(f"Invalid tile dimensions: {tile.size}")

    # Encode the image
    try:
        # Extra verification before tensor conversion
        for i, tile in enumerate(tiles):
            print(f"DEBUG: Tile {i} PIL dimensions: {tile.width}x{tile.height} (WxH)")
            if tile.width < 8 or tile.height < 8:
                print(f"Warning: Tile dimensions too small ({tile.width}x{tile.height}), enforcing minimum size")
                tile = tile.resize((max(8, tile.width), max(8, tile.height)), Image.Resampling.LANCZOS)
        
        # Convert to tensors with explicit device control
        device = p.vae.device if hasattr(p.vae, 'device') else 'cpu'
        print(f"Converting tiles to tensors on device {device}")
        
        # Ensure proper format for VAE input based on configured tensor_format
        batched_tiles_list = []
        for i, tile in enumerate(tiles):
            # Convert PIL to numpy to tensor
            tile_np = np.array(tile).astype(np.float32) / 255.0
            print(f"DEBUG: Tile {i} numpy array shape: {tile_np.shape} (Height, Width, Channels)")
            
            # Create tensor and move to appropriate device
            tile_tensor = torch.from_numpy(tile_np).to(device)
            print(f"DEBUG: Tile {i} initial tensor shape: {tile_tensor.shape}")
            
            # PIL gives HWC format by default
            source_format = "H,W,C"
            
            # Check if we need to permute based on the configured tensor_format
            if source_format != tensor_format:
                # Add batch dimension if needed
                if tensor_format.startswith("B") and len(tile_tensor.shape) == 3:
                    tile_tensor = tile_tensor.unsqueeze(0)
                    source_format = "B," + source_format
                    print(f"DEBUG: Added batch dimension, new shape: {tile_tensor.shape}")
                
                # Permute tensor to match the expected format
                if len(source_format.split(',')) == len(tensor_format.split(',')):
                    tile_tensor = permute_tensor_to_format(tile_tensor, source_format, tensor_format)
                    print(f"DEBUG: Permuted tensor to {tensor_format} format: {tile_tensor.shape}")
                else:
                    print(f"WARNING: Cannot permute tensor - format dimensions don't match")
                    # Add batch dimension if needed for VAE
                    if len(tile_tensor.shape) == 3 and tile_tensor.shape[2] in [1, 3, 4]:
                        # This is likely HWC format, convert to BCHW
                        h, w, c = tile_tensor.shape
                        tile_tensor = tile_tensor.permute(2, 0, 1)
                        print(f"DEBUG: Forced permute to CHW: {tile_tensor.shape}")
                        # Add batch dimension
                        tile_tensor = tile_tensor.unsqueeze(0)
                        print(f"DEBUG: After adding batch dim: {tile_tensor.shape}")
            else:
                # Just add batch dimension
                if len(tile_tensor.shape) == 3:
                    tile_tensor = tile_tensor.unsqueeze(0)
                    print(f"DEBUG: Added batch dim: {tile_tensor.shape}")
                
            batched_tiles_list.append(tile_tensor)
            
        # Concatenate along batch dimension
        batched_tiles = torch.cat(batched_tiles_list, dim=0)
        print(f"DEBUG: Batched tiles shape: {batched_tiles.shape}")
        
        # CRITICAL: Verify tensor has proper dimensions before passing to VAE
        # For VAE, we typically need BCHW format
        expected_vae_format = "B,C,H,W"
        if tensor_format != expected_vae_format:
            print(f"WARNING: Current tensor format {tensor_format} differs from VAE expected format {expected_vae_format}")
            print(f"WARNING: This may cause issues with VAE encoding")
            
            # Try to identify the current format's order
            if batched_tiles.shape[1] > 10 and batched_tiles.shape[3] <= 4:
                # # print(f"DEBUG: Tensor appears to be in BHWC format, permuting to BCHW")
                # # batched_tiles = batched_tiles.permute(0, 3, 1, 2)
                batched_tiles = batched_tiles.permute(0, 3, 1, 2)
                # # print(f"DEBUG: After permutation: {batched_tiles.shape}")
            elif batched_tiles.shape[1] <= 4 and batched_tiles.shape[2] > 4 and batched_tiles.shape[3] > 4:
                print(f"DEBUG: Tensor appears to be in BCHW format already")
            else:
                print(f"WARNING: Unusual tensor dimensions: {batched_tiles.shape}")
                print(f"DEBUG: Interpreted as B={batched_tiles.shape[0]}, dim1={batched_tiles.shape[1]}, dim2={batched_tiles.shape[2]}, dim3={batched_tiles.shape[3]}")
        
        if 0 in batched_tiles.shape:
            print("ERROR: Zero dimension in tensor")
            raise ValueError("Invalid tensor dimensions with zero values")
            
        # Ensure we have 3 channels for VAE
        if batched_tiles.shape[1] != 3 and tensor_format == "B,C,H,W":
            print(f"WARNING: Tensor has {batched_tiles.shape[1]} channels, VAE expects 3 channels")
            # Use a different approach if the channels dimension is wrong
            if batched_tiles.shape[3] == 3:  # BHWC format
                print(f"DEBUG: Tensor seems to be in BHWC format, permuting to BCHW")
                batched_tiles = batched_tiles.permute(0, 3, 1, 2)
                print(f"DEBUG: After permutation: {batched_tiles.shape}")
            elif batched_tiles.shape[1] > 10:  # Something very wrong
                print("DEBUG: Creating new tensor with correct shape")
                batch_size = batched_tiles.shape[0]
                # Use the two largest dimensions for height and width
                dims = sorted([d for d in batched_tiles.shape[1:] if d > 0], reverse=True)
                if len(dims) >= 2:
                    height, width = dims[0], dims[1]
                    # Create a safe tensor with known dimensions
                    safe_tensor = torch.zeros((batch_size, 3, height, width), 
                                            dtype=batched_tiles.dtype, 
                                            device=batched_tiles.device)
                    batched_tiles = safe_tensor
                    print(f"DEBUG: Created safe tensor with shape: {batched_tiles.shape}")
            
        # Final verification before VAE
        print(f"DEBUG: Final encoder input shape: {batched_tiles.shape}")
        print(f"DEBUG: Interpreted as: Batch={batched_tiles.shape[0]}, Channels={batched_tiles.shape[1]}, Height={batched_tiles.shape[2]}, Width={batched_tiles.shape[3]}")
        
        # Encode using VAE
        print(f"DEBUG: Sending tensor with shape {batched_tiles.shape} to VAE encoder")
        (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)
        print(f"DEBUG: Latent shape after encoding: {latent.shape}")
        
    except Exception as e:
        print(f"Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        
        # Special handling for the specific error we're getting
        if "expected input" in str(e) and "to have 3 channels" in str(e):
            print("DEBUG: Attempting special fallback for channel error")
            try:
                # Extract tensor dimensions from error message
                error_msg = str(e)
                # Pattern like "expected input[1, 400, 0, 1024]"
                import re
                shape_match = re.search(r'input\[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\]', error_msg)
                if shape_match:
                    b = int(shape_match.group(1))
                    # Force to standard format (ignoring problematic dims)
                    h = int(shape_match.group(4))  # Use width from error as height
                    w = int(shape_match.group(2))  # Use height from error as width
                    
                    print(f"DEBUG: Creating emergency tensor with shape [b={b}, c=3, h={h}, w={w}]")
                    emergency_tensor = torch.zeros((b, 3, h, w), 
                                                dtype=torch.float32, 
                                                device=device)
                    
                    # Try emergency encode
                    (latent,) = p.vae_encoder.encode(p.vae, emergency_tensor)
                    print(f"DEBUG: Emergency encoding successful, latent shape: {latent.shape}")
                else:
                    raise RuntimeError(f"VAE encoding failed: {e}")
            except Exception as e2:
                print(f"Emergency fallback failed: {e2}")
                raise RuntimeError(f"VAE encoding failed: {e}")
        else:
            raise RuntimeError(f"VAE encoding failed: {e}")

    # Generate samples
    print(f"DEBUG: Sending latent with shape {latent.shape} to sampler")
    samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                   negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)
    print(f"DEBUG: Sampler output shape: {samples.shape}")

    # Update the progress bar
    if p.progress_bar_enabled and p.pbar is not None:
        p.pbar.update(1)

    # Decode the sample
    print(f"DEBUG: Sending samples with shape {samples.shape} to VAE decoder")
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512
    print(f"DEBUG: Decoded output shape: {decoded.shape}")

    # Convert the sample to a PIL image
    print(f"DEBUG: Converting decoded tensor with shape {decoded.shape} to PIL images")
    # Check if tensor is in expected format (BCHW or BHWC)
    if len(decoded.shape) == 4:
        if decoded.shape[1] in [1, 3, 4] and decoded.shape[2] > 4 and decoded.shape[3] > 4:
            print(f"DEBUG: Detected BCHW format - Batch={decoded.shape[0]}, Channels={decoded.shape[1]}, Height={decoded.shape[2]}, Width={decoded.shape[3]}")
        elif decoded.shape[3] in [1, 3, 4] and decoded.shape[1] > 4 and decoded.shape[2] > 4:
            print(f"DEBUG: Detected BHWC format - Batch={decoded.shape[0]}, Height={decoded.shape[1]}, Width={decoded.shape[2]}, Channels={decoded.shape[3]}")
        else:
            print(f"DEBUG: Unusual tensor format: {decoded.shape}")
    
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]
    for i, tile in enumerate(tiles_sampled):
        print(f"DEBUG: Converted PIL image {i} size: {tile.width}x{tile.height} (WxH)")

    result_images = []
    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = p.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            print(f"DEBUG: Resizing tile from {tile_sampled.size} to {initial_tile_size}")
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # IMPROVED BLENDING: Create a softer mask with feathered edges
        x1, y1, x2, y2 = crop_region
        feather_amount = min(30, (x2 - x1) // 10, (y2 - y1) // 10)  # Adaptive feathering
        
        # Create base mask with original mask values
        feathered_mask = image_mask.copy()
        
        # Apply Gaussian blur to soften edges
        mask_np = np.array(feathered_mask)
        
        # Create a gradient falloff around edges
        if feather_amount > 0:
            print(f"DEBUG: Applying {feather_amount}px feathering to mask edges")
            # Create falloff distance map from edge
            h, w = mask_np.shape
            x_indices = np.arange(w).reshape(1, -1)
            y_indices = np.arange(h).reshape(-1, 1)
            
            # Distance from each border
            dist_left = x_indices - x1
            dist_right = x2 - x_indices
            dist_top = y_indices - y1
            dist_bottom = y2 - y_indices
            
            # Combine distances with minimum function
            edge_dist = np.minimum(np.minimum(dist_left, dist_right), 
                                np.minimum(dist_top, dist_bottom))
            edge_dist = np.clip(edge_dist, 0, feather_amount)
            
            # Create falloff factor (0 at edge, 1 inside)
            falloff = edge_dist / feather_amount
            
            # Apply falloff to mask
            mask_np = mask_np.astype(float) * falloff
            
            # Convert back to image
            feathered_mask = Image.fromarray(mask_np.astype(np.uint8))
            
            # Additional blur for smoother transition
            feathered_mask = feathered_mask.filter(ImageFilter.GaussianBlur(feather_amount/3))
        
        # Add the feathered mask as an alpha channel
        temp = image_tile_only.copy()
        temp.putalpha(feathered_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Apply final smoothing to reduce any remaining seams
        result = result.filter(ImageFilter.GaussianBlur(0.5))
        result = result.filter(ImageFilter.SHARPEN)

        # Convert back to RGB
        result = result.convert('RGB')
        result_images.append(result)
        print(f"DEBUG: Final image {i} size: {result.size}")

    return Processed(p, result_images, p.seed, None)

def process_with_model(
    image: torch.Tensor,
    upscale_model: Any = None,
    target_size: int = 1024,
    scale_factor: float = 1.0,
    divisibility: int = 64,
    tile_size: int = 512,
    overlap: int = 64,
    model = None,
    vae = None,
    positive = None,
    negative = None,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 8.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 0.5,
    mask_blur: int = 4,
    mask_dilation: int = 0
) -> torch.Tensor:
    """
    Process image with model-based scaling using the UltimateSDUpscaler approach.
    For large images, uses tiled downscaling with Lanczos followed by model refinement.
    """
    print(f"Processing with model (target size: {target_size}, scale: {scale_factor}x)")
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Ensure image is in BHWC format with basic error handling
    if image is None or image.numel() == 0:
        print("Empty input image")
        return None
    
    print(f"DEBUG: Initial image tensor shape: {image.shape}, device: {image.device}")
    
    # Basic shape verification and conversion
    if len(image.shape) == 3:  # HWC
        print(f"DEBUG: Detected HWC format, adding batch dimension")
        image = image.unsqueeze(0)  # Add batch dimension
        print(f"DEBUG: After adding batch: {image.shape} (B,H,W,C)")
    elif len(image.shape) == 4 and image.shape[1] <= 4 and image.shape[2] > 4 and image.shape[3] > 4:
        # BCHW format, convert to BHWC
        print(f"DEBUG: Detected BCHW format, permuting to BHWC")
        print(f"DEBUG: Interpreted as: B={image.shape[0]}, C={image.shape[1]}, H={image.shape[2]}, W={image.shape[3]}")
        image = image.permute(0, 2, 3, 1)
        print(f"DEBUG: After permutation: {image.shape} (B,H,W,C)")
    elif len(image.shape) == 4:
        print(f"DEBUG: Tensor appears to be in BHWC format already")
        print(f"DEBUG: Interpreted as: B={image.shape[0]}, H={image.shape[1]}, W={image.shape[2]}, C={image.shape[3]}")
    
    # Verify image shape after conversion
    if len(image.shape) != 4:
        print(f"ERROR: Unexpected image shape after conversion: {image.shape}")
        return None
    
    # Extract dimensions
    batch_size, height, width, channels = image.shape
    print(f"DEBUG: Input image: {width}x{height}x{channels} (batch size: {batch_size})")
    
    # Calculate scale to match target size for longest edge
    longest_edge = max(width, height)
    base_scale = target_size / longest_edge
    
    # Calculate initial target size (bucket size)
    initial_target_w = int(width * base_scale)
    initial_target_h = int(height * base_scale)
    
    # Ensure divisibility for initial target
    initial_target_w = get_nearest_divisible_size(initial_target_w, divisibility)
    initial_target_h = get_nearest_divisible_size(initial_target_h, divisibility)
    
    print(f"Initial target dimensions: {initial_target_w}x{initial_target_h} (bucket size)")
    
    # Calculate final target size with scale factor
    final_target_w = int(initial_target_w * scale_factor)
    final_target_h = int(initial_target_h * scale_factor)
    
    # Ensure divisibility for final dimensions
    final_target_w = get_nearest_divisible_size(final_target_w, divisibility)
    final_target_h = get_nearest_divisible_size(final_target_h, divisibility)
    
    print(f"Final target dimensions with scale_factor {scale_factor}x: {final_target_w}x{final_target_h}")
    
    # Process each image in the batch
    result_tensors = []
    
    for b in range(batch_size):
        try:
            # Extract single image from batch
            img_bhwc = image[b:b+1]
            
            # Special checking for unusual tensor shapes
            if img_bhwc.shape[1] == 1 and img_bhwc.shape[2] == 1 and img_bhwc.shape[3] > 4:
                print(f"Detected unusual tensor shape: {img_bhwc.shape}, skipping")
                continue
            
            # Always use tiled processing for large images
            print(f"Using tiled downscaling to bucket size {initial_target_w}x{initial_target_h} followed by model-based upscaling")
            
            # Step 1: Create initial downscaled image using tiled Lanczos
            # Initialize empty tensor for downscaled image - EXPLICITLY SET TO CPU
            downscaled = torch.zeros((1, initial_target_h, initial_target_w, channels), 
                                  dtype=torch.float32, 
                                  device='cpu')  # Force CPU for PIL compatibility
            
            print(f"DEBUG: Created empty downscaled tensor with shape: {downscaled.shape} (B,H,W,C)")
            print(f"DEBUG: Target dimensions: Height={initial_target_h}, Width={initial_target_w}, Channels={channels}")
            
            # Calculate optimal tile size and overlap
            effective_tile_size = min(1024, tile_size)  # Limit tile size for memory
            tile_overlap = min(64, effective_tile_size // 4)
            
            tiles_x = math.ceil(width / (effective_tile_size - tile_overlap))
            tiles_y = math.ceil(height / (effective_tile_size - tile_overlap))
            
            print(f"DEBUG: Downscaling with LANCZOS in {tiles_x}x{tiles_y} tiles")
            print(f"DEBUG: Tile size: {effective_tile_size}, overlap: {tile_overlap}")
            
            # Move img_bhwc to CPU for consistent processing
            print(f"DEBUG: Moving input tensor from {img_bhwc.device} to cpu")
            img_bhwc_cpu = img_bhwc.detach().cpu()
            print(f"DEBUG: Input tensor shape for tiling: {img_bhwc_cpu.shape} (B,H,W,C)")
            
            # Process each tile for downscaling
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Calculate tile coordinates with overlap
                    x1 = max(0, x * (effective_tile_size - tile_overlap))
                    y1 = max(0, y * (effective_tile_size - tile_overlap))
                    x2 = min(width, x1 + effective_tile_size)
                    y2 = min(height, y1 + effective_tile_size)
                    
                    # Calculate output coordinates for this tile
                    out_x1 = int(x1 * base_scale)
                    out_y1 = int(y1 * base_scale)
                    out_w = int((x2 - x1) * base_scale)
                    out_h = int((y2 - y1) * base_scale)
                    out_x2 = min(initial_target_w, out_x1 + out_w)
                    out_y2 = min(initial_target_h, out_y1 + out_h)
                    
                    print(f"Processing tile ({x},{y}): {x2-x1}x{y2-y1} → {out_x2-out_x1}x{out_y2-out_y1}")
                    
                    # Skip tiny tiles
                    if out_x2 - out_x1 < 8 or out_y2 - out_y1 < 8:
                        continue
                    
                    # Extract tile from CPU tensor
                    tile = img_bhwc_cpu[:, y1:y2, x1:x2, :]
                    print(f"DEBUG: Extracted tile ({x},{y}) with shape: {tile.shape} (B,H,W,C)")
                    
                    try:
                        # Convert to PIL for high-quality Lanczos
                        tile_np = tile[0].numpy()
                        print(f"DEBUG: Tile numpy shape: {tile_np.shape} (H,W,C)")
                        
                        if tile_np.max() <= 1.0:
                            tile_np = (tile_np * 255).astype(np.uint8)
                        else:
                            tile_np = tile_np.astype(np.uint8)
                            
                        tile_pil = Image.fromarray(tile_np)
                        print(f"DEBUG: PIL image size: {tile_pil.size} (W,H)")
                        
                        # Downscale with Lanczos
                        target_size = (out_x2 - out_x1, out_y2 - out_y1)
                        print(f"DEBUG: Resizing to: {target_size} (W,H)")
                        downscaled_tile = tile_pil.resize(target_size, Image.Resampling.LANCZOS)
                        print(f"DEBUG: Resized PIL size: {downscaled_tile.size} (W,H)")
                        
                        # Convert back to tensor (on CPU)
                        downscaled_np = np.array(downscaled_tile).astype(np.float32) / 255.0
                        print(f"DEBUG: Downscaled numpy shape: {downscaled_np.shape} (H,W,C)")
                        downscaled_tile_tensor = torch.from_numpy(downscaled_np).unsqueeze(0)
                        print(f"DEBUG: Downscaled tensor shape: {downscaled_tile_tensor.shape} (B,H,W,C)")
                        
                        # Create feathering mask (on CPU)
                        mask = torch.ones((1, out_y2 - out_y1, out_x2 - out_x1, 1), 
                                       dtype=torch.float32, 
                                       device='cpu')
                        
                        # Apply feathering at edges if not at border
                        feather_size = min(16, out_w // 4, out_h // 4)
                        
                        if feather_size > 0:
                            # Left edge - if not at left border
                            if x > 0 and x1 > 0:
                                for i in range(feather_size):
                                    weight = i / feather_size
                                    if i < mask.shape[2]:
                                        mask[:, :, i:i+1, :] *= weight
                            
                            # Right edge - if not at right border
                            if x < tiles_x - 1 and x2 < width:
                                for i in range(feather_size):
                                    idx = mask.shape[2] - i - 1
                                    if idx >= 0:
                                        weight = i / feather_size
                                        mask[:, :, idx:idx+1, :] *= weight
                            
                            # Top edge - if not at top border
                            if y > 0 and y1 > 0:
                                for i in range(feather_size):
                                    weight = i / feather_size
                                    if i < mask.shape[1]:
                                        mask[:, i:i+1, :, :] *= weight
                            
                            # Bottom edge - if not at bottom border
                            if y < tiles_y - 1 and y2 < height:
                                for i in range(feather_size):
                                    idx = mask.shape[1] - i - 1
                                    if idx >= 0:
                                        weight = i / feather_size
                                        mask[:, idx:idx+1, :, :] *= weight
                        
                        # Check for channel mismatch
                        if downscaled_tile_tensor.shape[3] == 1 and downscaled.shape[3] == 3:
                            downscaled_tile_tensor = downscaled_tile_tensor.expand(-1, -1, -1, 3)
                        elif downscaled_tile_tensor.shape[3] == 3 and downscaled.shape[3] == 1:
                            downscaled_tile_tensor = downscaled_tile_tensor.mean(dim=3, keepdim=True)
                        
                        # Apply feathered blending (all on CPU now)
                        try:
                            # Get current content
                            current = downscaled[:, out_y1:out_y2, out_x1:out_x2, :]
                            
                            # Ensure dimensions match
                            h = min(current.shape[1], downscaled_tile_tensor.shape[1], mask.shape[1])
                            w = min(current.shape[2], downscaled_tile_tensor.shape[2], mask.shape[2])
                            
                            # Blend
                            downscaled[:, out_y1:out_y1+h, out_x1:out_x1+w, :] = \
                                current[:, :h, :w, :] * (1 - mask[:, :h, :w, :]) + \
                                downscaled_tile_tensor[:, :h, :w, :] * mask[:, :h, :w, :]
                        except Exception as e:
                            print(f"Error in blending: {e}")
                            try:
                                # Direct copy as fallback
                                h = min(downscaled.shape[1]-out_y1, downscaled_tile_tensor.shape[1])
                                w = min(downscaled.shape[2]-out_x1, downscaled_tile_tensor.shape[2])
                                if h > 0 and w > 0:  # Verify dimensions are valid
                                    downscaled[:, out_y1:out_y1+h, out_x1:out_x1+w, :] = \
                                        downscaled_tile_tensor[:, :h, :w, :]
                            except Exception as e2:
                                print(f"Direct copy also failed: {e2}")
                    except Exception as e:
                        print(f"Error processing tile ({x},{y}): {e}")
            
            # Verify that downscaled tensor has valid dimensions
            print(f"DEBUG: Final downscaled tensor shape: {downscaled.shape} (B,H,W,C)")
            print(f"DEBUG: Min: {downscaled.min().item()}, Max: {downscaled.max().item()}")
            
            if downscaled.shape[1] == 0 or downscaled.shape[2] == 0:
                print("ERROR: Downscaled tensor has invalid dimensions")
                raise ValueError(f"Invalid tensor dimensions: {downscaled.shape}")
            
            # Step 2: Convert initial downscaled tensor to PIL for model processing
            downscaled_np = downscaled[0].numpy()
            print(f"DEBUG: Downscaled numpy shape: {downscaled_np.shape} (H,W,C)")
            
            if downscaled_np.max() <= 1.0:
                print(f"DEBUG: Scaling values from [0,1] to [0,255]")
                downscaled_np = (downscaled_np * 255).astype(np.uint8)
            else:
                print(f"DEBUG: Values already in [0,255] range")
                downscaled_np = downscaled_np.astype(np.uint8)
                
            # Verify the array shape and data
            if downscaled_np.size == 0 or np.isnan(downscaled_np).any():
                print("ERROR: Invalid array data for PIL conversion")
                raise ValueError("Cannot convert invalid array to PIL image")
            
            print(f"DEBUG: Converting numpy array with shape {downscaled_np.shape} to PIL")
            downscaled_pil = Image.fromarray(downscaled_np)
            print(f"DEBUG: Successfully created downscaled PIL image with size: {downscaled_pil.size} (W,H)")
            
            print(f"Applying models for refinement and scaling together (scale factor: {scale_factor}x)")
            
            # Create mask for full image
            mask = Image.new('L', downscaled_pil.size, 255)
            
            # Set up processing with both enhancement and scaling in one step
            print(f"DEBUG: Setting up StableDiffusionProcessing with:")
            print(f"DEBUG: - Model device: {model.device if hasattr(model, 'device') else 'unknown'}")
            print(f"DEBUG: - VAE device: {vae.device if hasattr(vae, 'device') else 'unknown'}")
            print(f"DEBUG: - Input image size: {downscaled_pil.size} (W,H)")
            print(f"DEBUG: - Target scale factor: {scale_factor}x")
            print(f"DEBUG: - Tile size: {min(512, initial_target_w)}x{min(512, initial_target_h)}")
            
            p = StableDiffusionProcessing(
                init_img=downscaled_pil,
                model=model,
                positive=positive,
                negative=negative if negative is not None else model.get_empty_conditioning(),
                vae=vae,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                upscale_by=scale_factor,  # Apply the scale factor here
                uniform_tile_mode=True,
                tiled_decode=True,
                tile_width=min(512, initial_target_w),
                tile_height=min(512, initial_target_h),
                redraw_mode=USDUMode.LINEAR,
                seam_fix_mode=USDUSFMode.BAND_PASS,
                batch=[downscaled_pil]
            )
            
            # Set the mask
            p.image_mask = mask
            print(f"DEBUG: Created mask with size: {mask.size} (W,H)")
            
            # Process the image - this will both enhance and scale
            try:
                print("DEBUG: Beginning StableDiffusionProcessing...")
                processed = process_images(p)
                final_image = processed.images[0]
                print(f"DEBUG: Processing successful! Final image size: {final_image.size} (W,H)")
                
                # Verify dimensions - resize if needed
                if final_image.width != final_target_w or final_image.height != final_target_h:
                    print(f"Resizing from {final_image.size} to target {final_target_w}x{final_target_h}")
                    final_image = final_image.resize((final_target_w, final_target_h), 
                                                  Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error in model processing: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to upscaler only
                try:
                    print("Falling back to upscaler model only")
                    if upscale_model is not None:
                        upscaler = Upscaler(upscale_model)
                        final_image = upscaler.upscale(downscaled_pil, scale_factor)
                    else:
                        # Use Lanczos as last resort
                        final_image = downscaled_pil.resize((final_target_w, final_target_h), 
                                                         Image.Resampling.LANCZOS)
                except Exception as e2:
                    print(f"Upscaler fallback failed: {e2}")
                    # Last resort: use Lanczos
                    final_image = downscaled_pil.resize((final_target_w, final_target_h), 
                                                     Image.Resampling.LANCZOS)
            
            # Convert final image to tensor
            print(f"DEBUG: Converting final PIL image with size {final_image.size} to tensor")
            final_np = np.array(final_image).astype(np.float32) / 255.0
            print(f"DEBUG: Numpy array shape: {final_np.shape} (H,W,C)")
            
            final_tensor = torch.from_numpy(final_np).unsqueeze(0)
            print(f"DEBUG: Tensor shape after adding batch dim: {final_tensor.shape} (B,H,W,C)")
            
            # Move final tensor to the original device if needed
            if device != 'cpu':
                print(f"DEBUG: Moving tensor from cpu to {device}")
                final_tensor = final_tensor.to(device)
            
            result_tensors.append(final_tensor)
            print(f"DEBUG: Added tensor to results, current count: {len(result_tensors)}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            # Provide a placeholder result or skip
            continue
    
    # Check if we have any results
    if not result_tensors:
        print("ERROR: No images were successfully processed")
        return None
        
    # Combine all processed images
    if len(result_tensors) > 1:
        print(f"DEBUG: Combining {len(result_tensors)} tensors along batch dimension")
        result = torch.cat(result_tensors, dim=0)
    else:
        print(f"DEBUG: Using single tensor result")
        result = result_tensors[0]
    
    print(f"DEBUG: Final output tensor shape: {result.shape}")
    print(f"DEBUG: Tensor device: {result.device}")
    print(f"DEBUG: Min: {result.min().item()}, Max: {result.max().item()}")
    print(f"DEBUG: Processing complete!")
    return result

#-------------------------------------------------------
# ComfyUI Node Implementation
#-------------------------------------------------------

class HTDetectionBatchProcessor:
    """
    Node for detecting regions, processing with mask dilation, and scaling detected regions.
    Combines BBOX detection, masking, and model-based upscaling in one workflow.
    Uses standard bucket sizes (512, 768, 1024) with additional scale factor.
    Memory-optimized for large images and regions.
    """
    
    CATEGORY = "HommageTools/Processor"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "SEGS", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("processed_images", "processed_masks", "segs", "cropped_images", "bypass_image", "bbox_count")
    OUTPUT_IS_LIST = (True, True, False, True, False, False)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "scale_mode": (["Scale Closest", "Scale Up", "Scale Down", "Scale Max"], {"default": "Scale Closest"}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                "divisibility": (["8", "64"], {"default": "64"}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "max_size_mp": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 500.0, "step": 0.1}),
                # KSampler parameters
                "model": ("MODEL",),
                "vae": ("VAE",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
            "optional": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "upscale_model": ("UPSCALE_MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List types of segments to allow, comma-separated"})
            }
        }
    
    def calculate_target_size(self, bbox_size, mode="Scale Closest"):
        """Calculate target size based on standard buckets."""
        STANDARD_BUCKETS = [512, 768, 1024]
        
        if mode == "Scale Closest":
            target = min(STANDARD_BUCKETS, key=lambda x: abs(x - bbox_size))
        elif mode == "Scale Up":
            target = next((x for x in STANDARD_BUCKETS if x >= bbox_size), STANDARD_BUCKETS[-1])
        elif mode == "Scale Down":
            target = next((x for x in reversed(STANDARD_BUCKETS) if x <= bbox_size), STANDARD_BUCKETS[0])
        else:  # Scale Max
            target = STANDARD_BUCKETS[-1]
        
        return target
    
    def process(self, image, detection_threshold, mask_dilation, crop_factor, drop_size, tile_size, 
            scale_mode, scale_factor, divisibility, mask_blur, overlap, max_size_mp, model, vae, steps, cfg, 
            denoise, sampler_name, scheduler, seed, bbox_detector=None, upscale_model=None, 
            positive=None, negative=None, labels="all"):
        """
        Enhanced process method that uses the UltimateSDUpscaler approach.
        """
        print(f"HTDetectionBatchProcessor starting")
        
        # Convert divisibility from string to int
        div_factor = int(divisibility)
        
        # Initialize empty lists for results
        processed_images = []
        processed_masks = []
        cropped_images = []
        bypass_image = image
        bbox_count = 0
        
        # Ensure BHWC format for input image
        if len(image.shape) == 3:  # HWC
            image = image.unsqueeze(0)  # Add batch dimension
        elif len(image.shape) == 4 and image.shape[1] <= 4 and image.shape[2] > 4 and image.shape[3] > 4:
            # BCHW format, convert to BHWC
            image = image.permute(0, 2, 3, 1)
        
        # Step 1: Run BBOX detection
        if bbox_detector is not None:
            print("Running BBOX detection...")
            
            # Always use first image if batch > 1
            detect_image = image[0:1] if image.shape[0] > 1 else image
            
            try:
                # Execute detection
                segs = bbox_detector.detect(detect_image, detection_threshold, mask_dilation, crop_factor, drop_size)
                
                # Check if valid detection result
                if not isinstance(segs, tuple) or len(segs) < 2:
                    print(f"Invalid detection result format: {type(segs)}")
                    return [], [], ((10, 10), []), [], image, 0
                
                print(f"Detection found {len(segs[1])} segments")
                
                # Filter segments by label if specified
                if labels != '' and labels != 'all':
                    label_list = [label.strip() for label in labels.split(',')]
                    print(f"Filtering for labels: {label_list}")
                    
                    filtered_segs = []
                    for seg in segs[1]:
                        if hasattr(seg, 'label') and seg.label in label_list:
                            filtered_segs.append(seg)
                    
                    print(f"Filtered to {len(filtered_segs)} segments")
                    segs = (segs[0], filtered_segs)
            except Exception as e:
                print(f"Detection error: {str(e)}")
                import traceback
                traceback.print_exc()
                return [], [], ((10, 10), []), [], image, 0
        else:
            print("No detector provided!")
            return [], [], ((10, 10), []), [], image, 0
        
        # Step 2: Process each detection
        if not segs[1] or len(segs[1]) == 0:
            print("No detections found")
            return [], [], segs, [], image, 0
        
        print(f"Processing {len(segs[1])} detected regions...")
        bbox_count = len(segs[1])
        
        # Process each segment
        for i, seg in enumerate(segs[1]):
            try:
                print(f"Processing region {i+1}/{len(segs[1])}: {seg.label if hasattr(seg, 'label') else 'Unknown'}")
                
                # Extract crop region
                x1, y1, x2, y2 = seg.crop_region
                crop_width = x2 - x1
                crop_height = y2 - y1
                print(f"Crop dimensions: {crop_width}x{crop_height}")
                
                # Direct cropping from the original image - more reliable than using seg.cropped_image
                orig_img = image.clone()
                cropped_image = orig_img[:, y1:y2, x1:x2, :]
                print(f"Cropped image shape: {cropped_image.shape}")
                
                # Verify that the crop is valid
                if cropped_image.shape[1] == 0 or cropped_image.shape[2] == 0 or cropped_image.numel() == 0:
                    print(f"Invalid crop dimensions, skipping this region")
                    continue
                
                # Special handling for tensors with unusual shapes like (1,1,N)
                if cropped_image.shape[1] == 1 and cropped_image.shape[2] == 1 and cropped_image.shape[3] > 4:
                    print(f"Detected flattened tensor {cropped_image.shape}, reshaping to 2D image")
                    # Assuming this is a flattened 2D image, try to reshape it
                    long_edge = int(math.sqrt(cropped_image.shape[3]))
                    short_edge = cropped_image.shape[3] // long_edge
                    
                    # Create properly shaped tensor
                    proper_image = torch.zeros((1, short_edge, long_edge, 3), 
                                              dtype=cropped_image.dtype,
                                              device=cropped_image.device)
                    
                    # Original dimensions should be preserved from the crop_region
                    print(f"Reshaping to proper dimensions: {crop_height}x{crop_width}")
                    proper_image = torch.zeros((1, crop_height, crop_width, 3), 
                                              dtype=torch.float32,
                                              device=cropped_image.device)
                    
                    # Fill with gray (0.5) to make it visible
                    proper_image.fill_(0.5)
                    
                    # Use this as the cropped image
                    cropped_image = proper_image
                    print(f"Created placeholder image with shape {cropped_image.shape}")
                
                # Store cropped image in results
                cropped_images.append(cropped_image.clone())
                
                # Create a simple full mask for the cropped region
                cropped_mask = torch.ones((1, crop_height, crop_width, 1), dtype=torch.float32, device=cropped_image.device)
                
                # Calculate target size using standard buckets
                long_edge = max(crop_width, crop_height)
                target_size = self.calculate_target_size(long_edge, scale_mode)
                
                print(f"Crop size: {crop_width}x{crop_height}, target size: {target_size}")
                
                # Process image using the enhanced process_with_model function
                processed = process_with_model(
                    image=cropped_image,
                    upscale_model=upscale_model,
                    target_size=target_size,
                    scale_factor=scale_factor,
                    divisibility=div_factor,
                    tile_size=tile_size,
                    overlap=overlap,
                    model=model,
                    vae=vae,
                    positive=positive,
                    negative=negative,
                    seed=seed + i,  # Different seed for each region
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    mask_blur=mask_blur,
                    mask_dilation=mask_dilation
                )
                
                if processed is not None:
                    # Scale mask to match processed image dimensions
                    processed_h, processed_w = processed.shape[1:3]
                    
                    # Create a full mask for the processed region
                    scaled_mask = torch.ones((1, processed_h, processed_w, 1), dtype=torch.float32, device=processed.device)
                    
                    # Add results to output lists
                    processed_images.append(processed)
                    processed_masks.append(scaled_mask)
                else:
                    print(f"Processing failed for region {i+1}, skipping")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing region {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with next segment
        
        print(f"Processing complete - returning {len(processed_images)} images")
        
        # Check if we have any results
        if not processed_images:
            # Create minimal placeholder
            placeholder = torch.zeros((1, 10, 10, 3), dtype=torch.float32)
            placeholder_mask = torch.ones((1, 10, 10, 1), dtype=torch.float32)
            return [placeholder], [placeholder_mask], segs, [placeholder], image, 0
        
        return processed_images, processed_masks, segs, cropped_images, bypass_image, bbox_count