"""
Region upscaling module for HTDetectionBatchProcessor
"""
import torch
import math
import copy
import numpy as np
import time
from PIL import Image, ImageFilter
from enum import Enum

from ..modules.debug_utils import save_debug_image

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

# Conversion utilities
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

# Conditioning functions
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

def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    """
    Crop ControlNet conditioning to match the current region.
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

def crop_tensor(tensor, region):
    """Crop a tensor to a specified region."""
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]

def resize_tensor(tensor, size, mode="nearest-exact"):
    """Resize a tensor to specified dimensions."""
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)

def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    Crop mask conditioning to match the current region.
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
            padded_image = padded_image.filter(ImageFilter.GaussianBlur(15))
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

def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    """
    Crop all conditioning information to match the current region.
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
        
        # Tile parameters
        self.tile_width = tile_width
        self.tile_height = tile_height

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[UPSCALER] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

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
            print("[UPSCALER] Using tiled decode")

        # Other required A1111 variables for the USDU script
        self.extra_generation_params = {}

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        import comfy
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = True
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

# Process class to hold results
class Processed:
    """Class to hold processed results."""
    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        """Get info text for a processed image."""
        return None

def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """Generate samples using the specified sampler and parameters."""
    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        custom_sample = SamplerCustom()
        result = getattr(custom_sample, custom_sample.FUNCTION)(
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
        # Check if result is a tuple (samples, ...)
        if isinstance(result, tuple) and len(result) > 0:
            return result[0]
        return result

    # Default sampling using common_ksampler
    from nodes import common_ksampler
    result = common_ksampler(model, seed, steps, cfg, sampler_name,
                         scheduler, positive, negative, latent, denoise=denoise)
    
    # Handle different return types
    # Case 1: Tuple of (samples, ...)
    if isinstance(result, tuple):
        if len(result) > 0:
            first_item = result[0]
            # Check if the first item is a dict with 'samples'
            if isinstance(first_item, dict) and 'samples' in first_item:
                return first_item['samples']
            return first_item
        return latent
        
    # Case 2: Dictionary with 'samples' key
    elif isinstance(result, dict) and 'samples' in result:
        return result['samples']
        
    # Case 3: Direct tensor return
    return result
        
def process_images(p: StableDiffusionProcessing) -> Processed:
    """Main image generation function."""
    from tqdm import tqdm
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    if p.image_mask is None:
        p.image_mask = Image.new('L', (p.init_images[0].width, p.init_images[0].height), 255)
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Get crop region
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    # Process crop region and tile size
    if p.uniform_tile_mode:
        x1, y1, x2, y2 = crop_region
        crop_width, crop_height = x2 - x1, y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
            
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, 
                                  target_width, target_height)
        tile_size = p.width, p.height
    else:
        x1, y1, x2, y2 = crop_region
        target_width = math.ceil((x2 - x1) / 8) * 8
        target_height = math.ceil((y2 - y1) / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                          image_mask.height, target_width, target_height)

    # Apply mask blur if needed
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop and prepare tiles
    global batch
    tiles = [img.crop(crop_region) for img in batch]
    initial_tile_size = tiles[0].size

    # Resize tiles if needed
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Convert to tensor
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    
    try:
        # Step 1: Encode
        latent_tuple = p.vae_encoder.encode(p.vae, batched_tiles)
        latent = latent_tuple[0] if isinstance(latent_tuple, tuple) else latent_tuple
        
        # Step 2: Sample
        samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, 
                      positive_cropped, negative_cropped, latent, p.denoise, 
                      p.custom_sampler, p.custom_sigmas)
                      
        # Update progress
        if p.progress_bar_enabled:
            p.pbar.update(1)

        # Step 3: Decode - ensure samples are properly wrapped in a dictionary
        print(f"[VAE-DECODE-INFO] Preparing samples for decode. Type: {type(samples)}")
        
        # Always ensure samples is wrapped in a dictionary
        if not isinstance(samples, dict):
            samples_dict = {"samples": samples}
            print(f"[VAE-DECODE-INFO] Wrapped tensor in dict with 'samples' key")
        else:
            samples_dict = samples
            print(f"[VAE-DECODE-INFO] Using existing dictionary format")
            # Extra safety check - make sure 'samples' key exists
            if "samples" not in samples_dict:
                print(f"[VAE-DECODE-WARNING] Dict missing 'samples' key, adding it")
                samples_dict = {"samples": samples_dict}  # Wrap again
        
        # Now use samples_dict for both decode paths
        try:
            if not p.tiled_decode:
                # Standard decode
                print(f"[VAE-DECODE-START] Using standard decoder")
                decoded_tuple = p.vae_decoder.decode(p.vae, samples_dict)
                print(f"[VAE-DECODE-COMPLETE] Standard decode successful")
            else:
                # Tiled decode
                print(f"[VAE-TILED-DECODE-START] Using tiled decoder with tile size 512")
                decoded_tuple = p.vae_decoder_tiled.decode(p.vae, samples_dict, 512)
                print(f"[VAE-TILED-DECODE-COMPLETE] Tiled decode successful")
        except Exception as e:
            print(f"[VAE-DECODE-ERROR] Decode failed: {str(e)}")
            print(f"[VAE-DECODE-ERROR-TYPE] Error type: {e.__class__.__name__}")
            
            # Try another approach - manually create samples tensor if needed
            try:
                print(f"[VAE-DECODE-FALLBACK] Trying alternative approach")
                # Extract the actual tensor if possible
                if isinstance(samples, dict) and "samples" in samples:
                    samples_tensor = samples["samples"]
                else:
                    samples_tensor = samples
                    
                # Create a fresh dictionary
                fresh_dict = {"samples": samples_tensor}
                print(f"[VAE-DECODE-FALLBACK] Created fresh dictionary with tensor")
                
                # Try decode with this fresh dictionary 
                if not p.tiled_decode:
                    decoded_tuple = p.vae_decoder.decode(p.vae, fresh_dict)
                else:
                    decoded_tuple = p.vae_decoder_tiled.decode(p.vae, fresh_dict, 512)
                    
                print(f"[VAE-DECODE-FALLBACK-COMPLETE] Fallback decode successful")
            except Exception as fallback_error:
                print(f"[VAE-DECODE-CRITICAL] Fallback also failed: {str(fallback_error)}")
                # Last resort - create a blank output tensor with the same batch size
                if isinstance(samples, dict) and "samples" in samples:
                    batch_size = samples["samples"].shape[0] 
                elif isinstance(samples, torch.Tensor):
                    batch_size = samples.shape[0]
                else:
                    batch_size = 1
                    
                print(f"[VAE-DECODE-EMERGENCY] Creating empty tensor with batch size {batch_size}")
                # Create small blank RGB tensor as emergency fallback
                blank = torch.zeros((batch_size, 64, 64, 3))
                return (blank,)
            
        decoded = decoded_tuple[0] if isinstance(decoded_tuple, tuple) else decoded_tuple
        print(f"[UPSCALER] Decoded tensor shape: {decoded.shape}")

        # Process results - FIXED for dimension handling
        tiles_sampled = []
        try:
            # Handle different tensor shapes
            if len(decoded.shape) == 3:  # [H,W,C]
                # Single image without batch dimension
                img_tensor = decoded.unsqueeze(0)  # Add batch dimension [1,H,W,C]
                img = tensor_to_pil(img_tensor, 0)
                if img.size != initial_tile_size:
                    img = img.resize(initial_tile_size, Image.Resampling.LANCZOS)
                tiles_sampled.append(img)
            else:
                # Normal case with batch dimension [B,H,W,C]
                for i in range(decoded.shape[0]):
                    # Use a more direct conversion to avoid indexing issues
                    img_tensor = decoded[i].unsqueeze(0)  # Make [1,H,W,C]
                    img = tensor_to_pil(img_tensor, 0)
                    if img.size != initial_tile_size:
                        img = img.resize(initial_tile_size, Image.Resampling.LANCZOS)
                    tiles_sampled.append(img)
        except Exception as e:
            print(f"[UPSCALER] Error converting tensor to images: {e}")
            # Fallback - create a blank image
            img = Image.new('RGB', initial_tile_size, color='black')
            tiles_sampled.append(img)
            
        print(f"[UPSCALER] Created {len(tiles_sampled)} processed tiles")

        # Apply tiles to images
        for i, tile_sampled in enumerate(tiles_sampled):
            if i >= len(batch):
                break
                
            init_image = batch[i]
            
            # Create tile with transparency
            image_tile_only = Image.new('RGBA', init_image.size)
            image_tile_only.paste(tile_sampled, crop_region[:2])
            
            # Create an expanded mask based on the final crop region
            expanded_mask = Image.new('L', init_image.size, 0)  # Start with black (transparent)
            # Create a white rectangle matching the expanded crop region
            crop_mask = Image.new('L', (crop_region[2]-crop_region[0], crop_region[3]-crop_region[1]), 255)
            expanded_mask.paste(crop_mask, crop_region[:2])
            
            # Apply blur if specified
            if p.mask_blur > 0:
                expanded_mask = expanded_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
            
            # Apply expanded mask
            temp = image_tile_only.copy()
            temp.putalpha(expanded_mask)
            image_tile_only.paste(temp, image_tile_only)
            
            # Composite onto original
            result = init_image.convert('RGBA')
            result.alpha_composite(image_tile_only)
            batch[i] = result.convert('RGB')
            
    except Exception as e:
        print(f"[UPSCALER] Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        # Original batch is preserved on error
        
    return Processed(p, [batch[0]], p.seed, None)

# USDU script adapter
class Script:
    """
    Script class that serves as an adapter for USDU processing.
    """
    def run(self, p, _, **kwargs):
        """Run the built-in implementation directly with audit logging."""
        import uuid
        import time
        
        # Generate a unique process ID
        process_id = str(uuid.uuid4())[:8]
        
        print(f"[USDU-START-{process_id}] Beginning processing with {p.rows}x{p.cols} tiles at {time.strftime('%H:%M:%S')}")
        start_time = time.time()
        
        try:
            # Use our built-in implementation
            result = process_images(p)
            elapsed = time.time() - start_time
            print(f"[USDU-COMPLETE-{process_id}] Processing completed successfully after {elapsed:.2f}s at {time.strftime('%H:%M:%S')}")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[USDU-ERROR-{process_id}] Processing failed after {elapsed:.2f}s: {str(e)}")
            raise

def upscale_regions(image, crop_regions, target_dimensions, model, positive, negative, vae, seed,
                   steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                   mode_type, tile_width, tile_height, mask_blur, tile_padding,
                   seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                   seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
    """
    Upscale detected regions in the image.
    """
    # Set up globals
    global sd_upscalers, actual_upscaler, batch
    
    # Set the upscaler
    sd_upscalers[0] = UpscalerData()
    actual_upscaler = upscale_model
    
    # Extract crops from regions
    image_crops = []
    valid_regions = []
    valid_dimensions = []
    
    for i, region in enumerate(crop_regions):
        x1, y1, x2, y2 = region
        # Do boundary checking
        if (x1 >= image.shape[2] or y1 >= image.shape[1] or 
            x2 > image.shape[2] or y2 > image.shape[1] or
            x1 >= x2 or y1 >= y2):
            print(f"[UPSCALER] Skipping invalid region {i}: {region}")
            continue
            
        # Extract crop
        crop = image[:, y1:y2, x1:x2, :]
        image_crops.append(crop)
        valid_regions.append(region)
        
        # Add corresponding target dimension
        if i < len(target_dimensions):
            valid_dimensions.append(target_dimensions[i])
        else:
            # Default to 2x if no specified dimension
            valid_dimensions.append((int((x2-x1)*2), int((y2-y1)*2)))
        
        # Save debug image
        save_debug_image(crop, f"upscale_crop_{i}")
    
    # If no valid crops were extracted, return original image
    if len(image_crops) == 0:
        return image, "No valid crops extracted"
    
    # Process each region individually 
    result_images = []
    
    for i in range(len(image_crops)):
        print(f"[UPSCALER] Processing region {i+1}/{len(image_crops)}")
        
        # Convert to PIL for processing
        crop_tensor = image_crops[i]
        crop_pil = tensor_to_pil(crop_tensor)
        
        # Set up batch
        batch = [crop_pil]
        
        # Get target dimensions
        if i < len(valid_dimensions):
            target_width, target_height = valid_dimensions[i]
            # Calculate effective upscale factor
            width_factor = target_width / crop_pil.width
            height_factor = target_height / crop_pil.height
            effective_upscale = max(width_factor, height_factor)
        else:
            # Default to 2x
            effective_upscale = 2.0
            target_width = int(crop_pil.width * 2)
            target_height = int(crop_pil.height * 2)
        
        # Create processing object
        sdprocessing = StableDiffusionProcessing(
            crop_pil, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, effective_upscale, 
            force_uniform_tiles, tiled_decode, tile_width, tile_height, 
            MODES[mode_type], SEAM_FIX_MODES[seam_fix_mode]
        )
        
        # Set mask for processing
        sdprocessing.image_mask = Image.new('L', (crop_pil.width, crop_pil.height), 255)
        sdprocessing.init_images = batch
        
        # Process the crop
        try:
            script = Script()
            script_result = script.run(
                p=sdprocessing, 
                _=None, 
                tile_width=tile_width, 
                tile_height=tile_height,
                mask_blur=mask_blur, 
                padding=tile_padding, 
                seams_fix_width=seam_fix_width,
                seams_fix_denoise=seam_fix_denoise, 
                seams_fix_padding=seam_fix_padding,
                upscaler_index=0, 
                save_upscaled_image=False, 
                redraw_mode=MODES[mode_type],
                save_seams_fix_image=False, 
                seams_fix_mask_blur=seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], 
                target_size_type=2,
                custom_width=None, 
                custom_height=None, 
                custom_scale=effective_upscale
            )
            
            # The batch global variable should have been updated by the script
            # But let's make sure we got a valid result
            if batch is None or len(batch) == 0:
                print(f"[UPSCALER] Warning: Empty batch result for region {i}")
                # Use 2x scaled original as fallback
                from torchvision.transforms.functional import resize
                fallback = crop_tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
                fallback = resize(fallback, [target_height, target_width])
                fallback = fallback.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
                result_images.append(fallback)
                continue
            
            # Check if target size matches
            processed_img = batch[0]
            if processed_img.width != target_width or processed_img.height != target_height:
                processed_img = processed_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                batch[0] = processed_img
            
            # Save debug image of processed result
            save_debug_image(pil_to_tensor(batch[0]), f"processed_region_{i}")
            
            # Convert processed image back to tensor
            processed_tensor = pil_to_tensor(batch[0])
            
            # Store results
            result_images.append(processed_tensor)
            
        except Exception as e:
            print(f"[UPSCALER] Error processing region {i}: {e}")
            # Use 2x scaled original as fallback
            from torchvision.transforms.functional import resize
            fallback = crop_tensor.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]
            fallback = resize(fallback, [target_height, target_width])
            fallback = fallback.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
            result_images.append(fallback)
    
    # Return processed batch - but DON'T stack them since they have different dimensions
    if len(result_images) > 0:
        processed_batch = result_images[0] if len(result_images) == 1 else result_images
        save_debug_image(result_images[0], "processed_region_example")
    else:
        print("[UPSCALER] No regions were successfully processed")
        return image, "No regions successfully processed"
    
    return processed_batch, f"Processed {len(valid_regions)} regions"