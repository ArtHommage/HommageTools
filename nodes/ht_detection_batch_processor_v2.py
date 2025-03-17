import os
import json
import math
import numpy as np
from enum import Enum
from tqdm import tqdm

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
# Utility Functions
#-------------------------------------------------------

def pil_to_tensor(image):
    """Convert PIL image to PyTorch tensor"""
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None,]
    return img.permute(0, 3, 1, 2)

def tensor_to_pil(tensor, index=0):
    """
    Convert a PyTorch tensor to a PIL image, ensuring the tensor shape
    is compatible (B, C, H, W), with C in [1, 3, 4].
    """
    # Check the number of dimensions
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D tensor of shape (B,C,H,W), but got shape {tuple(tensor.shape)}")
    
    b, c, h, w = tensor.shape
    if c not in [1, 3, 4]:
        raise ValueError(
            f"Expected channel dimension in [1, 3, 4], but got c={c}. "
            "Make sure your tensor is actually an image (e.g., grayscale, RGB, or RGBA)."
        )
    
    # Permute from (B,C,H,W) to (H,W,C) for PIL and pick a batch index
    i = tensor[index].cpu().permute(1, 2, 0).numpy()
    # Scale if data is in [0,1] range, then convert to uint8
    i = (255.0 * i).clip(0, 255).astype("uint8")
    
    return Image.fromarray(i)

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

def flatten(img, bgcolor):
    """Replace transparency with bgcolor"""
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")

#-------------------------------------------------------
# Core USDU Components
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
    """Wrapper for VAE encoding"""
    def encode(self, vae, image_tensor):
        return (vae.encode(image_tensor),)

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
    """Sample from the model with the given parameters"""
    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
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
    
    # Default KSampler
    sampler = KSampler()
    (samples,) = sampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)
    return samples

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
    """Main processing function for USDU"""
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

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                   negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled and p.pbar is not None:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    result_images = []
    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = p.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')
        result_images.append(result)

    return Processed(p, result_images, p.seed, None)

#-------------------------------------------------------
# ComfyUI Node Implementation
#-------------------------------------------------------

class HTDetectionBatchProcessorV2:
    """ComfyUI node for UltimateSDUpscaler"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "tiled_decode": ("BOOLEAN", {"default": True}),
                "uniform_tile_mode": ("BOOLEAN", {"default": True}),
                "redraw_mode": (["linear", "chess", "none"], {"default": "linear"}),
                "seam_fix_mode": (["none", "band_pass", "half_tile", "half_tile_plus_intersections"], {"default": "band_pass"}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "custom_sampler": ("SAMPLER_V2",),
                "custom_sigmas": ("SIGMAS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscalers"
    
    def upscale(self, model, vae, image, positive, negative, upscale_by, seed, steps, cfg, 
                sampler_name, scheduler, denoise, tile_width, tile_height, tiled_decode, 
                uniform_tile_mode, redraw_mode, seam_fix_mode, upscale_model=None, 
                custom_sampler=None, custom_sigmas=None):
        
        # Validate input
        if image is None or image.size(0) == 0:
            print("[USDU] Empty input image, returning empty result")
            return (torch.zeros((0, 3, 8, 8)),)
        
        # Process each image in the batch
        result_images = []
        
        for i in range(image.size(0)):
            # Convert tensor to PIL
            img = tensor_to_pil(image, i)
            
            # Check if image is too small (less than 100 pixels)
            if img.width * img.height < 100:
                print(f"[USDU] Image {i} is too small ({img.width}x{img.height}), skipping")
                continue
            
            # Map string values to enum values
            redraw_mode_enum = USDUMode[redraw_mode.upper()]
            seam_fix_mode_enum = USDUSFMode[seam_fix_mode.upper()]
            
            # Create upscaler
            upscaler = Upscaler(upscale_model)
            
            # Upscale the image first
            upscaled_img = upscaler.upscale(img, upscale_by)
            
            # Create a mask that covers the entire image
            mask = Image.new('L', upscaled_img.size, 255)
            
            # Initialize processing
            p = StableDiffusionProcessing(
                init_img=upscaled_img,
                model=model,
                positive=positive,
                negative=negative,
                vae=vae,
                seed=seed + i,  # Use different seed for each image in batch
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                upscale_by=upscale_by,
                uniform_tile_mode=uniform_tile_mode,
                tiled_decode=tiled_decode,
                tile_width=tile_width,
                tile_height=tile_height,
                redraw_mode=redraw_mode_enum,
                seam_fix_mode=seam_fix_mode_enum,
                batch=[upscaled_img],
                custom_sampler=custom_sampler,
                custom_sigmas=custom_sigmas
            )
            
            # Set the mask
            p.image_mask = mask
            
            # Process the image
            processed = process_images(p)
            
            # Add to result list
            result_images.append(processed.images[0])
        
        # If no valid images were processed, return empty tensor
        if len(result_images) == 0:
            print("[USDU] No valid images to process, returning empty result")
            return (torch.zeros((0, 3, 8, 8)),)
        
        # Convert all results back to tensors and stack
        result_tensors = torch.cat([pil_to_tensor(img) for img in result_images], dim=0)
        
        return (result_tensors,)

# Node registration function
NODE_CLASS_MAPPINGS = {
    "HTDetectionBatchProcessorV2": HTDetectionBatchProcessorV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HTDetectionBatchProcessorV2": "HT Detection Batch Processor V2"
}

# Default config file creation if it doesn't exist
def create_default_config():
    default_config = {
        "per_tile_progress": True
    }
    
    config_path = os.path.join(os.path.dirname(__file__), 'usdu_config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)

create_default_config()