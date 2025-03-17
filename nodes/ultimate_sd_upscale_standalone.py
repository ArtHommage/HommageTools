import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import comfy.model_management as model_management
from comfy.sd import load_checkpoint_guess_config
import comfy.utils
import comfy.sd
import folder_paths
from typing import List, Dict, Any, Tuple, Optional, Union


class TileProcessor:
    """Handles image tiling operations for processing large images in smaller chunks"""
    @classmethod
    def get_tiles_and_masks(cls, image, tile_width, tile_height, overlap):
        """
        Split an image into tiles with specified overlap between adjacent tiles.
        Returns both tiles and corresponding masks for seamless blending.
        """
        image_width, image_height = image.shape[2], image.shape[1]
        
        # Calculate parameters for tiling
        non_overlap_width = tile_width - overlap
        non_overlap_height = tile_height - overlap
        
        # Number of tiles in each dimension
        num_tiles_x = math.ceil((image_width - overlap) / non_overlap_width)
        num_tiles_y = math.ceil((image_height - overlap) / non_overlap_height)
        
        # Adjust image size for even tiling if needed
        new_width = num_tiles_x * non_overlap_width + overlap
        new_height = num_tiles_y * non_overlap_height + overlap
        
        # Create padded image if necessary
        padded = False
        if new_width != image_width or new_height != image_height:
            # Calculate padding
            pad_x = new_width - image_width
            pad_y = new_height - image_height
            
            # Apply padding
            padded_image = F.pad(image, (0, pad_x, 0, pad_y), mode='reflect')
            padded = True
        else:
            padded_image = image
        
        tiles = []
        masks = []
        
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                # Calculate tile boundaries
                x_start = x * non_overlap_width
                y_start = y * non_overlap_height
                x_end = min(x_start + tile_width, new_width)
                y_end = min(y_start + tile_height, new_height)
                
                # Extract the tile
                tile = padded_image[:, y_start:y_end, x_start:x_end].unsqueeze(0)
                tiles.append((tile, (x_start, y_start, x_end, y_end)))
                
                # Create blending mask for this tile
                mask = torch.ones((1, y_end - y_start, x_end - x_start), 
                                  device=padded_image.device)
                
                # Apply feathering to edges that overlap with other tiles
                if x > 0:  # Left edge
                    for i in range(overlap):
                        mask[:, :, i] = i / overlap
                if x < num_tiles_x - 1 and x_end == x_start + tile_width:  # Right edge
                    for i in range(overlap):
                        mask[:, :, -(i+1)] = i / overlap
                if y > 0:  # Top edge
                    for i in range(overlap):
                        mask[:, i, :] = i / overlap
                if y < num_tiles_y - 1 and y_end == y_start + tile_height:  # Bottom edge
                    for i in range(overlap):
                        mask[:, -(i+1), :] = i / overlap
                
                masks.append(mask)
        
        return tiles, masks, padded, (image_width, image_height)

    @classmethod
    def merge_tiles(cls, tiles_with_coords, masks, original_size, device):
        """
        Merge processed tiles back into a single image using masks for blending
        """
        # Determine the full output dimensions
        max_x = max([coords[2] for _, coords in tiles_with_coords])
        max_y = max([coords[3] for _, coords in tiles_with_coords])
        
        # Create empty image and weight map
        channels = tiles_with_coords[0][0].shape[1]
        merged = torch.zeros((channels, max_y, max_x), device=device)
        weights = torch.zeros((1, max_y, max_x), device=device)
        
        # Place each tile with its mask
        for idx, (tile, (x_start, y_start, x_end, y_end)) in enumerate(tiles_with_coords):
            mask = masks[idx]
            merged[:, y_start:y_end, x_start:x_end] += tile.squeeze(0) * mask
            weights[:, y_start:y_end, x_start:x_end] += mask
        
        # Normalize by the weight map
        merged = torch.where(weights > 0, merged / weights, merged)
        
        # Crop to original dimensions if different
        if (max_x, max_y) != original_size:
            merged = merged[:, :original_size[1], :original_size[0]]
        
        return merged


class UltimateSDUpscaleStandalone:
    """
    Standalone implementation of UltimateSDUpscale for ComfyUI
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "upscale_by": (["2x", "3x", "4x"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512}),
            },
            "optional": {
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def upscale(self, image, model, positive, negative, upscale_by, steps, cfg, 
                sampler_name, scheduler, denoise, tile_width, tile_height, 
                tile_overlap, vae=None, seed=None):
        """
        Process an image by upscaling it using Stable Diffusion
        with tiled processing to handle large images.
        """
        scale_factor = int(upscale_by[0])
        device = model_management.get_torch_device()
        
        # Convert scale factor text to numeric
        scale_factor = int(upscale_by[0])
        
        # Initialize VAE if not provided
        if vae is None:
            vae = model.first_stage_model
            
        # Initial upscale using bicubic interpolation
        batch_size, height, width, _ = image.shape
        scaled_height = height * scale_factor
        scaled_width = width * scale_factor
        
        # Convert to RGB if needed
        if image.shape[3] == 4:
            image = image[:, :, :, :3]
            
        # Preprocessing - convert from NHWC to NCHW and then normalize for VAE
        latent_image = image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        latent_image = comfy.utils.common_upscale(latent_image, scaled_width, scaled_height, "bicubic", "disabled")
        
        # Prepare for tile processing
        first_tile = True
        processed_image = None
        
        # Generate random seed if not provided
        if seed is None or seed == 0:
            seed = torch.randint(0, 2**63 - 1, (1,)).item()
        
        # Initialize latent space image tensor
        latent_tiles, latent_masks, padded, orig_size = TileProcessor.get_tiles_and_masks(
            latent_image[0], tile_width, tile_height, tile_overlap
        )
        
        processed_tiles = []
        
        # Process each tile with SD upscaling
        for idx, (latent_tile, coords) in enumerate(latent_tiles):
            # Encode tile to latent space
            with torch.no_grad():
                # Scale values for VAE if needed (0-1 to -1 to 1)
                vae_input = latent_tile * 2.0 - 1.0
                tile_latent = vae.encode(vae_input)
            
            tile_latent = tile_latent * 0.18215
            
            # Set up sampler with consistent seed between tiles
            sampler = comfy.samplers.KSampler(
                model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, 
                denoise=denoise, model_options={"cond": positive, "uncond": negative}
            )
            
            # Get latent size
            tile_seed = seed + idx  # Vary seed slightly per tile
            
            # Run stable diffusion sampling
            samples = sampler.sample(
                tile_seed, tile_latent, cfg=cfg, latent_image=tile_latent
            )
            
            # Decode back from latent space
            with torch.no_grad():
                decoded = vae.decode(samples.to(vae.dtype) / 0.18215)
            
            # Normalize to 0-1 range
            decoded = (decoded + 1.0) / 2.0
            processed_tiles.append((decoded, coords))
        
        # Merge all the tiles back together
        merged_image = TileProcessor.merge_tiles(
            processed_tiles, 
            latent_masks,
            orig_size, 
            device
        )
        
        # Convert back to NHWC format for ComfyUI
        result = merged_image.unsqueeze(0).permute(0, 2, 3, 1)  # NCHW -> NHWC
        
        # Clamp values to valid range
        result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)


# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleStandalone": UltimateSDUpscaleStandalone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleStandalone": "Ultimate SD Upscale (Standalone)"
}