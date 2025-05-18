"""
File: homage_tools/nodes/ht_save_image_plus.py
Version: 1.4.2
Description: Enhanced image saving node with integrated Florence2 captioning
"""

import os
import json
from PIL import Image, PngImagePlugin
import numpy as np
import torch
import logging
import folder_paths
from typing import Dict, Any, Tuple, Optional, Union

logger = logging.getLogger('HommageTools')

VERSION = "1.4.2"

#------------------------------------------------------------------------------
# Section 1: Helper Functions
#------------------------------------------------------------------------------
def ensure_directory_exists(directory: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False

def get_sequential_filename(base_path: str, extension: str) -> Tuple[str, int]:
    """Get next available filename with sequence number."""
    counter = 1
    while True:
        file_name = f"{base_path}_{counter:05d}{extension}"
        if not os.path.exists(file_name):
            return file_name, counter
        counter += 1

def write_text_file(file_path: str, content: str, encoding: str = 'UTF-8') -> bool:
    """Write content to text file with specified encoding."""
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write text file {file_path}: {str(e)}")
        return False

def get_total_pixels(image: torch.Tensor) -> int:
    """Calculate total number of pixels in an image."""
    if len(image.shape) == 4:  # BHWC format
        return image.shape[1] * image.shape[2]  # Height * Width
    elif len(image.shape) == 3:  # HWC format
        return image.shape[0] * image.shape[1]  # Height * Width
    else:
        return 0

def get_safe_relative_path(path: str, start: str) -> str:
    """Get relative path safely across different drives."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        # If on different drives, return just the path itself
        return os.path.basename(os.path.dirname(path))

#------------------------------------------------------------------------------
# Section 2: Main Node Class
#------------------------------------------------------------------------------
class HTSaveImagePlus:
    """Enhanced image saving node with multiple format, mask support, and auto-captioning."""
    
    CATEGORY = "HommageTools/output"
    FUNCTION = "save_image_plus"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "images": ("IMAGE", {
                    "tooltip": "Images in BHWC format to save"
                }),
                "output_format": (["PNG", "JPEG", "TIFF"], {
                    "default": "PNG",
                    "tooltip": "Output image format"
                }),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_",
                    "tooltip": "Prefix or full filename for output"
                }),
                "add_sequence_number": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append sequential number to filename"
                }),
                "output_dir": ("STRING", {
                    "default": folder_paths.get_output_directory(),
                    "tooltip": "Output directory path"
                }),
                "min_pixel_threshold": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,  # Using max 32-bit integer instead of arbitrary limit
                    "step": 1,          # Allow any integer
                    "tooltip": "Skip saving images with fewer pixels than this threshold"
                }),
                "crop_to_alpha": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Crop images to alpha channel content area"
                }),
                "enable_captioning": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable automatic captioning of saved images"
                })
            },
            "optional": {
                # Image Format Quality Options
                "jpeg_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG Quality (1-100)"
                }),
                "tiff_compression": (["adobe_deflate", "none"], {
                    "default": "adobe_deflate",
                    "tooltip": "TIFF compression method"
                }),
                # Mask/Alpha Options
                "mask": ("MASK", {
                    "tooltip": "Optional mask to use as alpha channel"
                }),
                "save_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save alpha channel if present"
                }),
                # Text Output Options  
                "text_content": ("STRING", {
                    "multiline": True,
                    "tooltip": "Optional text content to save alongside image",
                    "ui_height": 4    # Limit the height of the text input
                }),
                "text_extension": ("STRING", {
                    "default": ".txt",
                    "tooltip": "Extension for text file"
                }),
                "text_encoding": (["UTF-8", "ASCII", "UTF-16", "UTF-32"], {
                    "default": "UTF-8",
                    "tooltip": "Text file encoding"
                }),
                # Metadata control
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save prompt metadata in PNG files"
                }),
                # Florence2 Captioning Options
                "florence2_model": ("FL2MODEL", {
                    "tooltip": "Florence2 model for auto-captioning"
                }),
                "caption_type": (
                    ["caption", "detailed_caption", "more_detailed_caption"], {
                    "default": "detailed_caption",
                    "tooltip": "Type of caption to generate"
                }),
                "caption_pre": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "ui_height": 1,
                    "tooltip": "Text to add before the caption"
                }),
                "caption_post": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "ui_height": 1,
                    "tooltip": "Text to add after the caption"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, 
                    "min": 1, 
                    "max": 4096,
                    "tooltip": "Maximum number of tokens to generate for caption"
                }),
                "num_beams": ("INT", {
                    "default": 3, 
                    "min": 1, 
                    "max": 64,
                    "tooltip": "Number of beams for caption generation"
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to use sampling for caption generation"
                }),
                "seed": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for caption generation"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep Florence2 model loaded after processing"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    def _hash_seed(self, seed):
        """Hash seed for Florence2 captioning."""
        import hashlib
        seed_bytes = str(seed).encode('utf-8')
        hash_object = hashlib.sha256(seed_bytes)
        hashed_seed = int(hash_object.hexdigest(), 16)
        return hashed_seed % (2**32)

    def _generate_caption(self, image, florence2_model, caption_type, caption_pre="", caption_post="", 
                    num_beams=3, max_new_tokens=1024, do_sample=True, seed=None, keep_model_loaded=False):
        """Generate caption for an image using Florence2 model."""
        import comfy.model_management as mm
        from transformers import set_seed
        import torchvision.transforms.functional as F
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']
        model.to(device)
        
        if seed:
            set_seed(self._hash_seed(seed))

        prompts = {
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
        }
        
        task_prompt = prompts.get(caption_type, '<CAPTION>')
        prompt = task_prompt

        # Convert to format expected by Florence2
        try:
            print(f"[DEBUG] Image tensor shape before processing: {image.shape}")
            print(f"[DEBUG] Image tensor dtype: {image.dtype}")
            print(f"[DEBUG] Image tensor value range: min={image.min().item():.4f}, max={image.max().item():.4f}")
            
            # Skip tiny images - they cause problems with the captioning model
            if image.shape[1] <= 4 and image.shape[2] <= 4:
                print(f"[DEBUG] Image too small for captioning ({image.shape[1]}x{image.shape[2]}), skipping")
                return "Image too small for detailed captioning"
                
            # Check if the tensor has the expected format
            if len(image.shape) != 4:
                print(f"[DEBUG] WARNING: Expected 4D tensor (BHWC) but got {len(image.shape)}D")
                # Try to reshape if needed
                if len(image.shape) == 3:  # HWC format
                    print(f"[DEBUG] Converting HWC to BHWC format")
                    image = image.unsqueeze(0)
            
            # Handle alpha channel if present
            if image.shape[-1] == 4:
                print(f"[DEBUG] Removing alpha channel")
                image = image[..., :3]
                
            # Here's the key fix: we're expecting BHWC format, but need to convert to BCHW
            # for the model's processing pipeline
            image = image.permute(0, 3, 1, 2)
            print(f"[DEBUG] Image tensor shape after permute: {image.shape}")
            
            # Additional validation for unusually small images
            if image.shape[2] < 32 or image.shape[3] < 32:
                # For very small images, resize to a minimum size the model can handle
                print(f"[DEBUG] Resizing small image from {image.shape[2]}x{image.shape[3]} to 224x224")
                image = F.resize(image, [224, 224])
                
        except Exception as e:
            print(f"[DEBUG] Error during image format conversion: {str(e)}")
            return f"Error processing image for captioning: {str(e)}"

        out_results = []
        from comfy.utils import ProgressBar
        pbar = ProgressBar(len(image))
        
        try:
            for i, img in enumerate(image):
                print(f"Generating caption for image {i+1}/{len(image)}...")
                # Convert to PIL for the processor
                image_pil = F.to_pil_image(img)
                
                # Use a try-except block for the actual captioning
                try:
                    inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)

                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        num_beams=num_beams,
                    )

                    results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    
                    # Cleanup the special tokens
                    clean_results = str(results)       
                    clean_results = clean_results.replace('</s>', '')
                    clean_results = clean_results.replace('<s>', '')
                    
                    # Add pre and post text to the caption
                    processed_caption = f"{caption_pre}{clean_results}{caption_post}"
                except Exception as e:
                    print(f"[DEBUG] Error generating caption for image {i+1}: {str(e)}")
                    processed_caption = f"Unable to generate caption: {str(e)}"
                
                # Print caption to console with image index
                print(f"\n===== CAPTION FOR IMAGE {i+1}/{len(image)} =====")
                print(processed_caption)
                print("="*40 + "\n")

                # Return single string if only one image for compatibility
                if len(image) == 1:
                    out_results = processed_caption
                else:
                    out_results.append(processed_caption)
                    
                pbar.update(1)
        except Exception as e:
            print(f"[DEBUG] Error in caption generation loop: {str(e)}")
            return f"Error in caption generation: {str(e)}"

        if not keep_model_loaded:
            print("Offloading Florence2 model...")
            model.to(offload_device)
            mm.soft_empty_cache()
        
        return out_results

    def _crop_to_alpha(self, image: torch.Tensor) -> list:
        """
        Crop images to their alpha channel content areas.
        Returns a list of PIL images cropped to their content.
        """
        cropped_images = []
        print("Cropping images to alpha content...")
        
        # Ensure tensor is on CPU before numpy conversion
        image = image.cpu()
        
        # Process each image in batch
        for i in range(image.shape[0]):
            img = image[i]
            
            # Skip if no alpha channel
            if img.shape[-1] != 4:
                print(f"Image {i+1} has no alpha channel - keeping original dimensions")
                # Convert to PIL and add to results
                i = 255. * img.numpy()
                pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                cropped_images.append(pil_img)
                continue
                    
            # Get alpha channel
            alpha = img[:, :, 3]
            
            # Find non-zero alpha coordinates with multiple thresholds
            for threshold in [0.5, 0.1, 0.05, 0.01]:
                non_zero = torch.nonzero(alpha > threshold)
                if len(non_zero) > 100:  # We found enough points to make a bounding box
                    print(f"Image {i+1}: Found content with alpha threshold {threshold}")
                    break
            
            if len(non_zero) < 100:
                print(f"Image {i+1} has insufficient visible content in alpha mask")
                # Convert full image and use it instead
                i = 255. * img[:, :, :3].numpy()
                pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                cropped_images.append(pil_img)
                continue
                    
            # Get bounding box
            y_min = non_zero[:, 0].min().item()
            y_max = non_zero[:, 0].max().item() + 1
            x_min = non_zero[:, 1].min().item()
            x_max = non_zero[:, 1].max().item() + 1
            
            print(f"Image {i+1} content area: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            
            # Add a small padding around the content
            padding = 20
            y_min = max(0, y_min - padding)
            y_max = min(img.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(img.shape[1], x_max + padding)
            
            # Create RGB PIL image from cropped content
            rgb = img[y_min:y_max, x_min:x_max, :3]
            i = 255. * rgb.numpy()
            pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Store cropped dimensions for logging
            width, height = pil_img.size
            print(f"Cropped dimensions: {width}x{height}")
            
            cropped_images.append(pil_img)
        
        print(f"Cropping complete. Processed {len(cropped_images)} images.\n")
        return cropped_images

    def _prepare_image(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Image.Image:
        """Convert BHWC tensor to PIL Image, handling alpha channel."""
        # Ensure CPU tensors
        image = image.cpu()
        
        # Handle alpha channel
        if image.shape[-1] == 4:
            # Image already has alpha
            i = 255. * image.numpy()
            img_data = np.clip(i, 0, 255).astype(np.uint8)
        elif mask is not None:
            # Use provided mask as alpha
            mask = mask.cpu()
            i = 255. * image.numpy()
            rgb = np.clip(i, 0, 255).astype(np.uint8)
            # Ensure mask is 2D and matches image dimensions
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            a = 255. * (1 - mask.numpy())  # Invert mask for alpha
            img_data = np.concatenate([rgb, a[..., None]], axis=-1)
        else:
            # No alpha channel
            i = 255. * image.numpy()
            img_data = np.clip(i, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_data)

    def _save_image_with_format(
        self,
        image: Image.Image,
        file_path: str,
        format: str,
        save_alpha: bool = True,
        metadata: Optional[dict] = None,
        **kwargs
    ) -> bool:
        """Save image with specified format and options."""
        try:
            if not save_alpha and image.mode == 'RGBA':
                image = image.convert('RGB')
                
            if format == "PNG":
                # Create metadata for PNG if needed
                png_info = None
                if metadata and kwargs.get('save_metadata', True):
                    png_info = PngImagePlugin.PngInfo()
                    for k, v in metadata.items():
                        if isinstance(v, dict) or isinstance(v, list):
                            v = json.dumps(v)
                        png_info.add_text(str(k), str(v))

                image.save(file_path, format='PNG', pnginfo=png_info)
            elif format == "JPEG":
                # JPEG doesn't support alpha, always convert
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(file_path, format='JPEG', 
                         quality=kwargs.get('jpeg_quality', 90))
            elif format == "TIFF":
                # TIFF can handle alpha channel
                image.save(file_path, format='TIFF',
                         compression=kwargs.get('tiff_compression', 'adobe_deflate'))
            return True
        except Exception as e:
            logger.error(f"Failed to save image {file_path}: {str(e)}")
            return False

    def save_image_plus(
        self,
        images: torch.Tensor,
        output_format: str,
        filename_prefix: str,
        add_sequence_number: bool,
        output_dir: str,
        min_pixel_threshold: int,
        crop_to_alpha: bool = False,
        enable_captioning: bool = False,
        prompt: Optional[Dict] = None,
        extra_pnginfo: Optional[Dict] = None,
        # Optional parameters
        jpeg_quality: int = 90,
        tiff_compression: str = "adobe_deflate",
        mask: Optional[torch.Tensor] = None,
        save_alpha: bool = True,
        text_content: Optional[str] = None,
        text_extension: str = ".txt",
        text_encoding: str = "UTF-8",
        save_metadata: bool = True,
        # Florence2 Captioning Options
        florence2_model = None,
        caption_type: str = "detailed_caption",
        caption_pre: str = "",
        caption_post: str = "",
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        do_sample: bool = True,
        seed: int = 1,
        keep_model_loaded: bool = False
    ) -> Dict:
        """Process and save images with optional mask, text output, and auto-captioning."""
        print(f"\nHTSaveImagePlus v{VERSION} - Processing")
        print(f"Target directory: {output_dir}")
        print(f"Format: {output_format}, Sequence numbering: {'Enabled' if add_sequence_number else 'Disabled'}")
        print(f"Minimum pixel threshold: {min_pixel_threshold}")
        print(f"Auto-captioning: {'Enabled' if enable_captioning else 'Disabled'}")
        print(f"Crop to alpha: {'Enabled' if crop_to_alpha else 'Disabled'}")
        
        # Add detailed image tensor debugging
        print(f"[DEBUG] Input images tensor shape: {images.shape}")
        print(f"[DEBUG] Images tensor dtype: {images.dtype}")
        print(f"[DEBUG] Images value range: min={images.min().item():.4f}, max={images.max().item():.4f}")
        dims = len(images.shape)
        print(f"[DEBUG] Images tensor dimensions: {dims}D")
        
        if dims == 4:
            print(f"[DEBUG] Batch size: {images.shape[0]}, Height: {images.shape[1]}, Width: {images.shape[2]}, Channels: {images.shape[3]}")
            if images.shape[3] not in [3, 4]:
                print(f"[DEBUG] WARNING: Unexpected number of channels: {images.shape[3]} (expected 3 or 4)")
        elif dims == 3:
            print(f"[DEBUG] Single image - Height: {images.shape[0]}, Width: {images.shape[1]}, Channels: {images.shape[2]}")
            if images.shape[2] not in [3, 4]:
                print(f"[DEBUG] WARNING: Unexpected number of channels: {images.shape[2]} (expected 3 or 4)")
        
        # Ensure output directory exists
        if not ensure_directory_exists(output_dir):
            return {"ui": {"error": f"Failed to create output directory: {output_dir}"}}

        results = []
        skipped_count = 0
        
        # STEP 1: HANDLE ALPHA CROPPING FIRST
        processed_images = images  # Default to original images
        cropped_list = None
        if crop_to_alpha:
            print(f"Crop to alpha: Enabled - performing cropping before captioning")
            cropped_list = self._crop_to_alpha(images)
            if cropped_list and len(cropped_list) > 0:
                print(f"Successfully cropped {len(cropped_list)} images to alpha content")
                # We'll handle cropped images directly rather than trying to convert back to same-sized tensors
            else:
                print("No content found in alpha masks, using original images")

        # STEP 2: GENERATE CAPTIONS ON PROCESSED IMAGES
        generated_captions = None
        if enable_captioning and florence2_model is not None:
            try:
                if crop_to_alpha and cropped_list and len(cropped_list) > 0:
                    # Process each image individually
                    all_captions = []
                    print(f"Generating captions using {caption_type} on cropped images...")
                    
                    for i, pil_img in enumerate(cropped_list):
                        print(f"Processing caption for cropped image {i+1}/{len(cropped_list)}...")
                        # Convert PIL to tensor for just this one image
                        img_np = np.array(pil_img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np)
                        # Add batch dimension
                        img_tensor = img_tensor.unsqueeze(0)
                        
                        # Generate caption for just this image
                        caption = self._generate_caption(
                            img_tensor,
                            florence2_model,
                            caption_type,
                            caption_pre,
                            caption_post,
                            num_beams,
                            max_new_tokens,
                            do_sample,
                            seed + i if seed else None,
                            i == len(cropped_list) - 1  # Only keep model loaded on last image
                        )
                        
                        all_captions.append(caption)
                        
                    generated_captions = all_captions
                else:
                    # Original method for non-cropped images
                    print(f"Generating captions using {caption_type} on original images...")
                    generated_captions = self._generate_caption(
                        images,
                        florence2_model,
                        caption_type,
                        caption_pre,
                        caption_post,
                        num_beams,
                        max_new_tokens,
                        do_sample,
                        seed,
                        keep_model_loaded
                    )
                    print(f"Captions generated successfully")
            except Exception as e:
                print(f"Error generating captions: {str(e)}")
                import traceback
                traceback.print_exc()
                generated_captions = None
        
        # STEP 3: PREPARE METADATA FOR PNG
        metadata = None
        if output_format == "PNG" and save_metadata:
            metadata = {}
            if prompt is not None:
                metadata["prompt"] = prompt
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)

        # STEP 4: SAVE IMAGES
        # If we already have cropped PIL images, use those directly
        if crop_to_alpha and 'cropped_list' in locals() and cropped_list and len(cropped_list) > 0:
            for idx, img in enumerate(cropped_list):
                # Create base filename
                base_name = os.path.join(output_dir, filename_prefix)
                extension = f".{output_format.lower()}"
                
                if add_sequence_number:
                    file_path, counter = get_sequential_filename(base_name, extension)
                else:
                    file_path = base_name + extension
                    counter = idx
                
                # Check size threshold
                width, height = img.size
                total_pixels = width * height
                if total_pixels < min_pixel_threshold:
                    print(f"Skipping image {idx}: {total_pixels} pixels (below threshold of {min_pixel_threshold})")
                    skipped_count += 1
                    continue
                
                # Save the image
                if self._save_image_with_format(
                    img, 
                    file_path,
                    output_format,
                    save_alpha=False,  # Already cropped to content, no need for alpha
                    metadata=metadata if output_format == "PNG" and save_metadata else None,
                    jpeg_quality=jpeg_quality,
                    tiff_compression=tiff_compression,
                    save_metadata=save_metadata
                ):
                    # Log successful image save
                    print(f"Successfully saved cropped image: {file_path} ({total_pixels} pixels)")
                    
                    # Use safe relative path function for cross-drive compatibility
                    subfolder = get_safe_relative_path(output_dir, folder_paths.get_output_directory())
                    
                    results.append({
                        "filename": os.path.basename(file_path),
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    
                    # Save accompanying text
                    current_text = text_content or ""
                    # If captioning is enabled and captions were generated, use them
                    if generated_captions:
                        if isinstance(generated_captions, list) and idx < len(generated_captions):
                            # Append text_content to the generated caption if text_content exists
                            if current_text:
                                current_text = f"{generated_captions[idx]}\n{current_text}"
                            else:
                                current_text = generated_captions[idx]
                        elif isinstance(generated_captions, str):
                            # Append text_content to the generated caption if text_content exists
                            if current_text:
                                current_text = f"{generated_captions}\n{current_text}"
                            else:
                                current_text = generated_captions
                                
                    if current_text:
                        text_path = os.path.splitext(file_path)[0] + text_extension
                        if write_text_file(text_path, current_text, text_encoding):
                            print(f"Successfully saved text file: {text_path}")
        else:
            # Process original images if not using cropped versions
            for idx, image in enumerate(images):
                # Check pixel threshold
                total_pixels = get_total_pixels(image)
                if total_pixels < min_pixel_threshold:
                    print(f"Skipping image {idx}: {total_pixels} pixels (below threshold of {min_pixel_threshold})")
                    skipped_count += 1
                    continue
                    
                # Create base filename
                base_name = os.path.join(output_dir, filename_prefix)
                extension = f".{output_format.lower()}"
                
                if add_sequence_number:
                    file_path, counter = get_sequential_filename(base_name, extension)
                else:
                    file_path = base_name + extension
                    counter = idx
                
                # Get mask for this image if provided
                current_mask = None
                if mask is not None:
                    if len(mask.shape) == 4:  # BHWC format
                        current_mask = mask[idx] if idx < mask.shape[0] else mask[0]
                    else:  # Single mask for all images
                        current_mask = mask
                
                # Convert and save image
                pil_image = self._prepare_image(image, current_mask)
                if self._save_image_with_format(
                    pil_image, 
                    file_path,
                    output_format,
                    save_alpha=save_alpha,
                    metadata=metadata,
                    jpeg_quality=jpeg_quality,
                    tiff_compression=tiff_compression,
                    save_metadata=save_metadata
                ):
                    # Log successful image save to console
                    print(f"Successfully saved image: {file_path} ({total_pixels} pixels)")
                    
                    # Use safe relative path function for cross-drive compatibility
                    subfolder = get_safe_relative_path(output_dir, folder_paths.get_output_directory())
                    
                    results.append({
                        "filename": os.path.basename(file_path),
                        "subfolder": subfolder,
                        "type": "output"
                    })
                    
                    # Save accompanying text
                    current_text = text_content or ""
                    # If captioning is enabled and captions were generated, use them
                    if generated_captions:
                        if isinstance(generated_captions, list) and idx < len(generated_captions):
                            # Append text_content to the generated caption if text_content exists
                            if current_text:
                                current_text = f"{generated_captions[idx]}\n{current_text}"
                            else:
                                current_text = generated_captions[idx]
                        elif isinstance(generated_captions, str):
                            # Append text_content to the generated caption if text_content exists
                            if current_text:
                                current_text = f"{generated_captions}\n{current_text}"
                            else:
                                current_text = generated_captions
                            
                    if current_text:
                        text_path = os.path.splitext(file_path)[0] + text_extension
                        if write_text_file(text_path, current_text, text_encoding):
                            print(f"Successfully saved text file: {text_path}")
                        else:
                            logger.warning(f"Failed to save text file for image {file_path}")

        if skipped_count > 0:
            print(f"Skipped {skipped_count} images that were below the pixel threshold of {min_pixel_threshold}")
                
        return {"ui": {"images": results}}

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "HTSaveImagePlus": HTSaveImagePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HTSaveImagePlus": "HT Save Image Plus"
}