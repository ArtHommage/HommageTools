"""
File: homage_tools/nodes/ht_gemini_image_node.py
Version: 1.0.0
Description: Gemini-based image generation node with prompt enhancement and style transfer capabilities
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Configuration
#------------------------------------------------------------------------------
import os
import torch
import base64
import requests
import json
import time
from typing import Dict, Any, Tuple, Optional, List, Union
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import google.generativeai as genai
import folder_paths

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.0.0"

#------------------------------------------------------------------------------
# Section 2: Helper Functions
#------------------------------------------------------------------------------
def is_running_in_colab() -> bool:
    """
    Check if the code is running in Google Colab environment.
    
    Returns:
        bool: True if running in Colab, False otherwise
    """
    try:
        import google.colab
        # Check if the kernel object exists
        return hasattr(google.colab, '_kernel') or hasattr(google.colab, 'kernel')
    except ImportError:
        return False

def get_api_key() -> str:
    """
    Get the Google API key from various sources.
    
    Returns:
        str: The API key
    
    Raises:
        ValueError: If API key is not found
    """
    # First try environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        logger.info("API key found in environment variables")
        return api_key
        
    # Then try Colab userdata if available
    if is_running_in_colab():
        try:
            from google.colab import userdata
            api_key = userdata.get('GOOGLE_API_KEY')
            if api_key:
                logger.info("API key found in Colab userdata")
                return api_key
            else:
                logger.warning("API key not found in Colab userdata")
        except Exception as e:
            logger.warning(f"Could not access Colab userdata: {str(e)}")
    
    # Finally, try loading from file in ComfyUI directory
    try:
        # Get ComfyUI user directory
        import folder_paths
        user_dir = folder_paths.get_user_directory()
        key_file_path = os.path.join(user_dir, "default", "GOOGLE.key")
        
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r') as key_file:
                api_key = key_file.read().strip()
                if api_key:
                    logger.info(f"API key found in file: {key_file_path}")
                    return api_key
    except Exception as e:
        logger.warning(f"Could not read API key from file: {str(e)}")
    
    raise ValueError("Google API key not found. Set it in environment variables, Colab userdata, or place it in user/default/GOOGLE.key file.")

def test_api_connectivity(api_key: str) -> Tuple[bool, str]:
    """
    Test connectivity to the Gemini API and validate the API key.
    
    Args:
        api_key: Google API key to test
        
    Returns:
        Tuple[bool, str]: Success status and message
    """
    try:
        # Initialize genai with the provided key
        genai.configure(api_key=api_key)
        
        # Try to list models as a connectivity test
        models = genai.list_models()
        model_count = len(list(models))
        
        return True, f"API connected successfully. Found {model_count} available models."
    except Exception as e:
        error_msg = str(e)
        if "invalid api key" in error_msg.lower():
            return False, "Invalid API key. Please check your API key configuration."
        elif "quota" in error_msg.lower():
            return False, "API quota exceeded. Please check your usage limits."
        elif "not available" in error_msg.lower():
            return False, "Gemini API not available in your region."
        else:
            return False, f"API connection error: {error_msg}"

def tensor_to_base64(image_tensor: torch.Tensor, high_quality: bool = True) -> Tuple[str, str]:
    """
    Convert image tensor to base64 string for API transmission.
    
    Args:
        image_tensor: Image tensor in BHWC format
        high_quality: Whether to use high quality PNG (True) or JPEG (False)
        
    Returns:
        Tuple[str, str]: Base64 encoded image and mime type
    """
    # Ensure tensor is on CPU and convert to numpy
    tensor = image_tensor.detach().cpu()
    
    # Handle different tensor formats - ensure BHWC
    if len(tensor.shape) == 3:  # HWC format
        tensor = tensor.unsqueeze(0)  # Add batch dimension -> BHWC
    elif len(tensor.shape) == 4 and tensor.shape[1] == 3:  # BCHW format
        tensor = tensor.permute(0, 2, 3, 1)  # Convert to BHWC
        
    # Take first image if batch
    image_data = tensor[0]  # HWC format
    
    # Convert to PIL Image
    if image_data.shape[-1] == 3:  # RGB
        pil_image = Image.fromarray((image_data * 255).clamp(0, 255).byte().numpy(), 'RGB')
    elif image_data.shape[-1] == 4:  # RGBA
        pil_image = Image.fromarray((image_data * 255).clamp(0, 255).byte().numpy(), 'RGBA')
    elif len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[-1] == 1):  
        # Grayscale - convert to RGB
        if len(image_data.shape) == 3:
            image_data = image_data.squeeze(-1)  # Remove channel dimension
        pil_image = Image.fromarray((image_data * 255).clamp(0, 255).byte().numpy(), 'L').convert('RGB')
    else:
        raise ValueError(f"Unsupported image format with shape {image_data.shape}")
    
    # Convert to base64
    buffered = BytesIO()
    if high_quality:
        pil_image.save(buffered, format="PNG")
        mime_type = "image/png"
    else:
        pil_image.save(buffered, format="JPEG", quality=90)
        mime_type = "image/jpeg"
        
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str, mime_type

def initialize_genai(api_key: str) -> None:
    """
    Initialize the Gemini API with the provided key.
    
    Args:
        api_key: Google API key
    """
    genai.configure(api_key=api_key)

def get_available_models() -> List[str]:
    """
    Get list of available Gemini models from the API.
    
    Returns:
        List[str]: Names of available models
    """
    try:
        # Get API key and initialize
        api_key = get_api_key()
        initialize_genai(api_key)
        
        # Get models
        models = genai.list_models()
        gemini_models = []
        
        # Filter for Gemini models with vision capabilities
        for model in models:
            if "gemini" in model.name and hasattr(model, "supported_generation_methods"):
                supported = model.supported_generation_methods
                if "generateContent" in supported:
                    model_name = model.name.split('/')[-1]
                    gemini_models.append(model_name)
        
        if not gemini_models:
            # Default list of commonly used vision-capable Gemini models
            return [
                # Recommended Vision-capable models
                "gemini-2.5-pro-preview", 
                "gemini-2.0-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash-preview"
            ]
            
        # Sort models by capability and recency
        sorted_models = []
        
        # Add recommended models first
        for prefix in ["gemini-2.5-", "gemini-2.0-", "gemini-1.5-"]:
            for model in sorted([m for m in gemini_models if m.startswith(prefix)], reverse=True):
                sorted_models.append(model)
        
        # Add any remaining models
        for model in gemini_models:
            if model not in sorted_models:
                sorted_models.append(model)
                
        return sorted_models
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        # Default list of vision-capable models
        return [
            "gemini-2.5-pro-preview", 
            "gemini-2.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash-preview"
        ]

def create_default_tensor(height: int = 512, width: int = 512) -> torch.Tensor:
    """
    Create a default placeholder image tensor.
    
    Args:
        height: Height of the image
        width: Width of the image
        
    Returns:
        torch.Tensor: Placeholder image tensor in BHWC format
    """
    # Create a gradient background
    y = torch.linspace(0, 1, height).view(-1, 1).repeat(1, width)
    x = torch.linspace(0, 1, width).view(1, -1).repeat(height, 1)
    
    # Create RGB channels
    r = x * 0.8
    g = y * 0.6
    b = (1 - x) * (1 - y) * 0.5
    
    # Add semi-transparent overlay text markers
    grid = ((x * 10).floor() + (y * 10).floor()) % 2
    grid = grid * 0.1 + 0.9  # Make grid subtle
    
    # Combine channels with grid overlay
    r = r * grid
    g = g * grid
    b = b * grid
    
    # Assemble RGB image and add batch dimension
    image = torch.stack([r, g, b], dim=2)
    image = image.unsqueeze(0)  # Add batch dimension -> BHWC
    
    return image

def text_to_tensor(text: str, width: int = 512, height: int = 512) -> torch.Tensor:
    """
    Create an image tensor with rendered text.
    This is a placeholder tensor with embedded text.
    
    Args:
        text: Text to embed in the image
        width: Width of the image
        height: Height of the image
        
    Returns:
        torch.Tensor: Image tensor with text in BHWC format
    """
    # Create a base gradient image
    base_tensor = create_default_tensor(height, width)
    
    # For actual text rendering, we'd need more complex code using PIL
    # For now, this is a placeholder that creates a distinctive pattern
    # based on the text content
    
    # Hash the text to create a deterministic pattern
    text_hash = hash(text) % 1000 / 1000
    
    # Modify the base image using the hash
    modified = base_tensor.clone()
    
    # Add some pattern based on text (simple approach)
    b, h, w, c = modified.shape
    y_pattern = torch.sin(torch.linspace(0, 10 * np.pi * text_hash, h)).view(-1, 1).repeat(1, w)
    x_pattern = torch.cos(torch.linspace(0, 10 * np.pi * (1-text_hash), w)).view(1, -1).repeat(h, 1)
    
    # Apply pattern to channels differently
    modified[0, :, :, 0] = 0.7 * modified[0, :, :, 0] + 0.3 * y_pattern
    modified[0, :, :, 1] = 0.7 * modified[0, :, :, 1] + 0.3 * x_pattern
    modified[0, :, :, 2] = 0.7 * modified[0, :, :, 2] + 0.3 * (x_pattern * y_pattern)
    
    return modified

#------------------------------------------------------------------------------
# Section 3: Image Generation Modes
#------------------------------------------------------------------------------
def enhance_prompt(
    prompt: str,
    enhancement_type: str,
    gemini_model,
    max_tokens: int = 1000
) -> str:
    """
    Enhance a prompt using Gemini API.
    
    Args:
        prompt: Original prompt
        enhancement_type: Type of enhancement to apply
        gemini_model: Initialized Gemini model
        max_tokens: Maximum tokens in response
        
    Returns:
        str: Enhanced prompt
    """
    enhancement_instructions = {
        "detailed": "Enhance this prompt with vivid details and clear descriptions that would help an AI image generator create a better image. Keep the core theme intact but add visual details, lighting, mood, and style guidance:",
        "artistic": "Transform this prompt into an artistic concept with specific style references, color palettes, and composition guidance for an AI image generator. Make it suitable for creating a visually striking artwork:",
        "photorealistic": "Enhance this prompt to generate a photorealistic image. Add details about lighting, camera perspective, depth of field, and realistic elements that would make the image appear like a photograph:",
        "cinematic": "Transform this prompt into a cinematic scene description with details about framing, lighting, atmosphere, depth, and visual storytelling that would make it appear like a still from a film:",
        "minimal": "Refine this prompt into a minimal, elegant concept that focuses on simplicity and essential elements while maintaining visual impact:",
        "surreal": "Transform this prompt into a surreal, dreamlike concept with unexpected juxtapositions, symbolic elements, and unique visual ideas that challenge reality:"
    }
    
    instruction = enhancement_instructions.get(
        enhancement_type,
        "Enhance this prompt with more details to create a better image:"
    )
    
    try:
        response = gemini_model.generate_content(
            f"{instruction}\n\n{prompt}",
            generation_config={"max_output_tokens": max_tokens}
        )
        
        if response and hasattr(response, 'text') and response.text:
            # Clean up the response text
            enhanced = str(response.text).strip()
            enhanced = enhanced.replace('</s>', '').replace('<s>', '')
            return enhanced
        else:
            return prompt
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        return prompt

def generate_image_description(
    image: torch.Tensor,
    prompt: str,
    mode: str,
    gemini_model,
    max_tokens: int = 1000
) -> str:
    """
    Generate an image description or transformation using Gemini's vision capabilities.
    
    Args:
        image: Input image tensor
        prompt: Text prompt or instruction
        mode: Processing mode
        gemini_model: Initialized Gemini model
        max_tokens: Maximum tokens in response
        
    Returns:
        str: Generated text description
    """
    # Convert image to base64 for API
    img_base64, mime_type = tensor_to_base64(image)
    
    # Define mode-specific instructions
    mode_instructions = {
        "describe": "Describe this image in vivid detail, focusing on visual elements that would be important for an image generation system:",
        "analyze": "Analyze this image and create a detailed description that captures its composition, color palette, lighting, subject matter, and style:",
        "enhance": "Using this image as inspiration, create an enhanced description that builds upon what's visible but makes it more visually striking. Focus on improving aspects like composition, lighting, detail, and style:",
        "transform": "Transform this image concept into a different style or medium as specified in the prompt while maintaining the core subject matter:",
        "extend": "Describe what might exist beyond the boundaries of this image, extending the scene in all directions with consistent style and content:"
    }
    
    # Get instruction based on mode
    base_instruction = mode_instructions.get(mode, "Describe this image in detail:")
    
    # Combine with user prompt if provided
    instruction = f"{base_instruction}\n\n{prompt}" if prompt else base_instruction
    
    try:
        # Create multimodal prompt with image and text
        response = gemini_model.generate_content(
            [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": img_base64
                    }
                },
                instruction
            ],
            generation_config={"max_output_tokens": max_tokens}
        )
        
        if response and hasattr(response, 'text') and response.text:
            # Clean up the response text
            description = str(response.text).strip()
            description = description.replace('</s>', '').replace('<s>', '')
            return description
        else:
            return "Failed to generate description."
    except Exception as e:
        logger.error(f"Error generating image description: {str(e)}")
        return f"Error: {str(e)}"

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTGeminiImageNode:
    """
    ComfyUI node for generating image descriptions and placeholder images using Google Gemini.
    Features prompt enhancement, image description, and various image generation modes.
    """
    
    CATEGORY = "HommageTools/AI"
    FUNCTION = "process_image"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "placeholder_image", "status")
    
    # Default models list - will be refreshed when needed
    _models = None
    _api_key_status = None
    _last_connectivity_check = 0
    _connectivity_check_interval = 300  # 5 minutes
    
    def __init__(self):
        """Initialize the node and check connectivity."""
        self.check_initial_connectivity()
    
    def check_initial_connectivity(self):
        """Check connectivity and API key validity during initialization."""
        current_time = time.time()
        
        # Only check if haven't checked recently
        if current_time - self._last_connectivity_check < self._connectivity_check_interval:
            return
            
        try:
            api_key = get_api_key()
            success, message = test_api_connectivity(api_key)
            self._api_key_status = message
            self._last_connectivity_check = current_time
            
            if success:
                logger.info(f"HTGeminiImageNode initialized successfully: {message}")
            else:
                logger.warning(f"HTGeminiImageNode initialization warning: {message}")
        except Exception as e:
            self._api_key_status = f"Initialization error: {str(e)}"
            logger.error(f"HTGeminiImageNode initialization error: {str(e)}")
    
    @classmethod
    def get_model_list(cls):
        """Get or refresh the model list with vision-capable models."""
        if not hasattr(cls, '_models') or cls._models is None:
            # Default list focusing on vision-capable models
            cls._models = [
                "gemini-2.5-pro-preview", 
                "gemini-2.0-pro",
                "gemini-1.5-pro",
                "gemini-1.5-flash-preview"
            ]
            try:
                # Try to update from API without blocking startup
                api_models = get_available_models()
                if api_models:
                    cls._models = api_models
            except:
                pass
        return cls._models
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": (cls.get_model_list(), {
                    "default": cls.get_model_list()[0]
                }),
                "mode": (["prompt_enhancement", "image_description", "style_transfer"], {
                    "default": "prompt_enhancement",
                    "description": "Processing mode"
                }),
                "enhancement_type": (["detailed", "artistic", "photorealistic", "cinematic", "minimal", "surreal"], {
                    "default": "detailed",
                    "description": "Type of enhancement to apply to the prompt"
                }),
                "description_mode": (["describe", "analyze", "enhance", "transform", "extend"], {
                    "default": "describe",
                    "description": "How to describe or transform the input image"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt or instructions here..."
                }),
                "output_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 8
                }),
                "output_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 8
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 8192,
                    "step": 100
                })
            },
            "optional": {
                "input_image": ("IMAGE",),
                "refresh_models": ("BOOLEAN", {
                    "default": False
                })
            }
        }

#------------------------------------------------------------------------------
# Section 5: Processing Logic
#------------------------------------------------------------------------------
    def process_image(
        self,
        model: str,
        mode: str,
        enhancement_type: str,
        description_mode: str,
        prompt: str,
        output_width: int,
        output_height: int,
        max_tokens: int,
        input_image: Optional[torch.Tensor] = None,
        refresh_models: bool = False
    ) -> Tuple[str, torch.Tensor, str]:
        """
        Process image or prompt using Gemini API.
        
        Args:
            model: Gemini model to use
            mode: Processing mode
            enhancement_type: Type of prompt enhancement
            description_mode: Type of image description
            prompt: Text prompt or instruction
            output_width: Width of output placeholder image
            output_height: Height of output placeholder image
            max_tokens: Maximum tokens in response
            input_image: Optional input image tensor
            refresh_models: Whether to refresh the model list
            
        Returns:
            Tuple[str, torch.Tensor, str]: Enhanced prompt, placeholder image, and status
        """
        print(f"\nHTGeminiImageNode v{VERSION} - Processing request")
        print(f"Mode: {mode}, Model: {model}")
        
        # Refresh models if requested
        if refresh_models:
            print("Refreshing model list...")
            self.__class__._models = None
            self.__class__.get_model_list()
        
        start_time = time.time()
        
        try:
            # Get API key and initialize Gemini
            api_key = get_api_key()
            if not api_key:
                placeholder = create_default_tensor(output_height, output_width)
                return (prompt, placeholder, "Error: No API key found")
                
            initialize_genai(api_key)
            
            # Create the Gemini model
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7
                }
            )
            
            result_text = ""
            status_msg = ""
            
            # Process based on mode
            if mode == "prompt_enhancement":
                # Enhance the prompt
                print(f"Enhancing prompt with style: {enhancement_type}")
                result_text = enhance_prompt(
                    prompt, 
                    enhancement_type, 
                    gemini_model, 
                    max_tokens
                )
                
                # Create a placeholder image based on the enhanced prompt
                placeholder_image = text_to_tensor(result_text, output_width, output_height)
                status_msg = f"Prompt enhanced with {enhancement_type} style"
                
            elif mode == "image_description" and input_image is not None:
                # Generate description from input image
                print(f"Generating image description with mode: {description_mode}")
                result_text = generate_image_description(
                    input_image,
                    prompt,
                    description_mode,
                    gemini_model,
                    max_tokens
                )
                
                # Pass through the input image with some modifications
                if input_image is not None:
                    # Create a slightly modified version to show it's been processed
                    placeholder_image = input_image.clone()
                    # Add a subtle colored border by adjusting a few pixels around the edge
                    b, h, w, c = placeholder_image.shape
                    border_width = max(3, min(w, h) // 50)  # Proportional border width
                    
                    # Top and bottom borders
                    placeholder_image[0, :border_width, :, 0] = 0.7  # Red tint
                    placeholder_image[0, -border_width:, :, 0] = 0.7
                    # Left and right borders
                    placeholder_image[0, :, :border_width, 1] = 0.7  # Green tint
                    placeholder_image[0, :, -border_width:, 1] = 0.7
                else:
                    placeholder_image = create_default_tensor(output_height, output_width)
                    
                status_msg = f"Image described with {description_mode} mode"
                
            elif mode == "style_transfer" and input_image is not None:
                # Generate style transfer prompt
                print("Generating style transfer prompt")
                result_text = generate_image_description(
                    input_image,
                    prompt,
                    "transform",
                    gemini_model,
                    max_tokens
                )
                
                # Create a placeholder image with the style transfer prompt
                placeholder_image = text_to_tensor(result_text, output_width, output_height)
                status_msg = "Style transfer prompt generated"
                
            else:
                # Default fallback
                if not prompt:
                    prompt = "Generate a detailed, high-quality image"
                result_text = prompt
                placeholder_image = create_default_tensor(output_height, output_width)
                status_msg = "Using default mode with original prompt"
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            status_msg = f"{status_msg} in {elapsed_time:.2f} seconds"
            
            # Verify output tensor format
            if len(placeholder_image.shape) != 4:  # Ensure BHWC
                placeholder_image = placeholder_image.unsqueeze(0)
            
            b, h, w, c = placeholder_image.shape
            print(f"Output tensor: {placeholder_image.shape} (BHWC format)")
            print(f"Status: {status_msg}")
            
            return (result_text, placeholder_image, status_msg)
                
        except Exception as e:
            logger.error(f"Error in Gemini Image processing: {str(e)}")
            
            # Create placeholder on error
            placeholder = create_default_tensor(output_height, output_width)
            
            # Return original prompt and error status
            status_msg = f"Error: {str(e)}"
            return (prompt, placeholder, status_msg)

#------------------------------------------------------------------------------
# Section 6: Additional Utility Methods
#------------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-run the node to ensure fresh results."""
        return float("nan")