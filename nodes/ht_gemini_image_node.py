"""
File: homage_tools/nodes/ht_gemini_image_node.py
Version: 1.1.0
Description: Gemini Image Generation node with API support for text-to-image generation
"""

import os
import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import traceback
import folder_paths
import re
from typing import Dict, Any, Tuple, Optional, List, Union
import time

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.1.0"

# Default models for image generation
DEFAULT_MODELS = [
    "gemini-2.0-flash-preview-image-generation",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro"
]

#------------------------------------------------------------------------------
# Section 1: Helper Functions
#------------------------------------------------------------------------------
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
        return api_key
        
    # Then try Colab userdata if available
    try:
        from google.colab import userdata
        api_key = userdata.get('GOOGLE_API_KEY')
        if api_key:
            return api_key
    except:
        pass
    
    # Finally, try loading from file in ComfyUI directory
    try:
        # Get ComfyUI user directory
        user_dir = folder_paths.get_output_directory()
        parent_dir = os.path.dirname(user_dir)
        possible_paths = [
            os.path.join(parent_dir, "GOOGLE.key"),
            os.path.join(user_dir, "GOOGLE.key"),
            os.path.join(os.path.dirname(parent_dir), "GOOGLE.key")
        ]
        
        for key_file_path in possible_paths:
            if os.path.exists(key_file_path):
                with open(key_file_path, 'r') as key_file:
                    api_key = key_file.read().strip()
                    if api_key:
                        return api_key
    except Exception as e:
        logger.error(f"Could not read API key from file: {str(e)}")
    
    raise ValueError("Google API key not found. Set it in environment variables or place it in GOOGLE.key file.")

def base64_to_tensor(base64_str: str) -> torch.Tensor:
    """
    Convert base64 encoded image to tensor.
    
    Args:
        base64_str: Base64 encoded image
        
    Returns:
        torch.Tensor: Image tensor in BHWC format
    """
    try:
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        np_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor with batch dimension (BHWC format)
        tensor = torch.from_numpy(np_array).unsqueeze(0)
        
        return tensor
    except Exception as e:
        logger.error(f"Error converting base64 to tensor: {str(e)}")
        # Return a fallback colored tensor
        fallback = torch.zeros(1, 512, 512, 3)
        # Add a gradient to make it obvious there was an error
        h, w = 512, 512
        for y in range(h):
            for x in range(w):
                fallback[0, y, x, 0] = y / h  # Red gradient vertically
                fallback[0, y, x, 1] = x / w  # Green gradient horizontally
                fallback[0, y, x, 2] = 0.5    # Blue constant
        return fallback

def parse_style_prompt(style: str, prompt: str) -> str:
    """
    Parse style and prompt to create a comprehensive image generation prompt.
    
    Args:
        style: Style for the image
        prompt: Base prompt
        
    Returns:
        str: Combined prompt
    """
    style_descriptions = {
        "photography": "A high-quality, detailed photograph of",
        "photorealistic": "A photorealistic image of",
        "digital art": "A detailed digital art piece of",
        "cartoon": "A colorful cartoon-style illustration of",
        "3d render": "A high-quality 3D rendered image of",
        "watercolor": "A delicate watercolor painting of",
        "oil painting": "An oil painting in the style of a master artist of",
        "pencil sketch": "A detailed pencil sketch of",
        "pixel art": "A retro pixel art image of",
        "fantasy": "A fantasy-style digital painting of",
        "none": ""
    }
    
    style_text = style_descriptions.get(style.lower(), "")
    
    # Combine style and prompt
    if style_text:
        # Check if prompt already starts with style text to avoid duplication
        if prompt.lower().startswith(style_text.lower()):
            return prompt
        return f"{style_text} {prompt}"
    else:
        return prompt

#------------------------------------------------------------------------------
# Section 2: Image Generation Function
#------------------------------------------------------------------------------
def generate_image_with_gemini(
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash-preview-image-generation",
    width: int = 1024,
    height: int = 1024,
    style: str = "none"
) -> Tuple[torch.Tensor, str]:
    """
    Generate an image using the Gemini API.
    
    Args:
        prompt: Text prompt for image generation
        api_key: Google API key
        model_name: Name of the Gemini model to use
        width: Desired image width
        height: Desired image height
        style: Image style
        
    Returns:
        Tuple[torch.Tensor, str]: Generated image tensor and status message
    """
    try:
        # Import here to avoid startup errors if package is missing
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        # Add AspectRatio to the prompt
        aspect_ratio = f"{width}:{height}"
        if not re.search(r'\baspect ratio\b', prompt.lower()):
            prompt += f", aspect ratio {aspect_ratio}"
        
        # Create full prompt with style
        full_prompt = parse_style_prompt(style, prompt)
        print(f"Sending prompt to Gemini: {full_prompt[:100]}...")
        print(f"Using model: {model_name}")
        
        # Configure the client
        genai.configure(api_key=api_key)
        
        # Try direct API request (more control over parameters)
        import requests
        import json
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Set up safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "responseModalities": ["TEXT", "IMAGE"]
            },
            "safetySettings": safety_settings
        }
        
        print("Sending API request with responseModalities: TEXT, IMAGE")
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return torch.zeros(1, height, width, 3), f"API Error: Status code {response.status_code} - {response.text[:200]}"
        
        # Parse the response
        result = response.json()
        
        # Look for image data in the response
        image_tensor = None
        text_response = ""
        
        if "candidates" in result and result["candidates"]:
            parts = result["candidates"][0]["content"]["parts"]
            for part in parts:
                if "text" in part:
                    text_response = part["text"]
                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    if inline_data.get("mimeType", "").startswith("image/"):
                        image_tensor = base64_to_tensor(inline_data["data"])
                        print(f"Found image data ({inline_data['mimeType']})")
        
        if image_tensor is not None:
            return image_tensor, f"Image generated successfully. {text_response}"
        else:
            return torch.zeros(1, height, width, 3), f"No image data in response. Response text: {text_response[:200]}"
                
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return torch.zeros(1, height, width, 3), error_msg
            
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return torch.zeros(1, height, width, 3), error_msg

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTGeminiImageNode:
    """
    ComfyUI node for generating images using Google Gemini API.
    Supports text to image generation with various styles and parameters.
    """
    
    CATEGORY = "HommageTools/AI"
    FUNCTION = "generate_image"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_image", "status")
    
    # Class-level caching for model list
    _cached_models = None
    _models_last_refresh = 0
    _refresh_interval = 3600  # Refresh model list every hour
    
    def __init__(self):
        """Initialize the node"""
        # Ensure models are loaded
        self.get_available_models(force_refresh=False)
    
    @classmethod
    def get_available_models(cls, force_refresh=False):
        """Get or refresh the list of available models"""
        current_time = time.time()
        
        # Check if we need to refresh the models list
        if (cls._cached_models is None or 
            force_refresh or 
            (current_time - cls._models_last_refresh) > cls._refresh_interval):
            
            print("Loading available Gemini models...")
            try:
                # Try to get API key
                api_key = get_api_key()
                
                # Try to get available models from API
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # List models
                models = genai.list_models()
                available_models = []
                
                # Filter for appropriate models
                for model in models:
                    model_name = model.name.split('/')[-1]
                    # Look for models that might support image generation
                    if any(x in model_name.lower() for x in ["gemini", "image", "vision", "flash", "pro"]):
                        available_models.append(model_name)
                
                # If we found models, use them
                if available_models:
                    cls._cached_models = available_models
                    print(f"Found {len(available_models)} models: {', '.join(available_models[:3])}...")
                else:
                    # Otherwise use defaults
                    cls._cached_models = DEFAULT_MODELS
                    print(f"No models found, using defaults: {', '.join(DEFAULT_MODELS)}")
                
            except Exception as e:
                print(f"Error getting models: {str(e)}")
                # Fall back to default models
                cls._cached_models = DEFAULT_MODELS
            
            # Update refresh timestamp
            cls._models_last_refresh = current_time
        
        return cls._cached_models
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        # Available image styles
        image_styles = [
            "none",
            "photography",
            "photorealistic",
            "digital art", 
            "cartoon",
            "3d render",
            "watercolor",
            "oil painting",
            "pencil sketch",
            "pixel art",
            "fantasy"
        ]
        
        # Get cached models or load them if needed
        available_models = cls.get_available_models()
        
        # Put the default model first if it's in the list
        default_model = "gemini-2.0-flash-preview-image-generation"
        if default_model in available_models:
            # Move default to the front
            available_models = [m for m in available_models if m != default_model]
            available_models.insert(0, default_model)
        elif available_models:
            default_model = available_models[0]
        else:
            # Fallback - shouldn't happen since we have DEFAULT_MODELS
            available_models = DEFAULT_MODELS
            default_model = DEFAULT_MODELS[0]
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt for image generation..."
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8
                }),
                "style": (image_styles, {
                    "default": "none"
                }),
                "model": (available_models, {
                    "default": default_model
                }),
                "refresh_models": ("BOOLEAN", {
                    "default": False,
                    "description": "Refresh the available models list"
                })
            }
        }

    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str,
        model: str,
        refresh_models: bool
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate an image using the Gemini API.
        
        Args:
            prompt: Text prompt for image generation
            width: Desired image width
            height: Desired image height
            style: Image style
            model: Model to use for generation
            refresh_models: Whether to refresh the model list
            
        Returns:
            Tuple[torch.Tensor, str]: Generated image tensor and status message
        """
        print(f"\nHTGeminiImageNode v{VERSION} - Processing")
        print(f"Size: {width}x{height}, Style: {style}")
        print(f"Using model: {model}")
        print(f"Prompt: '{prompt[:50]}...'")
        
        # Refresh models if requested
        if refresh_models:
            print("Refreshing model list...")
            self.get_available_models(force_refresh=True)
        
        start_time = time.time()
        
        try:
            # Check if prompt is empty
            if not prompt.strip():
                return torch.zeros(1, height, width, 3), "Error: Empty prompt"
            
            # Get API key
            api_key = get_api_key()
            
            # Generate the image
            image_tensor, status = generate_image_with_gemini(
                prompt,
                api_key,
                model,
                width,
                height,
                style
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            full_status = f"{status} in {elapsed_time:.2f} seconds"
            
            return image_tensor, full_status
            
        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            logger.error(error_msg)
            return torch.zeros(1, height, width, 3), error_msg

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-run the node to ensure fresh results."""
        return float("nan")