"""
File: homage_tools/nodes/ht_gemini_node.py
Version: 1.1.0
Description: Node for interfacing with Google Gemini API with image support and connectivity checks
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
from typing import Dict, Any, Tuple, Optional, List
from io import BytesIO
from PIL import Image
import logging
import google.generativeai as genai
import folder_paths

# Configure logging
logger = logging.getLogger('HommageTools')

# Version tracking
VERSION = "1.1.1"

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
        
        # Filter for Gemini models
        for model in models:
            if "gemini" in model.name:
                model_name = model.name.split('/')[-1]
                gemini_models.append(model_name)
        
        if not gemini_models:
            # Default list of commonly used Gemini models (as of May 2025)
            return [
                # Recommended Stable Models
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                # Key Preview Models
                "gemini-2.5-pro-preview", 
                "gemini-2.5-flash-preview",
                # Older stable models
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                # Specific versions
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-lite-001",
                # Experimental Models
                "gemini-2.0-flash-thinking-exp",
                "gemini-2.0-flash-live-preview"
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
        # Default list of commonly used Gemini models (as of May 2025)
        return [
            # Recommended Stable Models
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            # Key Preview Models
            "gemini-2.5-pro-preview", 
            "gemini-2.5-flash-preview",
            # Older stable models
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            # Specific versions
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            # Experimental Models
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.0-flash-live-preview"
        ]

#------------------------------------------------------------------------------
# Section 3: Node Class Definition
#------------------------------------------------------------------------------
class HTGeminiNode:
    """
    ComfyUI node for interfacing with Google Gemini API.
    Supports text and image inputs with customizable generation parameters.
    """
    
    CATEGORY = "HommageTools/AI"
    FUNCTION = "generate_content"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "status")
    
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
                logger.info(f"HTGeminiNode initialized successfully: {message}")
            else:
                logger.warning(f"HTGeminiNode initialization warning: {message}")
        except Exception as e:
            self._api_key_status = f"Initialization error: {str(e)}"
            logger.error(f"HTGeminiNode initialization error: {str(e)}")
    
    @classmethod
    def get_model_list(cls):
        """Get or refresh the model list."""
        if not hasattr(cls, '_models') or cls._models is None:
            # Default list of commonly used Gemini models (as of May 2025)
            cls._models = [
                # Recommended Stable Models
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                # Key Preview Models
                "gemini-2.5-pro-preview", 
                "gemini-2.5-flash-preview",
                # Older stable models
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
                # Specific versions
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-lite-001",
                # Experimental Models
                "gemini-2.0-flash-thinking-exp",
                "gemini-2.0-flash-live-preview"
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
                "refresh_models": ("BOOLEAN", {
                    "default": False
                }),
                "system_instructions": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "Enter system instructions here..."
                }),
                "content": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt here..."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                })
            },
            "optional": {
                "image": ("IMAGE",),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "response_mime_type": (["text/plain", "text/markdown"], {
                    "default": "text/markdown"
                }),
                "safety_settings": ("STRING", {
                    "multiline": True,
                    "default": '{"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE", "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE", "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"}',
                    "placeholder": "JSON safety settings (advanced)"
                }),
            }
        }

#------------------------------------------------------------------------------
# Section 4: API Interaction Logic
#------------------------------------------------------------------------------
    def generate_content(
        self,
        model: str,
        refresh_models: bool,
        system_instructions: str,
        content: str,
        temperature: float,
        image: Optional[torch.Tensor] = None,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 2048,
        response_mime_type: str = "text/markdown",
        high_quality_image: bool = True,
        safety_settings: str = "{}"
    ) -> Tuple[str, str]:
        """
        Generate content using Google Gemini API.
        
        Args:
            model: Gemini model to use
            refresh_models: Whether to refresh the model list
            system_instructions: System instructions for the model
            content: Text content to process
            temperature: Controls randomness (0.0-1.0)
            image: Optional image tensor (BHWC format)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens in response
            response_mime_type: Desired response format
            high_quality_image: Whether to use high quality images
            safety_settings: JSON string of safety settings
            
        Returns:
            Tuple[str, str]: Generated text and status/error information
        """
        print(f"\nHTGeminiNode v{VERSION} - Processing request")
        
        # Debug API key retrieval
        try:
            api_key_test = get_api_key()
            print(f"API key found successfully")
        except Exception as e:
            print(f"API key retrieval error: {str(e)}")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
        
        # Check API connectivity status
        if hasattr(self, '_api_key_status') and self._api_key_status:
            print(f"API Status: {self._api_key_status}")
        
        # Refresh models if requested
        if refresh_models:
            print("Refreshing model list...")
            self.__class__._models = None
            self.__class__.get_model_list()
        
        start_time = time.time()
        status_msg = ""
        
        # Maximum retry attempts for rate limiting
        max_retries = 3
        retry_count = 0
        retry_delay = 2  # Initial delay in seconds
        
        while retry_count <= max_retries:
            try:
                # Get API key and initialize Gemini
                api_key = get_api_key()
                if not api_key:
                    return (content, "Error: No API key found. In Google Colab, set a secret named 'GOOGLE_API_KEY' with your key.")
                    
                initialize_genai(api_key)
                
                # Parse safety settings if provided
                parsed_safety = {}
                if safety_settings and safety_settings != "{}":
                    try:
                        parsed_safety = json.loads(safety_settings)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid safety settings JSON, using defaults")
                
                # Configure generation parameters
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": response_mime_type
                }
                
                # Create the Gemini model
                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config,
                    safety_settings=parsed_safety,
                    system_instruction=system_instructions if system_instructions else None
                )
                
                # Prepare content parts
                user_parts = []
                
                # Add text content
                if content:
                    user_parts.append(content)
                
                # Add image if provided
                if image is not None:
                    # Check if model supports vision
                    supports_vision = any(vision_indicator in model for vision_indicator in ["vision", "pro-preview", "pro-", "2.0-", "2.5-"])
                    
                    if supports_vision:
                        # Process batch of images if needed
                        if len(image.shape) == 4 and image.shape[0] > 1:
                            print(f"Processing batch of {image.shape[0]} images")
                            # Process up to 16 images maximum
                            max_images = min(16, image.shape[0])
                            
                            for i in range(max_images):
                                single_image = image[i:i+1]  # Keep batch dimension
                                print(f"Processing image {i+1} with shape: {single_image.shape}")
                                img_base64, mime_type = tensor_to_base64(single_image, high_quality_image)
                                user_parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": img_base64
                                    }
                                })
                        else:
                            # Process single image
                            print(f"Processing image with shape: {image.shape}")
                            img_base64, mime_type = tensor_to_base64(image, high_quality_image)
                            user_parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": img_base64
                                }
                            })
                    else:
                        status_msg = f"Warning: Image provided but model {model} may not support vision"
                        print(status_msg)
                
                # Generate response
                print("Sending request to Gemini API...")
                response = gemini_model.generate_content(user_parts)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Process response
                if response and hasattr(response, 'text') and response.text:
                    # Get token usage if available
                    token_info = ""
                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        if hasattr(usage, 'total_token_count'):
                            token_info = f", Tokens: {usage.total_token_count}"
                    
                    # Clean up the response text by removing special tokens
                    clean_results = str(response.text)       
                    clean_results = clean_results.replace('</s>', '')
                    clean_results = clean_results.replace('<s>', '')
                    
                    status_msg = f"Success: Response generated in {elapsed_time:.2f} seconds{token_info}"
                    return (clean_results, status_msg)
                else:
                    return ("", "Empty response received")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in Gemini API request: {error_msg}")
                
                # Check for kernel-related errors
                if "kernel" in error_msg.lower() or "NoneType" in error_msg.lower():
                    return (content, f"Environment error: {error_msg}. The node may not be compatible with this environment. Using original content.")
                
                # Check for quota/availability errors
                quota_errors = [
                    "quota exceeded", 
                    "resource exhausted",
                    "out of capacity",
                    "unavailable",
                    "service unavailable",
                    "not available in your region"
                ]
                
                # Check for rate limit errors
                rate_limit_errors = [
                    "rate limit",
                    "ratelimit", 
                    "too many requests",
                    "request limit exceeded",
                    "try again later"
                ]
                
                if any(quota_term in error_msg.lower() for quota_term in quota_errors):
                    # Quota or availability error - pass through original content
                    return (content, f"API Quota/Availability Error: {error_msg}. Using original content.")
                    
                elif any(rate_term in error_msg.lower() for rate_term in rate_limit_errors):
                    # Rate limit error - retry with backoff if within retry limit
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        print(f"Rate limit hit. Retry {retry_count}/{max_retries} in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Max retries exceeded - pass through original content
                        return (content, f"Rate limit exceeded after {max_retries} retries. Using original content.")
                else:
                    # Other error - pass through content
                    return (content, f"API Error: {error_msg}. Using original content.")
        
        # This should not be reached unless all retries failed
        return (content, "Failed after multiple retry attempts. Using original content.")

#------------------------------------------------------------------------------
# Section 5: Additional Utility Methods
#------------------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-run the node to ensure fresh results."""
        return float("nan")