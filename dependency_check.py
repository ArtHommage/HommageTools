"""
File: ComfyUI-HommageTools/dependency_check.py
Version: 1.0.2
Description: Dependency validation for HommageTools with cleaner warning output
"""

import importlib
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 1: Dependency Definitions
#------------------------------------------------------------------------------
REQUIRED_PACKAGES = {
    'torch': {
        'min_version': '2.0.0',
        'reason': 'Required for tensor operations'
    },
    'numpy': {
        'min_version': '1.22.0',
        'reason': 'Required for array operations'
    },
    'PIL': {
        'min_version': '9.0.0',
        'reason': 'Required for image processing'
    },
    'tifffile': {
        'min_version': '2023.3.15',
        'reason': 'Required for TIFF export'
    },
    'psd_tools': {
        'min_version': '1.9.24',
        'reason': 'Required for PSD export'
    }
}

OPTIONAL_PACKAGES = {
    'oidn': {
        'min_version': '2.0.0',
        'reason': 'Intel denoising features will be disabled'
    }
}

#------------------------------------------------------------------------------
# Section 2: Version Checking
#------------------------------------------------------------------------------
def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string into comparable tuple."""
    try:
        # Handle CUDA version suffix
        version_str = version_str.split('+')[0]
        return tuple(map(int, version_str.split('.')))
    except Exception:
        return (0,)

def check_version(current: str, required: str) -> bool:
    """Check if current version meets minimum requirement."""
    try:
        current_parts = parse_version(current)
        required_parts = parse_version(required)
        return current_parts >= required_parts
    except Exception:
        return False

#------------------------------------------------------------------------------
# Section 3: Dependency Validation
#------------------------------------------------------------------------------
def check_dependencies() -> Tuple[bool, List[str], List[str]]:
    """
    Check all required and optional dependencies.
    
    Returns:
        Tuple[bool, List[str], List[str]]: Success flag, error messages, warning messages
    """
    errors = []
    warnings = []
    
    # Check required packages
    for package, requirements in REQUIRED_PACKAGES.items():
        try:
            if package == 'PIL':
                module = importlib.import_module('PIL')
            else:
                module = importlib.import_module(package)
                
            version = getattr(module, '__version__', 'unknown')
            min_version = requirements['min_version']
            
            if version == 'unknown':
                errors.append(f"Could not determine version for {package}")
                continue
                
            if not check_version(version, min_version):
                errors.append(
                    f"{package} version {version} is below required {min_version}. "
                    f"Reason: {requirements['reason']}"
                )
                
        except ImportError:
            errors.append(
                f"Missing required package: {package}. "
                f"Reason: {requirements['reason']}"
            )
    
    # Check optional packages
    for package, requirements in OPTIONAL_PACKAGES.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            min_version = requirements['min_version']
            
            if version == 'unknown':
                warnings.append(f"{package}: Version check skipped. {requirements['reason']}")
                continue
                
            if not check_version(version, min_version):
                warnings.append(
                    f"{package} version {version} is below recommended {min_version}. "
                    f"{requirements['reason']}"
                )
                
        except ImportError:
            warnings.append(f"{package} not found. {requirements['reason']}")
    
    return len(errors) == 0, errors, warnings

#------------------------------------------------------------------------------
# Section 4: Error Formatting
#------------------------------------------------------------------------------
def format_messages(title: str, messages: List[str]) -> str:
    """Format dependency messages into readable format."""
    if not messages:
        return ""
    formatted = [f"\n{title}"]
    for msg in messages:
        formatted.append(f"  • {msg}")
    return "\n".join(formatted)

#------------------------------------------------------------------------------
# Section 5: Main Validation Function
#------------------------------------------------------------------------------
def validate_environment(silent: bool = False) -> None:
    """
    Validate environment and raise error if required dependencies missing.
    Will log warnings for optional dependencies.
    
    Args:
        silent: If True, suppress warning messages to stdout
        
    Raises:
        ImportError: If any required dependencies are missing or invalid
    """
    success, errors, warnings = check_dependencies()
    
    # Log warnings for optional packages
    if warnings and not silent:
        warning_msg = format_messages("Optional Dependencies:", warnings)
        logger.info(warning_msg)  # Use info level since these are optional
    
    # Raise error if required dependencies fail
    if not success:
        error_msg = format_messages("Required Dependencies Failed:", errors)
        raise ImportError(error_msg)