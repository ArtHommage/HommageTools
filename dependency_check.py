"""
File: ComfyUI-HommageTools/dependency_check.py
Version: 1.1.0
Description: Comprehensive dependency validation for HommageTools

Sections:
1. Imports and Constants
2. Version Validation Functions
3. Main Dependency Check Function
4. Error Handling
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Constants
#------------------------------------------------------------------------------
import importlib
from typing import Dict, Tuple, List
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

# Define required packages and their minimum versions
REQUIRED_PACKAGES = {
    'torch': {
        'min_version': '2.0.0',
        'reason': 'Required for tensor operations and image processing'
    },
    'numpy': {
        'min_version': '1.22.0',
        'reason': 'Required for array operations and image processing'
    },
    'PIL': {
        'min_version': '9.0.0',
        'reason': 'Required for image processing and format conversions'
    },
    'tifffile': {
        'min_version': '2023.3.15',
        'reason': 'Required for TIFF file export functionality'
    },
    'psd_tools': {
        'min_version': '1.9.24',
        'reason': 'Required for PSD file export functionality'
    }
}

#------------------------------------------------------------------------------
# Section 2: Version Validation Functions
#------------------------------------------------------------------------------
def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse version string into comparable tuple.
    
    Args:
        version_str: Version string (e.g., '1.2.3')
        
    Returns:
        Tuple[int, ...]: Version numbers as tuple
    """
    return tuple(map(int, version_str.split('.')))

def check_version_requirement(
    package_name: str,
    current_version: str,
    required_version: str
) -> bool:
    """
    Check if package version meets minimum requirement.
    
    Args:
        package_name: Name of the package
        current_version: Installed version
        required_version: Minimum required version
        
    Returns:
        bool: True if version requirement is met
    """
    try:
        current = parse_version(current_version)
        required = parse_version(required_version)
        return current >= required
    except ValueError as e:
        logger.error(f"Error parsing version for {package_name}: {str(e)}")
        return False

#------------------------------------------------------------------------------
# Section 3: Main Dependency Check Function
#------------------------------------------------------------------------------
def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are available with correct versions.
    
    Returns:
        Tuple[bool, List[str]]: (success, list of error messages)
    """
    missing_packages = []
    version_issues = []
    
    for package_name, requirements in REQUIRED_PACKAGES.items():
        try:
            # Special case for Pillow
            if package_name == 'PIL':
                module = importlib.import_module('PIL')
                version = module.__version__
            else:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')
            
            # Check version if available
            if version != 'unknown':
                if not check_version_requirement(
                    package_name,
                    version,
                    requirements['min_version']
                ):
                    version_issues.append(
                        f"{package_name}: Installed version {version} is below "
                        f"required version {requirements['min_version']}. "
                        f"Reason: {requirements['reason']}"
                    )
        except ImportError:
            missing_packages.append(
                f"{package_name}: Not installed. "
                f"Reason: {requirements['reason']}"
            )

    # Combine all issues
    all_issues = missing_packages + version_issues
    success = len(all_issues) == 0
    
    # Log results
    if success:
        logger.info("All dependencies validated successfully")
    else:
        for issue in all_issues:
            logger.error(issue)
    
    return success, all_issues

#------------------------------------------------------------------------------
# Section 4: Error Handling
#------------------------------------------------------------------------------
def validate_environment() -> None:
    """
    Validate environment and raise informative error if dependencies missing.
    
    Raises:
        ImportError: If any dependencies are missing or version requirements not met
    """
    success, issues = check_dependencies()
    if not success:
        raise ImportError(
            "HommageTools dependency validation failed:\n" +
            "\n".join(f"- {issue}" for issue in issues)
        )