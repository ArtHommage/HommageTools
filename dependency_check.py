"""
File: ComfyUI-HommageTools/dependency_check.py
"""

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = {
        'torch': 'PyTorch is required for image processing nodes',
        're': 'Regular expressions module is required',
        'json': 'JSON module is required for file operations',
        'pathlib': 'Pathlib is required for file operations'
    }
    
    missing = []
    import importlib
    
    for package, message in required_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(f"{package}: {message}")
            
    if missing:
        raise ImportError(
            "Missing required dependencies:\n" + 
            "\n".join(missing)
        )
        