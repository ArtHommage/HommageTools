"""
Shared utility functions for HommageTools nodes
"""

def get_nearest_divisible_size(size: int, divisor: int) -> int:
    """Calculate the nearest size divisible by the specified divisor."""
    return ((size + divisor - 1) // divisor) * divisor