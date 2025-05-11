"""
File: homage_tools/nodes/ht_inspector_node.py
Version: 1.0.1
Description: Node for inspecting and reporting input types and values for debugging
"""

import torch
import numpy as np
import sys
import datetime
from typing import Tuple, Any, Dict

#------------------------------------------------------------------------------
# Section 1: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.0.1"

#------------------------------------------------------------------------------
# Section 2: Node Class Definition
#------------------------------------------------------------------------------
class HTInspectorNode:
    """
    Inspector node that provides detailed information about data passing through it.
    Inspects tensor shapes, data types, value ranges, and other properties.
    """
    
    CATEGORY = "HommageTools/Debug"
    FUNCTION = "inspect"
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("data", "info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "data": ("*", {"description": "Any data to inspect"})
            }
        }

    #--------------------------------------------------------------------------
    # Section 3: Data Inspection Logic
    #--------------------------------------------------------------------------
    def inspect(self, data: Any) -> Tuple[Any, str]:
        """
        Inspect provided data and return detailed information.
        
        Args:
            data: Any data type to inspect
            
        Returns:
            Tuple[Any, str]: Original data and info string
        """
        info = []
        
        # Add timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info.append(f"Inspection Time: {current_time}")
        
        info.append("=== HT INSPECTOR DATA ANALYSIS ===")
        
        # Basic data type information
        info.append(f"Type: {type(data).__name__}")
        
        # Handle different data types
        if isinstance(data, torch.Tensor):
            self._inspect_tensor(data, info)
        elif isinstance(data, list):
            self._inspect_list(data, info)
        elif isinstance(data, dict):
            self._inspect_dict(data, info)
        elif isinstance(data, tuple):
            self._inspect_tuple(data, info)
        elif isinstance(data, np.ndarray):
            self._inspect_numpy(data, info)
        elif isinstance(data, str):
            self._inspect_string(data, info)
        else:
            self._inspect_generic(data, info)
        
        info.append("=== END OF INSPECTION ===")
        
        # Memory usage of the object (approximate)
        try:
            size_bytes = sys.getsizeof(data)
            info.append(f"Object memory size: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")
        except Exception as e:
            info.append(f"Could not determine object size: {str(e)}")
            
        info_str = "\n".join(info)

        # Get the execution context
        context = getattr(self, "context", None)
        
        # Only print to console if the info output is not connected
        if context is None or not context.get_output_connection(1):
            print("\n" + "=" * 50)
            print("HT INSPECTOR OUTPUT")
            print("=" * 50)
            print(info_str)
            print("=" * 50 + "\n")
        
        return (data, info_str)

    #--------------------------------------------------------------------------
    # Section 4: Type-Specific Inspection Methods
    #--------------------------------------------------------------------------
    def _inspect_tensor(self, data: torch.Tensor, info: list) -> None:
        """Inspect torch.Tensor objects."""
        info.append(f"Shape: {data.shape}")
        info.append(f"Dtype: {data.dtype}")
        info.append(f"Device: {data.device}")
        info.append(f"Memory size: {data.element_size() * data.nelement()} bytes")
        info.append(f"Requires grad: {data.requires_grad}")
        
        # Check for BHWC format
        if len(data.shape) == 4:
            b, h, w, c = data.shape
            info.append(f"BHWC format detected: Batch={b}, Height={h}, Width={w}, Channels={c}")
        elif len(data.shape) == 3 and data.shape[2] in [1, 3, 4]:
            h, w, c = data.shape
            info.append(f"HWC format detected: Height={h}, Width={w}, Channels={c}")
        
        # Value ranges and stats
        if data.numel() > 0:
            try:
                info.append(f"Min value: {data.min().item():.6f}")
                info.append(f"Max value: {data.max().item():.6f}")
                info.append(f"Mean value: {data.mean().item():.6f}")
                info.append(f"Standard deviation: {data.std().item():.6f}")
            except Exception as e:
                info.append(f"Could not calculate statistics: {str(e)}")
        
        # Check for invalid values
        try:
            has_nan = torch.isnan(data).any().item()
            has_inf = torch.isinf(data).any().item()
            if has_nan or has_inf:
                info.append(f"WARNING: Tensor contains {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Infinity' if has_inf else ''} values")
        except Exception as e:
            info.append(f"Could not check for NaN/Inf: {str(e)}")
            
        # Sample values
        try:
            sample_values = data.flatten()[:5].tolist()
            info.append(f"Sample values (first 5): {sample_values}")
        except Exception as e:
            info.append(f"Could not show sample values: {str(e)}")

    def _inspect_list(self, data: list, info: list) -> None:
        """Inspect list objects."""
        info.append(f"Length: {len(data)}")
        
        if len(data) > 0:
            info.append(f"First element type: {type(data[0]).__name__}")
            
            # Show a few items if the list isn't too long
            if len(data) <= 5:
                for i, item in enumerate(data):
                    info.append(f"Item {i}: {str(item)[:100]}{'...' if len(str(item)) > 100 else ''}")
            else:
                for i in range(2):
                    info.append(f"Item {i}: {str(data[i])[:100]}{'...' if len(str(data[i])) > 100 else ''}")
                info.append("...")
                for i in range(-2, 0):
                    info.append(f"Item {len(data)+i}: {str(data[i])[:100]}{'...' if len(str(data[i])) > 100 else ''}")

    def _inspect_dict(self, data: dict, info: list) -> None:
        """Inspect dictionary objects."""
        info.append(f"Number of keys: {len(data)}")
        
        if len(data) > 0:
            info.append(f"Keys: {list(data.keys())[:10]}{'...' if len(data) > 10 else ''}")
            
            # Show key-value pairs (limited number)
            for i, (k, v) in enumerate(list(data.items())[:5]):
                info.append(f"Key '{k}': {type(v).__name__} = {str(v)[:50]}{'...' if len(str(v)) > 50 else ''}")
                if i >= 4 and len(data) > 5:
                    info.append(f"... and {len(data) - 5} more items")
                    break

    def _inspect_tuple(self, data: tuple, info: list) -> None:
        """Inspect tuple objects."""
        info.append(f"Length: {len(data)}")
        
        if len(data) > 0:
            if len(data) <= 5:
                for i, item in enumerate(data):
                    info.append(f"Item {i}: {type(item).__name__} = {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}")
            else:
                for i in range(3):
                    info.append(f"Item {i}: {type(data[i]).__name__} = {str(data[i])[:50]}{'...' if len(str(data[i])) > 50 else ''}")
                info.append(f"... and {len(data) - 3} more items")

    def _inspect_numpy(self, data: np.ndarray, info: list) -> None:
        """Inspect numpy array objects."""
        info.append(f"Shape: {data.shape}")
        info.append(f"Dtype: {data.dtype}")
        info.append(f"Size: {data.size} elements")
        info.append(f"Memory size: {data.nbytes} bytes")
        
        if data.size > 0:
            try:
                info.append(f"Min value: {data.min()}")
                info.append(f"Max value: {data.max()}")
                info.append(f"Mean value: {data.mean()}")
                info.append(f"Standard deviation: {data.std()}")
            except Exception as e:
                info.append(f"Could not calculate statistics: {str(e)}")

    def _inspect_string(self, data: str, info: list) -> None:
        """Inspect string objects."""
        info.append(f"Length: {len(data)} characters")
        if len(data) > 0:
            # Truncate long strings
            preview = data[:100]
            info.append(f"Content: {preview}{'...' if len(data) > 100 else ''}")
            
            # Count lines
            line_count = data.count('\n') + 1
            info.append(f"Lines: {line_count}")

    def _inspect_generic(self, data: Any, info: list) -> None:
        """Inspect other object types."""
        info.append("Attributes:")
        
        # Get attributes that don't start with underscore
        public_attrs = [attr for attr in dir(data) if not attr.startswith("_")]
        
        if public_attrs:
            for attr in public_attrs[:10]:
                try:
                    value = getattr(data, attr)
                    if not callable(value):
                        info.append(f"  {attr}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
                except Exception as e:
                    info.append(f"  {attr}: <Error accessing: {str(e)}>")
                    
            if len(public_attrs) > 10:
                info.append(f"  ... and {len(public_attrs) - 10} more attributes")
        else:
            info.append("  No public attributes found")

    #--------------------------------------------------------------------------
    # Section 5: Execution Context and Change Detection
    #--------------------------------------------------------------------------
    def execution_start(self, context):
        """Store the execution context when the node is executed."""
        self.context = context

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Ensure node updates on every execution.
        Returns a unique timestamp value each time to ensure the node always runs.
        """
        return datetime.datetime.now().timestamp()