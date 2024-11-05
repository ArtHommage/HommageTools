"""
File: homage_tools/nodes/ht_file_queue_node.py

HommageTools File Queue Node
Version: 1.0.0
Description: A node that manages a queue of file paths from a directory,
outputting them in controlled batches while maintaining state between sessions.

Sections:
1. Imports and Setup
2. Helper Classes and Functions
3. Node Class Definition and Configuration
4. State Management
5. File System Operations
6. Queue Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Setup
#------------------------------------------------------------------------------
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

#------------------------------------------------------------------------------
# Section 2: Helper Classes and Functions
#------------------------------------------------------------------------------
class QueueState:
    """Helper class to manage queue state persistence."""
    
    def __init__(self, state_dir: str):
        """
        Initialize queue state manager.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "file_queue_state.json"
        self.files_list = self.state_dir / "file_queue_files.txt"
        
    def load_state(self) -> Tuple[int, List[str]]:
        """
        Load current state from files.
        
        Returns:
            Tuple[int, List[str]]: Current index and list of files
        """
        try:
            # Load current index
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    current_index = state.get('current_index', 1)
            else:
                current_index = 1
                
            # Load file list
            files = []
            if self.files_list.exists():
                with open(self.files_list, 'r') as f:
                    files = [line.strip() for line in f if line.strip()]
                    
            return current_index, files
            
        except Exception as e:
            print(f"Error loading queue state: {str(e)}")
            return 1, []
            
    def save_state(self, current_index: int, files: List[str] = None):
        """
        Save current state to files.
        
        Args:
            current_index: Current position in queue
            files: Optional list of files to save
        """
        try:
            # Save current index
            with open(self.state_file, 'w') as f:
                json.dump({'current_index': current_index}, f)
                
            # Save file list if provided
            if files is not None:
                with open(self.files_list, 'w') as f:
                    f.write('\n'.join(files))
                    
        except Exception as e:
            print(f"Error saving queue state: {str(e)}")

#------------------------------------------------------------------------------
# Section 3: Node Class Definition and Configuration
#------------------------------------------------------------------------------
class HTFileQueueNode:
    """
    A node that manages queued access to files in a directory.
    Outputs batches of file paths while maintaining state between sessions.
    """
    
    CATEGORY = "HommageTools"
    FUNCTION = "process_queue"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("file_paths", "current_index", "total_files")
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "description": "Directory containing files to queue"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "description": "Number of files to output per batch (1-100)"
                }),
                "file_extensions": ("STRING", {
                    "default": ".png,.jpg,.jpeg",
                    "multiline": False,
                    "description": "Comma-separated list of file extensions to include"
                }),
                "reset_queue": ("BOOLEAN", {
                    "default": False,
                    "description": "Reset queue to start"
                })
            }
        }
    
    #--------------------------------------------------------------------------
    # Section 4: State Management
    #--------------------------------------------------------------------------
    def __init__(self):
        """Initialize the queue node."""
        self.state = QueueState(os.path.join("configs", "homage_tools"))
        
    #--------------------------------------------------------------------------
    # Section 5: File System Operations
    #--------------------------------------------------------------------------
    def scan_directory(self, directory: str, extensions: List[str]) -> List[str]:
        """
        Scan directory for files with specified extensions.
        
        Args:
            directory: Directory path to scan
            extensions: List of file extensions to include
            
        Returns:
            List[str]: List of matching file paths
        """
        try:
            files = []
            directory_path = Path(directory)
            
            if not directory_path.exists():
                print(f"Directory not found: {directory}")
                return []
                
            # Convert extensions to lowercase for comparison
            extensions = [ext.lower() for ext in extensions]
            
            # Scan directory for matching files
            for file_path in directory_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    files.append(str(file_path.absolute()))
                    
            # Sort for consistent ordering
            return sorted(files)
            
        except Exception as e:
            print(f"Error scanning directory: {str(e)}")
            return []
    
    #--------------------------------------------------------------------------
    # Section 6: Queue Processing Logic
    #--------------------------------------------------------------------------
    def process_queue(
        self,
        directory_path: str,
        batch_size: int,
        file_extensions: str,
        reset_queue: bool
    ) -> Tuple[str, int, int]:
        """
        Process the file queue and output the next batch of files.
        
        Args:
            directory_path: Path to directory containing files
            batch_size: Number of files to output
            file_extensions: Comma-separated list of extensions
            reset_queue: Whether to reset the queue
            
        Returns:
            Tuple[str, int, int]: Newline-separated file paths, current index, total files
        """
        try:
            # Parse extensions
            extensions = [
                ext.strip() if ext.strip().startswith('.')
                else f".{ext.strip()}"
                for ext in file_extensions.split(',')
            ]
            
            # Load current state
            current_index, files = self.state.load_state()
            
            # Reset or initialize if needed
            if reset_queue or not files:
                files = self.scan_directory(directory_path, extensions)
                current_index = 1
                self.state.save_state(current_index, files)
            
            total_files = len(files)
            if total_files == 0:
                return ("", 0, 0)
                
            # Calculate batch
            start_idx = current_index - 1
            end_idx = min(start_idx + batch_size, total_files)
            
            # Get batch of files
            batch_files = files[start_idx:end_idx]
            
            # Update index for next run
            next_index = end_idx + 1
            if next_index > total_files:
                next_index = 1
                
            # Save new state
            self.state.save_state(next_index)
            
            # Return batch
            return (
                "\n".join(batch_files),
                current_index,
                total_files
            )
            
        except Exception as e:
            print(f"Error processing queue: {str(e)}")
            return ("", 0, 0)
            
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Ensure the node updates on every execution.
        
        Returns:
            float: NaN to indicate state should always be checked
        """
        return float("nan")