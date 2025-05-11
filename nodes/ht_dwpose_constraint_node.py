"""
File: homage_tools/nodes/ht_dwpose_constraint_node.py
Version: 1.0.0
Description: Node for applying anatomical constraints to DWPose control images

Sections:
1. Imports and Type Definitions
2. Constants and Configuration
3. Helper Functions
4. Node Class Definition
5. Keypoint Extraction
6. Anatomical Constraints
7. Orientation Detection
8. Joint Correction
9. Processing Logic
"""

#------------------------------------------------------------------------------
# Section 1: Imports and Type Definitions
#------------------------------------------------------------------------------
import torch
import numpy as np
import cv2
import math
from typing import Dict, Any, Tuple, Optional, List, Union
import logging

# Configure logging
logger = logging.getLogger('HommageTools')

#------------------------------------------------------------------------------
# Section 2: Constants and Configuration
#------------------------------------------------------------------------------
VERSION = "1.0.0"

# Define DWPose keypoint color mapping (RGB format)
KEYPOINT_COLORS = {
    # Face keypoints
    "nose": [255, 0, 0],
    "left_eye": [255, 85, 0], 
    "right_eye": [255, 170, 0],
    
    # Upper body keypoints
    "neck": [255, 255, 0],
    "right_shoulder": [170, 255, 0],
    "right_elbow": [85, 255, 0],
    "right_wrist": [0, 255, 0],
    "left_shoulder": [0, 255, 85],
    "left_elbow": [0, 255, 170],
    "left_wrist": [0, 255, 255],
    
    # Torso keypoints
    "right_hip": [0, 170, 255],
    "left_hip": [0, 85, 255],
    
    # Lower body keypoints
    "right_knee": [0, 0, 255],
    "right_ankle": [85, 0, 255],
    "left_knee": [170, 0, 255],
    "left_ankle": [255, 0, 255],
    
    # Feet keypoints
    "right_heel": [255, 0, 170],
    "left_heel": [255, 0, 85],
    "right_foot_index": [255, 0, 0],
    "left_foot_index": [255, 85, 0]
}

# Define joint angle limits (in degrees)
DEFAULT_JOINT_LIMITS = {
    "elbow": {"min": 0, "max": 160},
    "knee": {"min": 0, "max": 150},
    "shoulder": {
        "forward": {"min": -60, "max": 180},
        "sideways": {"min": -30, "max": 180},
        "rotation": {"min": -90, "max": 90}
    },
    "hip": {
        "forward": {"min": -30, "max": 120},
        "sideways": {"min": -50, "max": 45},
        "rotation": {"min": -40, "max": 40}
    },
    "neck": {"min": -45, "max": 45},
    "torso": {"min": -30, "max": 30},
    "ankle": {"min": -45, "max": 45}
}

# Define limb connections for visualization
LIMB_CONNECTIONS = [
    # Upper body
    ("nose", "neck", (255, 255, 0)),
    ("neck", "right_shoulder", (170, 255, 0)),
    ("right_shoulder", "right_elbow", (85, 255, 0)),
    ("right_elbow", "right_wrist", (0, 255, 0)),
    ("neck", "left_shoulder", (0, 255, 85)),
    ("left_shoulder", "left_elbow", (0, 255, 170)),
    ("left_elbow", "left_wrist", (0, 255, 255)),
    
    # Torso
    ("neck", "right_hip", (0, 170, 255)),
    ("neck", "left_hip", (0, 85, 255)),
    ("right_hip", "left_hip", (0, 0, 255)),
    
    # Lower body
    ("right_hip", "right_knee", (85, 0, 255)),
    ("right_knee", "right_ankle", (170, 0, 255)),
    ("left_hip", "left_knee", (255, 0, 255)),
    ("left_knee", "left_ankle", (255, 0, 170)),
    
    # Feet
    ("right_ankle", "right_heel", (255, 0, 85)),
    ("right_heel", "right_foot_index", (255, 0, 0)),
    ("left_ankle", "left_heel", (255, 85, 0)),
    ("left_heel", "left_foot_index", (255, 170, 0))
]

#------------------------------------------------------------------------------
# Section 3: Helper Functions
#------------------------------------------------------------------------------
def verify_tensor_dimensions(tensor: torch.Tensor, context: str) -> Tuple[int, int, int, int]:
    """
    Verify and extract dimensions from BHWC tensor.
    
    Args:
        tensor: Input tensor
        context: Context for logging
        
    Returns:
        Tuple of batch, height, width, channels
    """
    shape = tensor.shape
    print(f"{context} - Tensor shape: {shape}")
    
    if len(shape) == 3:  # HWC format
        height, width, channels = shape
        batch = 1
        print(f"{context} - HWC format detected")
    elif len(shape) == 4:  # BHWC format
        batch, height, width, channels = shape
        print(f"{context} - BHWC format detected")
    else:
        print(f"{context} - ERROR: Invalid tensor shape: {shape}")
        raise ValueError(f"Invalid tensor shape: {shape}")
        
    print(f"{context} - Dimensions: {batch}x{height}x{width}x{channels}")
    return batch, height, width, channels

def calculate_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate angle between two vectors in degrees.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Angle in degrees
    """
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    # Handle potential numerical issues
    cos_angle = np.clip(dot / max(norm, 1e-10), -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle

def rotate_point(pivot: np.ndarray, point: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate a point around a pivot by the given angle.
    
    Args:
        pivot: Center point of rotation
        point: Point to rotate
        angle_degrees: Angle to rotate in degrees
        
    Returns:
        Rotated point
    """
    angle_radians = np.radians(angle_degrees)
    s = np.sin(angle_radians)
    c = np.cos(angle_radians)
    
    # Translate point to origin
    point_centered = point - pivot
    
    # Rotate point
    rotated_x = point_centered[0] * c - point_centered[1] * s
    rotated_y = point_centered[0] * s + point_centered[1] * c
    
    # Translate back
    return np.array([rotated_x, rotated_y]) + pivot

def apply_flexibility(limits: Dict[str, Any], flexibility_factor: float) -> Dict[str, Any]:
    """
    Apply flexibility factor to joint limits.
    
    Args:
        limits: Original joint limits
        flexibility_factor: Flexibility factor (1.0 = normal, <1.0 = stiffer, >1.0 = more flexible)
        
    Returns:
        Modified joint limits
    """
    result = {}
    
    for joint, constraints in limits.items():
        if isinstance(constraints, dict) and "min" in constraints and "max" in constraints:
            # Simple joint with min/max
            mid_point = (constraints["max"] + constraints["min"]) / 2
            range_half = (constraints["max"] - constraints["min"]) / 2
            
            # Apply flexibility
            new_range = range_half * flexibility_factor
            result[joint] = {
                "min": mid_point - new_range,
                "max": mid_point + new_range
            }
        elif isinstance(constraints, dict):
            # Complex joint with multiple constraints
            result[joint] = {}
            for direction, limit in constraints.items():
                if isinstance(limit, dict) and "min" in limit and "max" in limit:
                    mid_point = (limit["max"] + limit["min"]) / 2
                    range_half = (limit["max"] - limit["min"]) / 2
                    
                    # Apply flexibility
                    new_range = range_half * flexibility_factor
                    result[joint][direction] = {
                        "min": mid_point - new_range,
                        "max": mid_point + new_range
                    }
                else:
                    result[joint][direction] = limit
        else:
            result[joint] = constraints
    
    return result

#------------------------------------------------------------------------------
# Section 4: Node Class Definition
#------------------------------------------------------------------------------
class HTDWPoseConstraintNode:
    """
    Node for applying anatomical constraints to DWPose control images.
    Handles joint angle limits, orientation detection, and forced limbs.
    """
    
    CATEGORY = "HommageTools/Pose"
    FUNCTION = "process_pose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("constrained_pose",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "dwpose_image": ("IMAGE",),
                "flexibility_factor": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.5, 
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "description": "Controls joint flexibility (1.0=normal, <1.0=stiffer, >1.0=more flexible)"
                }),
                "constrain_joints": ("BOOLEAN", {
                    "default": True,
                    "description": "Apply anatomical constraints to joints"
                })
            },
            "optional": {
                "force_limbs": ("BOOLEAN", {
                    "default": False,
                    "description": "Force limbs to appear if missing"
                }),
                "force_hands": ("BOOLEAN", {
                    "default": False,
                    "description": "Force hands to appear if missing"
                }),
                "force_feet": ("BOOLEAN", {
                    "default": False,
                    "description": "Force feet to appear if missing"
                }),
                "preserve_height": ("BOOLEAN", {
                    "default": True,
                    "description": "Preserve the original height of detected humans"
                })
            }
        }

#------------------------------------------------------------------------------
# Section 5: Keypoint Extraction
#------------------------------------------------------------------------------
    def extract_keypoints(self, pose_image: np.ndarray) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Extract keypoints from DWPose image based on color detection.
        
        Args:
            pose_image: DWPose control image
            
        Returns:
            Dictionary of keypoint positions
        """
        keypoints = {}
        
        # Find each keypoint by color
        for name, color in KEYPOINT_COLORS.items():
            # Create color thresholds with some tolerance
            lower_bound = np.array([max(0, c-20) for c in color])
            upper_bound = np.array([min(255, c+20) for c in color])
            
            # Create mask
            mask = cv2.inRange(pose_image, lower_bound, upper_bound)
            
            # Find coordinates of the keypoint
            coordinates = np.where(mask > 0)
            if len(coordinates[0]) > 0:
                # Take average position if multiple points detected
                y = int(np.mean(coordinates[0]))
                x = int(np.mean(coordinates[1]))
                keypoints[name] = (x, y)
            else:
                keypoints[name] = None
        
        # Print detected keypoints for debugging
        detected = [k for k, v in keypoints.items() if v is not None]
        print(f"Detected {len(detected)} keypoints: {', '.join(detected)}")
        
        return keypoints

#------------------------------------------------------------------------------
# Section 6: Anatomical Constraints
#------------------------------------------------------------------------------
    def apply_constraints(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        flexibility_factor: float
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Apply anatomical constraints to keypoints.
        
        Args:
            keypoints: Original keypoints
            flexibility_factor: Flexibility factor
            
        Returns:
            Corrected keypoints
        """
        # Apply flexibility factor to joint limits
        joint_limits = apply_flexibility(DEFAULT_JOINT_LIMITS, flexibility_factor)
        
        # Detect orientation to apply appropriate constraints
        orientation = self.detect_orientation(keypoints)
        print(f"Detected orientation: {orientation['facing']} (confidence: {orientation['confidence']:.2f})")
        
        # Start with a copy of the original keypoints
        corrected = keypoints.copy()
        
        # Apply elbow constraints
        corrected = self.constrain_elbow_joints(corrected, joint_limits)
        
        # Apply knee constraints
        corrected = self.constrain_knee_joints(corrected, joint_limits)
        
        # Apply shoulder constraints (depends on orientation)
        corrected = self.constrain_shoulder_joints(corrected, joint_limits, orientation)
        
        # Apply hip-torso rotation constraints
        corrected = self.constrain_hip_torso_rotation(corrected, joint_limits, orientation)
        
        return corrected

#------------------------------------------------------------------------------
# Section 7: Orientation Detection
#------------------------------------------------------------------------------
    def detect_orientation(self, keypoints: Dict[str, Optional[Tuple[int, int]]]) -> Dict[str, Any]:
        """
        Determine body orientation based on visible keypoints.
        
        Args:
            keypoints: Detected keypoints
            
        Returns:
            Orientation information
        """
        orientation = {
            "facing": "unknown",  # forward, backward, left, right
            "confidence": 0.0
        }
        
        # Method 1: Check face and shoulders
        if all(keypoints.get(k) is not None for k in ['left_shoulder', 'right_shoulder', 'nose']):
            left = np.array(keypoints['left_shoulder'])
            right = np.array(keypoints['right_shoulder'])
            nose = np.array(keypoints['nose'])
            
            # Calculate shoulder line vector and nose-to-shoulder-midpoint vector
            shoulder_vector = right - left
            midpoint = (left + right) / 2
            nose_to_mid = nose - midpoint
            
            # Angle between these vectors helps determine facing direction
            angle = np.arccos(np.clip(np.dot(shoulder_vector, nose_to_mid) / 
                             (np.linalg.norm(shoulder_vector) * np.linalg.norm(nose_to_mid) + 1e-10), -1.0, 1.0))
            
            if angle < np.pi/4:  # Nose is near shoulder line
                orientation["facing"] = "left" if nose_to_mid[0] < 0 else "right"
                orientation["confidence"] = 0.8
            else:  # Nose is above/below shoulder line
                orientation["facing"] = "forward" if nose_to_mid[1] < 0 else "backward"
                orientation["confidence"] = 0.9
        
        # Method 2: Check torso shape if orientation still unknown
        if orientation["facing"] == "unknown" and all(keypoints.get(k) is not None 
                                                   for k in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
            # Calculate torso shape to determine orientation
            shoulders_width = np.linalg.norm(
                np.array(keypoints['right_shoulder']) - np.array(keypoints['left_shoulder'])
            )
            hips_width = np.linalg.norm(
                np.array(keypoints['right_hip']) - np.array(keypoints['left_hip'])
            )
            ratio = shoulders_width / max(hips_width, 1)  # Avoid division by zero
            
            if ratio > 1.5:
                orientation["facing"] = "forward"
                orientation["confidence"] = 0.7
            elif ratio < 0.75:
                orientation["facing"] = "backward"
                orientation["confidence"] = 0.7
        
        # Method 3: Check visibility pattern of keypoints if still unknown
        if orientation["facing"] == "unknown":
            # Count visible keypoints on left vs right side
            left_keys = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle']
            right_keys = ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee', 'right_ankle']
            
            left_visible = sum(1 for k in left_keys if keypoints.get(k) is not None)
            right_visible = sum(1 for k in right_keys if keypoints.get(k) is not None)
            
            # If one side has significantly more visible keypoints, the person is likely facing that side
            if left_visible > right_visible * 1.5:
                orientation["facing"] = "left"
                orientation["confidence"] = 0.6
            elif right_visible > left_visible * 1.5:
                orientation["facing"] = "right"
                orientation["confidence"] = 0.6
            else:
                # Default to forward if all else fails
                orientation["facing"] = "forward"
                orientation["confidence"] = 0.3
        
        return orientation

#------------------------------------------------------------------------------
# Section 8: Joint Correction
#------------------------------------------------------------------------------
    def constrain_elbow_joints(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        joint_limits: Dict[str, Any]
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Apply constraints to elbow joints.
        
        Args:
            keypoints: Original keypoints
            joint_limits: Joint limits
            
        Returns:
            Corrected keypoints
        """
        result = keypoints.copy()
        elbow_limits = joint_limits['elbow']
        
        # Constrain right elbow
        if all(result.get(k) is not None for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            shoulder = np.array(result['right_shoulder'])
            elbow = np.array(result['right_elbow'])
            wrist = np.array(result['right_wrist'])
            
            # Calculate vectors
            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            
            # Calculate current angle
            angle = calculate_angle(upper_arm, forearm)
            
            # Check if angle exceeds limits
            if angle < elbow_limits['min'] or angle > elbow_limits['max']:
                # Constrain by moving the wrist
                constrained_angle = max(elbow_limits['min'], min(angle, elbow_limits['max']))
                
                # Calculate correction angle
                correction = constrained_angle - angle
                
                # Apply correction by rotating the wrist
                corrected_wrist = rotate_point(elbow, wrist, correction)
                result['right_wrist'] = (int(corrected_wrist[0]), int(corrected_wrist[1]))
                print(f"Constrained right elbow: {angle:.1f}° → {constrained_angle:.1f}°")
        
        # Constrain left elbow (similar approach)
        if all(result.get(k) is not None for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            shoulder = np.array(result['left_shoulder'])
            elbow = np.array(result['left_elbow'])
            wrist = np.array(result['left_wrist'])
            
            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            
            angle = calculate_angle(upper_arm, forearm)
            
            if angle < elbow_limits['min'] or angle > elbow_limits['max']:
                constrained_angle = max(elbow_limits['min'], min(angle, elbow_limits['max']))
                correction = constrained_angle - angle
                
                corrected_wrist = rotate_point(elbow, wrist, correction)
                result['left_wrist'] = (int(corrected_wrist[0]), int(corrected_wrist[1]))
                print(f"Constrained left elbow: {angle:.1f}° → {constrained_angle:.1f}°")
        
        return result

    def constrain_knee_joints(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        joint_limits: Dict[str, Any]
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Apply constraints to knee joints.
        
        Args:
            keypoints: Original keypoints
            joint_limits: Joint limits
            
        Returns:
            Corrected keypoints
        """
        result = keypoints.copy()
        knee_limits = joint_limits['knee']
        
        # Constrain right knee
        if all(result.get(k) is not None for k in ['right_hip', 'right_knee', 'right_ankle']):
            hip = np.array(result['right_hip'])
            knee = np.array(result['right_knee'])
            ankle = np.array(result['right_ankle'])
            
            # Calculate vectors
            thigh = knee - hip
            shin = ankle - knee
            
            # Calculate current angle
            angle = calculate_angle(thigh, shin)
            
            # Check if angle exceeds limits
            if angle < knee_limits['min'] or angle > knee_limits['max']:
                # Constrain by moving the ankle
                constrained_angle = max(knee_limits['min'], min(angle, knee_limits['max']))
                
                # Calculate correction angle
                correction = constrained_angle - angle
                
                # Apply correction by rotating the ankle
                corrected_ankle = rotate_point(knee, ankle, correction)
                result['right_ankle'] = (int(corrected_ankle[0]), int(corrected_ankle[1]))
                print(f"Constrained right knee: {angle:.1f}° → {constrained_angle:.1f}°")
                
                # Also adjust foot keypoints if they exist
                if result.get('right_heel') is not None:
                    heel = np.array(result['right_heel'])
                    corrected_heel = rotate_point(knee, heel, correction)
                    result['right_heel'] = (int(corrected_heel[0]), int(corrected_heel[1]))
                
                if result.get('right_foot_index') is not None:
                    foot = np.array(result['right_foot_index'])
                    corrected_foot = rotate_point(knee, foot, correction)
                    result['right_foot_index'] = (int(corrected_foot[0]), int(corrected_foot[1]))
        
        # Constrain left knee (similar approach)
        if all(result.get(k) is not None for k in ['left_hip', 'left_knee', 'left_ankle']):
            hip = np.array(result['left_hip'])
            knee = np.array(result['left_knee'])
            ankle = np.array(result['left_ankle'])
            
            thigh = knee - hip
            shin = ankle - knee
            
            angle = calculate_angle(thigh, shin)
            
            if angle < knee_limits['min'] or angle > knee_limits['max']:
                constrained_angle = max(knee_limits['min'], min(angle, knee_limits['max']))
                correction = constrained_angle - angle
                
                corrected_ankle = rotate_point(knee, ankle, correction)
                result['left_ankle'] = (int(corrected_ankle[0]), int(corrected_ankle[1]))
                print(f"Constrained left knee: {angle:.1f}° → {constrained_angle:.1f}°")
                
                # Adjust foot keypoints
                if result.get('left_heel') is not None:
                    heel = np.array(result['left_heel'])
                    corrected_heel = rotate_point(knee, heel, correction)
                    result['left_heel'] = (int(corrected_heel[0]), int(corrected_heel[1]))
                
                if result.get('left_foot_index') is not None:
                    foot = np.array(result['left_foot_index'])
                    corrected_foot = rotate_point(knee, foot, correction)
                    result['left_foot_index'] = (int(corrected_foot[0]), int(corrected_foot[1]))
        
        return result

    def constrain_shoulder_joints(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        joint_limits: Dict[str, Any],
        orientation: Dict[str, Any]
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Apply constraints to shoulder joints based on orientation.
        
        Args:
            keypoints: Original keypoints
            joint_limits: Joint limits
            orientation: Body orientation
            
        Returns:
            Corrected keypoints
        """
        result = keypoints.copy()
        
        # Select limits based on orientation
        shoulder_limits = joint_limits['shoulder']
        facing = orientation['facing']
        
        if facing in ('left', 'right'):
            limits = shoulder_limits['sideways']
        else:  # forward or backward
            limits = shoulder_limits['forward']
        
        # Constrain right shoulder
        if all(result.get(k) is not None for k in ['neck', 'right_shoulder', 'right_elbow']):
            neck = np.array(result['neck'])
            shoulder = np.array(result['right_shoulder'])
            elbow = np.array(result['right_elbow'])
            
            # Calculate vectors
            trunk = shoulder - neck
            upper_arm = elbow - shoulder
            
            # Calculate current angle
            angle = calculate_angle(trunk, upper_arm)
            
            # Adjust angle based on orientation
            if facing == 'right':
                # When facing right, right shoulder is visible from side
                adjusted_angle = angle
            else:
                # Other orientations
                adjusted_angle = angle
            
            # Check if angle exceeds limits
            if adjusted_angle < limits['min'] or adjusted_angle > limits['max']:
                # Constrain by moving the elbow
                constrained_angle = max(limits['min'], min(adjusted_angle, limits['max']))
                
                # Calculate correction angle
                correction = constrained_angle - adjusted_angle
                
                # Apply correction by rotating the elbow and arm
                corrected_elbow = rotate_point(shoulder, elbow, correction)
                result['right_elbow'] = (int(corrected_elbow[0]), int(corrected_elbow[1]))
                
                # If wrist exists, rotate it too
                if result.get('right_wrist') is not None:
                    wrist = np.array(result['right_wrist'])
                    corrected_wrist = rotate_point(shoulder, wrist, correction)
                    result['right_wrist'] = (int(corrected_wrist[0]), int(corrected_wrist[1]))
                
                print(f"Constrained right shoulder: {adjusted_angle:.1f}° → {constrained_angle:.1f}°")
        
        # Constrain left shoulder (similar approach)
        if all(result.get(k) is not None for k in ['neck', 'left_shoulder', 'left_elbow']):
            neck = np.array(result['neck'])
            shoulder = np.array(result['left_shoulder'])
            elbow = np.array(result['left_elbow'])
            
            trunk = shoulder - neck
            upper_arm = elbow - shoulder
            
            angle = calculate_angle(trunk, upper_arm)
            
            # Adjust angle based on orientation
            if facing == 'left':
                # When facing left, left shoulder is visible from side
                adjusted_angle = angle
            else:
                # Other orientations
                adjusted_angle = angle
            
            if adjusted_angle < limits['min'] or adjusted_angle > limits['max']:
                constrained_angle = max(limits['min'], min(adjusted_angle, limits['max']))
                correction = constrained_angle - adjusted_angle
                
                corrected_elbow = rotate_point(shoulder, elbow, correction)
                result['left_elbow'] = (int(corrected_elbow[0]), int(corrected_elbow[1]))
                
                if result.get('left_wrist') is not None:
                    wrist = np.array(result['left_wrist'])
                    corrected_wrist = rotate_point(shoulder, wrist, correction)
                    result['left_wrist'] = (int(corrected_wrist[0]), int(corrected_wrist[1]))
                
                print(f"Constrained left shoulder: {adjusted_angle:.1f}° → {constrained_angle:.1f}°")
        
        return result

    def constrain_hip_torso_rotation(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        joint_limits: Dict[str, Any],
        orientation: Dict[str, Any]
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Constrain rotation between hips and torso based on orientation.
        
        Args:
            keypoints: Original keypoints
            joint_limits: Joint limits
            orientation: Body orientation
            
        Returns:
            Corrected keypoints
        """
        result = keypoints.copy()
        
        # Check if we have all required keypoints
        if not all(keypoints.get(k) is not None for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            return result
        
        # Calculate torso and hip angles
        shoulders = (np.array(keypoints['right_shoulder']) - np.array(keypoints['left_shoulder']))
        hips = (np.array(keypoints['right_hip']) - np.array(keypoints['left_hip']))
        
        shoulder_angle = np.arctan2(shoulders[1], shoulders[0])
        hip_angle = np.arctan2(hips[1], hips[0])
        
        # Calculate relative rotation between hips and shoulders (in degrees)
        rotation_angle = (hip_angle - shoulder_angle) * 180 / np.pi
        
        # Normalize to range [-180, 180]
        if rotation_angle > 180:
            rotation_angle -= 360
        elif rotation_angle < -180:
            rotation_angle += 360
        
        # Define rotation limits based on orientation
        facing = orientation['facing']
        
        # Set limits based on facing direction
        rotation_limits = {
            "forward": {"min": -30, "max": 30},
            "backward": {"min": -30, "max": 30},
            "left": {"min": -60, "max": 10},
            "right": {"min": -10, "max": 60},
            "unknown": {"min": -45, "max": 45}
        }
        
        limits = rotation_limits.get(facing, rotation_limits["unknown"])
        
        # Check if rotation exceeds limits
        if rotation_angle < limits['min'] or rotation_angle > limits['max']:
            # Constrain rotation
            constrained_angle = max(limits['min'], min(rotation_angle, limits['max']))
            
            # Calculate correction angle
            correction = constrained_angle - rotation_angle
            print(f"Constraining torso rotation: {rotation_angle:.1f}° → {constrained_angle:.1f}°")
            
            # Calculate midpoint between shoulders as rotation pivot
            left_shoulder = np.array(result['left_shoulder'])
            right_shoulder = np.array(result['right_shoulder'])
            pivot = (left_shoulder + right_shoulder) / 2
            
            # Rotate hip keypoints around this pivot
            for key in ['left_hip', 'right_hip']:
                point = np.array(result[key])
                rotated = rotate_point(pivot, point, correction)
                result[key] = (int(rotated[0]), int(rotated[1]))
                
                # Also adjust dependent limbs (legs)
                hip_point = np.array(result[key])
                
                # Adjust knee if it exists
                knee_key = f"{key.split('_')[0]}_knee"
                if result.get(knee_key) is not None:
                    knee = np.array(result[knee_key])
                    # Calculate relative position and rotate
                    knee_rotated = rotate_point(hip_point, knee, 0)  # No additional rotation
                    result[knee_key] = (int(knee_rotated[0]), int(knee_rotated[1]))
                    
                    # Adjust ankle if it exists
                    ankle_key = f"{key.split('_')[0]}_ankle"
                    if result.get(ankle_key) is not None:
                        ankle = np.array(result[ankle_key])
                        # Calculate relative position and rotate
                        ankle_rotated = rotate_point(knee_rotated, ankle, 0)  # No additional rotation
                        result[ankle_key] = (int(ankle_rotated[0]), int(ankle_rotated[1]))
                        
                        # Adjust foot keypoints
                        heel_key = f"{key.split('_')[0]}_heel"
                        if result.get(heel_key) is not None:
                            heel = np.array(result[heel_key])
                            heel_rotated = rotate_point(ankle_rotated, heel, 0)
                            result[heel_key] = (int(heel_rotated[0]), int(heel_rotated[1]))
                        
                        foot_key = f"{key.split('_')[0]}_foot_index"
                        if result.get(foot_key) is not None:
                            foot = np.array(result[foot_key])
                            foot_rotated = rotate_point(ankle_rotated, foot, 0)
                            result[foot_key] = (int(foot_rotated[0]), int(foot_rotated[1]))
        
        return result

    def force_body_parts(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        force_limbs: bool, 
        force_hands: bool, 
        force_feet: bool
    ) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Estimate positions for missing body parts.
        
        Args:
            keypoints: Original keypoints
            force_limbs: Whether to force missing limbs
            force_hands: Whether to force missing hands
            force_feet: Whether to force missing feet
            
        Returns:
            Updated keypoints
        """
        result = keypoints.copy()
        
        # Force limbs if needed
        if force_limbs:
            # Right arm
            if result.get('right_shoulder') is not None and result.get('right_elbow') is None:
                shoulder = np.array(result['right_shoulder'])
                # Estimate elbow position (simplified)
                elbow_x = shoulder[0] + 40  # Move right
                elbow_y = shoulder[1] + 20  # Move down
                result['right_elbow'] = (int(elbow_x), int(elbow_y))
                print(f"Forced right elbow at position ({elbow_x}, {elbow_y})")
            
            # Left arm
            if result.get('left_shoulder') is not None and result.get('left_elbow') is None:
                shoulder = np.array(result['left_shoulder'])
                # Estimate elbow position (simplified)
                elbow_x = shoulder[0] - 40  # Move left
                elbow_y = shoulder[1] + 20  # Move down
                result['left_elbow'] = (int(elbow_x), int(elbow_y))
                print(f"Forced left elbow at position ({elbow_x}, {elbow_y})")
            
            # Right leg
            if result.get('right_hip') is not None and result.get('right_knee') is None:
                hip = np.array(result['right_hip'])
                # Estimate knee position
                knee_x = hip[0] + 15  # Move slightly right
                knee_y = hip[1] + 60  # Move down
                result['right_knee'] = (int(knee_x), int(knee_y))
                print(f"Forced right knee at position ({knee_x}, {knee_y})")
            
            # Left leg
            if result.get('left_hip') is not None and result.get('left_knee') is None:
                hip = np.array(result['left_hip'])
                # Estimate knee position
                knee_x = hip[0] - 15  # Move slightly left
                knee_y = hip[1] + 60  # Move down
                result['left_knee'] = (int(knee_x), int(knee_y))
                print(f"Forced left knee at position ({knee_x}, {knee_y})")
        
        # Force hands if needed
        if force_hands:
            # Right wrist
            if result.get('right_elbow') is not None and result.get('right_wrist') is None:
                elbow = np.array(result['right_elbow'])
                # Estimate wrist position
                wrist_x = elbow[0] + 40  # Move right
                wrist_y = elbow[1] + 10  # Move slightly down
                result['right_wrist'] = (int(wrist_x), int(wrist_y))
                print(f"Forced right wrist at position ({wrist_x}, {wrist_y})")
            
            # Left wrist
            if result.get('left_elbow') is not None and result.get('left_wrist') is None:
                elbow = np.array(result['left_elbow'])
                # Estimate wrist position
                wrist_x = elbow[0] - 40  # Move left
                wrist_y = elbow[1] + 10  # Move slightly down
                result['left_wrist'] = (int(wrist_x), int(wrist_y))
                print(f"Forced left wrist at position ({wrist_x}, {wrist_y})")
        
        # Force feet if needed
        if force_feet:
            # Right ankle
            if result.get('right_knee') is not None and result.get('right_ankle') is None:
                knee = np.array(result['right_knee'])
                # Estimate ankle position
                ankle_x = knee[0] + 10  # Move slightly right
                ankle_y = knee[1] + 60  # Move down
                result['right_ankle'] = (int(ankle_x), int(ankle_y))
                print(f"Forced right ankle at position ({ankle_x}, {ankle_y})")
                
                # Add heel and foot index
                if result.get('right_heel') is None:
                    heel_x = ankle_x - 5
                    heel_y = ankle_y + 15
                    result['right_heel'] = (int(heel_x), int(heel_y))
                
                if result.get('right_foot_index') is None:
                    foot_x = ankle_x + 25
                    foot_y = ankle_y + 5
                    result['right_foot_index'] = (int(foot_x), int(foot_y))
            
            # Left ankle
            if result.get('left_knee') is not None and result.get('left_ankle') is None:
                knee = np.array(result['left_knee'])
                # Estimate ankle position
                ankle_x = knee[0] - 10  # Move slightly left
                ankle_y = knee[1] + 60  # Move down
                result['left_ankle'] = (int(ankle_x), int(ankle_y))
                print(f"Forced left ankle at position ({ankle_x}, {ankle_y})")
                
                # Add heel and foot index
                if result.get('left_heel') is None:
                    heel_x = ankle_x + 5
                    heel_y = ankle_y + 15
                    result['left_heel'] = (int(heel_x), int(heel_y))
                
                if result.get('left_foot_index') is None:
                    foot_x = ankle_x - 25
                    foot_y = ankle_y + 5
                    result['left_foot_index'] = (int(foot_x), int(foot_y))
        
        return result

    def generate_pose_image(
        self, 
        keypoints: Dict[str, Optional[Tuple[int, int]]], 
        original_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Generate a pose image from keypoints.
        
        Args:
            keypoints: Keypoint positions
            original_shape: Shape of original image
            
        Returns:
            DWPose control image
        """
        # Create a blank canvas with same shape as input
        new_pose = np.zeros(original_shape, dtype=np.uint8)
        
        # Draw limbs
        for start, end, color in LIMB_CONNECTIONS:
            if keypoints.get(start) and keypoints.get(end):
                cv2.line(new_pose, 
                        keypoints[start], 
                        keypoints[end], 
                        color, 
                        thickness=6)
        
        # Draw keypoints
        for key, position in keypoints.items():
            if position:
                # Use color from mapping if available
                color = KEYPOINT_COLORS.get(key, (255, 255, 255))
                cv2.circle(new_pose, position, 6, color, -1)
        
        return new_pose

#------------------------------------------------------------------------------
# Section 9: Processing Logic
#------------------------------------------------------------------------------
    def process_pose(
        self,
        dwpose_image: torch.Tensor,
        flexibility_factor: float = 1.0,
        constrain_joints: bool = True,
        force_limbs: bool = False,
        force_hands: bool = False,
        force_feet: bool = False,
        preserve_height: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        Apply anatomical constraints to DWPose control image.
        
        Args:
            dwpose_image: DWPose control image as tensor
            flexibility_factor: Joint flexibility factor
            constrain_joints: Whether to apply anatomical constraints
            force_limbs: Whether to force limbs to appear if missing
            force_hands: Whether to force hands to appear if missing
            force_feet: Whether to force feet to appear if missing
            preserve_height: Whether to preserve original height
            
        Returns:
            Tuple containing constrained pose image
        """
        print(f"\nHTDWPoseConstraintNode v{VERSION} - Processing")
        
        try:
            # Verify dimensions
            batch, height, width, channels = verify_tensor_dimensions(dwpose_image, "Input pose")
            
            # Convert to numpy for processing
            image_np = (255.0 * dwpose_image.cpu().numpy()).astype(np.uint8)
            
            # Process each image in batch
            result_batch = []
            
            for b in range(batch):
                # Get single image from batch
                single_img = image_np[b]
                print(f"\nProcessing image {b+1}/{batch}")
                
                # Extract keypoints from the pose image
                keypoints = self.extract_keypoints(single_img)
                
                # Apply constraints if requested
                if constrain_joints:
                    keypoints = self.apply_constraints(keypoints, flexibility_factor)
                
                # Force limbs/extremities if requested
                if force_limbs or force_hands or force_feet:
                    keypoints = self.force_body_parts(
                        keypoints, 
                        force_limbs, 
                        force_hands, 
                        force_feet
                    )
                
                # Generate new pose image
                new_pose_image = self.generate_pose_image(keypoints, single_img.shape)
                result_batch.append(new_pose_image)
            
            # Convert batch back to tensor
            result_tensor = torch.from_numpy(np.stack(result_batch).astype(np.float32) / 255.0)
            
            return (result_tensor,)
            
        except Exception as e:
            logger.error(f"Error in DWPose constraint processing: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return original image on error
            return (dwpose_image,)