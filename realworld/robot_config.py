#!/usr/bin/env python3
"""
Robot Configuration
Description: Configuration settings for xArm robot
"""

# xArm Robot IP Address
XARM_IP = "192.168.1.227"

# Safety Settings
DEFAULT_SPEED = 0.35  # Safe production speed per request
MAX_SPEED = 0.35      # Cap at 10 for safety

# Robot Positions (Joint Angles in degrees)
HOME_POSITION = [0, 0, 0, 0, 0, 0, 0]

# Target positions from screenshot analysis
OBJECT_POSITION = [-0.2, 48.8, 0.8, 66.2, 0.6, 17.4, 0.1]
APPROACH_POSITION = [-0.2, 40.0, 0.8, 60.0, 0.6, 15.0, 0.1]  # Slightly higher
PLACE_POSITION = [10.0, 40.0, 0.8, 60.0, 0.6, 15.0, 0.1]     # Rotated base

# Target Cartesian Position (from screenshot)
TARGET_CARTESIAN = {
    'position': [506.3, 4, 19],      # X, Y, Z in mm
    'orientation': [180, 0, 0]        # Roll, Pitch, Yaw in degrees
}

# Gripper Settings
GRIPPER_OPEN_POSITION = 850     # Fully open position
GRIPPER_CLOSE_POSITION = 100    # Closed position for gripping
GRIPPER_SPEED = 5000           # Safe gripper speed

# Timing Settings (in seconds)
MOVEMENT_PAUSE = 1.0    # Pause between movements
GRIPPER_PAUSE = 0.5     # Pause for gripper operations