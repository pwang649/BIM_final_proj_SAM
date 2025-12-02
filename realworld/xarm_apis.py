#!/usr/bin/env python3
"""
xArm Robot API Functions
"""

import os
import sys
import time
import math
import numpy as np

# Add xArm SDK to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'xArm-Python-SDK'))

from xarm.wrapper import XArmAPI


def safe_robot_init(arm):
    """Initialize robot with safety checks"""
    print("Initializing robot with safety protocols...")

    # Enable motion
    ret = arm.motion_enable(enable=True)
    if ret != 0:
        print(f"Failed to enable motion: {ret}")
        return False

    # Set position mode
    ret = arm.set_mode(0)  # Position mode
    if ret != 0:
        print(f"Failed to set mode: {ret}")
        return False

    # Set normal state
    ret = arm.set_state(state=0)  # Normal state
    if ret != 0:
        print(f"Failed to set state: {ret}")
        return False

    print("Robot initialized successfully")
    return True


def move_to_home_position(arm, speed=5):
    """Move robot to home position safely"""
    print("Moving to home position...")
    ret = arm.move_gohome(wait=True, speed=speed)
    if ret != 0:
        print(f"Failed to move to home: {ret}")
        return False
    print("Reached home position")
    return True


def move_to_approach_position(arm, speed=5):
    """Move to approach position (slightly above target object)"""
    print("Moving to approach position...")

    # Approach joint angles (slightly higher than target)
    approach_joints = [-0.2, 40.0, 0.8, 60.0, 0.6, 15.0, 0.1]

    ret = arm.set_servo_angle(angle=approach_joints, speed=speed, wait=True)
    if ret != 0:
        print(f"Failed to move to approach position: {ret}")
        return False

    print("Reached approach position")
    return True


def move_down_to_object(arm, speed=5):
    """Move down slightly from approach position to object"""
    print("Moving down to object...")

    # Object position (slightly lower than approach)
    object_joints = [-0.2, 56.1, 0.8, 72.6, 0.6, 16.6, 0.1]

    ret = arm.set_servo_angle(angle=object_joints, speed=speed, wait=True)
    if ret != 0:
        print(f"Failed to move down to object: {ret}")
        return False

    print("Reached object position")
    return True


def move_to_drop_position(arm, speed=5):
    """Move left to drop position above box"""
    print("Moving to drop position...")

    # Drop position (rotated left from approach position)
    drop_joints = [30.0, 40.0, 0.8, 60.0, 0.6, 15.0, 0.1]  # More rotation left

    ret = arm.set_servo_angle(angle=drop_joints, speed=speed, wait=True)
    if ret != 0:
        print(f"Failed to move to drop position: {ret}")
        return False

    print("Reached drop position")
    return True


def return_to_home_position(arm, speed=5):
    """Return robot to home position safely"""
    print("Returning to home position...")
    ret = arm.move_gohome(wait=True, speed=speed)
    if ret != 0:
        print(f"Failed to return to home: {ret}")
        return False
    print("Returned to home position")
    return True


def initialize_gripper(arm):
    """Initialize standard gripper"""
    print("Initializing standard gripper...")

    ret = arm.set_gripper_mode(0)  # location mode
    if ret != 0:
        print(f"Failed to set gripper mode: {ret}")
        return False

    ret = arm.set_gripper_enable(True)
    if ret != 0:
        print(f"Failed to enable gripper: {ret}")
        return False

    ret = arm.set_gripper_speed(5000)
    if ret != 0:
        print(f"Failed to set gripper speed: {ret}")
        return False

    print("Standard gripper initialized successfully")
    return True


def open_gripper(arm, position=600):
    """Open standard gripper to specified position"""
    print("Opening gripper...")

    code = arm.set_gripper_position(position, wait=True)
    if code != 0:
        print(f"Failed to open gripper: code={code}")
        return False

    print(f"Gripper opened to position {position}: code={code}")
    return True


def close_gripper(arm, position=0):
    """Close standard gripper to specified position (0 = fully closed)"""
    print("Closing gripper...")

    code = arm.set_gripper_position(position, wait=True)
    if code != 0:
        print(f"Failed to close gripper: code={code}")
        return False

    print(f"Gripper closed to position {position}: code={code}")
    return True


def partially_close_gripper(arm, position=200):
    """Partially close gripper for gripping object"""
    print("Partially closing gripper...")

    code = arm.set_gripper_position(position, wait=True)
    if code != 0:
        print(f"Failed to partially close gripper: code={code}")
        return False

    print(f"Gripper partially closed to position {position}: code={code}")
    return True


def create_xarm_connection(ip_address):
    """Create and return XArmAPI connection"""
    return XArmAPI(ip_address)


def disconnect_robot(arm):
    """Safely disconnect from robot"""
    print("Disconnecting from robot...")
    arm.disconnect()


def move_to_observation_pose(arm, speed=5):
    """Move robot to observation pose for camera capture"""
    print("Moving to observation pose...")

    # Conservative observation position - slightly above and back from workspace
    obs_joints = [0.0, -30.0, 0.0, 45.0, 0.0, 75.0, 0.0]

    ret = arm.set_servo_angle(angle=obs_joints, speed=speed, wait=True)
    if ret != 0:
        print(f"Failed to move to observation pose: {ret}")
        return False

    print("Reached observation pose")
    return True


def move_cartesian_delta(arm, dx=0.0, dy=0.0, dz=0.0, speed=5):
    """Move end-effector by small Cartesian delta using relative move if available."""
    try:
        ret = arm.set_position(x=dx, y=dy, z=dz, relative=True, speed=speed, wait=True)
        if ret != 0:
            print(f"Failed to move Cartesian delta: {ret}")
            return False
        return True
    except Exception as e:
        print(f"Cartesian delta move failed: {e}")
        return False


def move_joint_angles(arm, joints, speed=5):
    """Wrapper to move to absolute joint angles list."""
    ret = arm.set_servo_angle(angle=joints, speed=speed, wait=True)
    if ret != 0:
        print(f"Failed to move joints: {ret}")
        return False
    return True


def open_then_close_gripper(arm, open_pos=600, close_pos=175):
    """Utility to open then partially close gripper."""
    if not open_gripper(arm, position=open_pos):
        return False
    time.sleep(0.3)
    if not partially_close_gripper(arm, position=close_pos):
        return False
    return True


def get_eef_pose(arm, radians: bool = True):
    """Get current end-effector pose (x, y, z, roll, pitch, yaw).
    Returns position in meters and angles in radians if radians=True.
    """
    code, pose = arm.get_position(is_radian=radians)
    if code != 0 or pose is None:
        raise RuntimeError(f"Failed to get EEF pose: code={code}")
    x, y, z, roll, pitch, yaw = pose[:6]
    # SDK returns mm by default; with is_radian=True, angles in rad, pos in mm.
    # Convert mm -> m to keep metric consistent.
    return (np.array([x, y, z]) / 1000.0, np.array([roll, pitch, yaw]))


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    # Standard Euler composition: R = Rx(roll) * Ry(pitch) * Rz(yaw)
    return Rx @ Ry @ Rz


def pose_to_matrix(position_m: np.ndarray, rpy_rad: np.ndarray):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy_rad[0], rpy_rad[1], rpy_rad[2])
    T[:3, 3] = position_m
    return T


def estimate_T_base_color(arm):
    """Estimate T_base_color using uFactory calibration constants and current EEF pose.
    This assumes GraspNet points are expressed in the COLOR camera frame (we use color intrinsics and aligned depth).
    For production, replace with measured hand-eye calibration.
    """
    # Calibration (meters, radians) from uFactory example
    eef_to_color = np.array([0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775])
    color_to_depth = np.array([0.015, 0, 0, 0, 0, 0])

    p_base_eef_m, rpy_base_eef = get_eef_pose(arm, radians=True)
    T_base_eef = pose_to_matrix(p_base_eef_m, rpy_base_eef)

    p_eef_color = eef_to_color[:3]
    rpy_eef_color = eef_to_color[3:]
    T_eef_color = pose_to_matrix(p_eef_color, rpy_eef_color)

    # We return base->color (not including color->depth), since our K and point cloud are color-frame
    T_base_color = T_base_eef @ T_eef_color
    return T_base_color


def move_to_xyz(arm, x_m: float, y_m: float, z_m: float, speed=5):
    """Move to absolute XYZ in meters keeping current orientation."""
    # Get current orientation (in radians)
    _, rpy = get_eef_pose(arm, radians=True)
    # Clamp workspace and height for safety
    z_m_clamped = max(0.05, min(1.2, z_m))
    # Convert meters -> mm
    x_mm, y_mm, z_mm = x_m * 1000.0, y_m * 1000.0, z_m_clamped * 1000.0
    code = arm.set_position(x=x_mm, y=y_mm, z=z_mm, roll=rpy[0], pitch=rpy[1], yaw=rpy[2], is_radian=True, speed=speed,
                            wait=True)
    if code != 0:
        print(f"Failed to move to XYZ: code={code}")
        return False
    return True

