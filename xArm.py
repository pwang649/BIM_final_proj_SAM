# -------------------------- WORLD / ROBOT -----------------------------
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

import numpy as np
import pybullet as p
import pybullet_data

from xarm.wrapper import XArmAPI
from realworld.robot_config import *
from realworld.xarm_apis import *


@dataclass
class Body:
    id: int
    link_names: List[str]

class World:
    def __init__(self, gui, gravity, dt):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity)
        p.setTimeStep(dt)
        self.dt = dt
        self._setup_env()

    def _setup_env(self):
        p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=135, cameraPitch=-35,
                                     cameraTargetPosition=[0.55, 0.0, 0.5])
        self.plane = p.loadURDF("plane.urdf")

    def add_table(self, size: List[float], pose: Tuple[List[float], List[float]]):
        half = [s/2 for s in size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.7,0.7,0.7,1])
        pos, quat = pose
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=pos, baseOrientation=quat)

    def add_bin_floor(self, pos: List[float], quat: List[float]):
        bx, by = 0.25, 0.25
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bx/2, by/2, 0.005])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[bx/2, by/2, 0.005], rgbaColor=[0.2,0.6,0.8,1])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=pos, baseOrientation=quat)

    def load_robot(self, urdf_path: str, base_pos=(0,0,0.02), base_quat=(0,0,0,1)) -> Body:
        rid = p.loadURDF(urdf_path, basePosition=base_pos, baseOrientation=base_quat, flags=p.URDF_USE_SELF_COLLISION)
        link_names = []
        for i in range(p.getNumJoints(rid)):
            info = p.getJointInfo(rid, i)
            link_names.append(info[12].decode("utf-8"))
        return Body(id=rid, link_names=link_names)

    def step(self, n=1, realtime_gui=True):
        for _ in range(n):
            p.stepSimulation()
            if realtime_gui:
                time.sleep(self.dt)

    def disconnect(self):
        p.disconnect(self.cid)


class XArm:
    def __init__(self, body: Body, ee_substr: str = "link_tcp", real = False):
        self.id = body.id
        self.link_names = body.link_names
        # Gather movable joints
        self.all_joints = [j for j in range(p.getNumJoints(self.id))
                       if p.getJointInfo(self.id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        self.joints = self.all_joints[:7]
        self.gripper_joints = self.all_joints[7:]
        # End effector
        self.ee = self._find_link_index(ee_substr) or (self.joints[-1] if self.joints else 0)
        # Camera link
        self.camera_link = self._find_link_index("camera_optical") or (self.joints[-1] if self.joints else 0)
        self.camera_offset = np.array([0.0, 0.0, 0.0])
        self.camera_rotation = np.array([0.0, 0.0, 0.0])
        # Limits
        self.lower, self.upper, self.ranges, self.rest = [], [], [], []
        for j in self.joints:
            info = p.getJointInfo(self.id, j)
            lo, hi = info[8], info[9]
            if lo > hi:  # continuous
                lo, hi = -math.pi, math.pi
            self.lower.append(lo); self.upper.append(hi); self.ranges.append(hi-lo); self.rest.append(0.0)
        self.constraint = None

        self.real = real
        if real:
            self.xArm = XArmAPI(XARM_IP)
            safe_robot_init(self.xArm)
            initialize_gripper(self.xArm)
            # move_to_home_position(self.xArm, speed=DEFAULT_SPEED)

    def _find_link_index(self, substr: str) -> Optional[int]:
        s = substr.lower()
        for i, name in enumerate(self.link_names):
            if s in name.lower():
                return i
        return None

    def reset(self, q):
        for j, qj in zip(self.all_joints, q):
            p.resetJointState(self.id, j, float(qj))
        if self.real:
            self.execute_path([q])
            self.open_gripper()

    def get_q(self) -> np.ndarray:
        return np.array([p.getJointState(self.id, j)[0] for j in self.joints])

    def _attach(self, obj_id: int):
        self.constraint = p.createConstraint(self.id, self.ee, obj_id, -1, p.JOINT_FIXED,
                                             [0, 0, 0], [0, 0, 0], [0, 0, 0])

    def _detach(self):
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None

    def fk(self, link: Optional[int] = None, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        link = self.ee if link is None else link
        if q is None:
            st = p.getLinkState(self.id, link, computeForwardKinematics=True)
            return np.array(st[4]), np.array(st[5])
        else:
            for k, j in enumerate(self.joints):
                p.resetJointState(self.id, j, float(q[k]))
            st = p.getLinkState(self.id, link, computeForwardKinematics=True)
            return np.array(st[4]), np.array(st[5])

    def ik(self, pos: np.ndarray, quat: np.ndarray, iters: int = 200) -> np.ndarray:
        q = p.calculateInverseKinematics(self.id, self.ee, pos.tolist(), quat.tolist(),
                                         lowerLimits=self.lower, upperLimits=self.upper,
                                         jointRanges=self.ranges, restPoses=self.rest,
                                         maxNumIterations=iters, residualThreshold=1e-4)
        return np.array(q[:len(self.joints)])

    def cmd(self, q: np.ndarray, force=250, vel=1.0, steps=90, settle=20):
        q0 = self.get_q()
        for s in range(steps):
            a = s / max(steps-1, 1)
            a = 3*a*a - 2*a*a*a  # smoothstep
            qt = q0 + a*(q - q0)
            for k, j in enumerate(self.joints):
                p.setJointMotorControl2(self.id, j, p.POSITION_CONTROL, targetPosition=float(qt[k]), force=force, maxVelocity=vel)
            p.stepSimulation()
        for _ in range(settle):
            p.stepSimulation()

    # def open_gripper(self, opening: float):
    #     for j in self.gripper_joints:
    #         p.setJointMotorControl2(self.id, j, p.POSITION_CONTROL, targetPosition=opening, force=80)
    #     for _ in range(60):
    #         p.stepSimulation()

    def _get_camera_to_world_transform(self):
        ee_pos, ee_quat = self.fk(link=self.camera_link)
        ee_rot = R.from_quat(ee_quat).as_matrix()

        # --- Rotate camera frame 90° clockwise around Z ---
        Rz = R.from_euler('z', 90.0, degrees=True).as_matrix()
        camera_pos = ee_pos + ee_rot @ (Rz @ self.camera_offset)
        camera_rot = ee_rot @ Rz

        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, :3] = camera_rot
        T_cam_to_world[:3, 3] = camera_pos

        return T_cam_to_world

    # xArm APIs
    def execute_path(self, path):
        for q in path:
            self.xArm.set_servo_angle(angle=q, is_radian=True, speed=DEFAULT_SPEED, wait=False)

    def open_gripper(self):
        self.xArm.set_gripper_position(GRIPPER_OPEN_POSITION, wait=True)

    def close_gripper(self):
        self.xArm.set_gripper_position(GRIPPER_CLOSE_POSITION, wait=True)

