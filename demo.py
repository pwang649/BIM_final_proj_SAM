import math
import numpy as np
import torch
import pybullet as p
import trimesh

from collision_checker import CollisionChecker
from mesh_gen import RGBD3DReconstructor
from camera import Camera
from rrt import RRT
from xArm import World, XArm
from scipy.spatial.transform import Rotation as R

class Demo:
    def __init__(self, gui: bool = True, real = False, xarm_path: str = "urdf/xarm/xarm7_with_gripper.urdf"):
        self.world = World(gui, -9.81, 1.0 / 240.0)

        robot_body = self.world.load_robot(xarm_path, (0, 0, 0.002))
        self.robot = XArm(robot_body, real=real)

        # Set home position
        home_pos = np.array([0.3, 0.0, 0.3])
        home_euler = np.array([3.1416, 0.0, 0.0])
        home_quat = R.from_euler('xyz', home_euler).as_quat()
        self.home = self.robot.ik(home_pos, home_quat)
        self.home = np.concatenate([self.home, np.zeros(len(self.robot.all_joints) - len(self.home))])
        self.robot.reset(self.home)

        place_pos = np.array([0.0, 0.3, 0.1])
        place_euler = np.array([3.1416, 0.0, 3.1416 / 2])
        place_quat = R.from_euler('xyz', place_euler).as_quat()
        self.q_place = self.robot.ik(place_pos, place_quat)

        # self.mesh_id, self.mesh_centroid = self.load_mesh_object(mesh_path)

        obstacles = []  # self.world.plane]
        self.cc = CollisionChecker(self.robot, obstacles)

        self.planner = RRT(
            self.cc,
            q_min=np.array(self.robot.lower),
            q_max=np.array(self.robot.upper),
            step_size=0.2,
            goal_bias=0.1,
            max_iter=3000,
            goal_threshold=0.05
        )

        self.camera = Camera()
        self.camera_K = np.load("intrinsics.npy")
        self.mesh_gen = RGBD3DReconstructor(sam3d_config_path="sam-3d-objects/checkpoints/hf/pipeline.yaml")

    def load_mesh_object(self, scene, mass=0.0):
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in scene.geometry.values()]
            )
            print("creepy if")
        else:
            mesh = scene
        local_centroid = mesh.center_mass
        mesh.fill_holes()
        # mesh.remove_degenerate_faces()
        # mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        # if not mesh.is_watertight:
        #     mesh = mesh.convex_hull
        mesh = mesh.convex_hull
        vertices = mesh.vertices
        faces = mesh.faces

        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, vertices=vertices.tolist(),
                                                    indices=faces.flatten().tolist())
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, vertices=vertices.tolist(),
                                              indices=faces.flatten().tolist())

        T_cam_to_world = self.robot._get_camera_to_world_transform()
        camera_rot = T_cam_to_world[:3, :3]
        camera_pos = T_cam_to_world[:3, 3]
        camera_quat = R.from_matrix(camera_rot).as_quat()

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=camera_pos,
            baseOrientation=camera_quat,
        )

        world_centroid = camera_rot @ local_centroid + camera_pos
        return body_id, world_centroid

    def run_realworld_demo(self):
        while True:
            self.robot.reset(self.home)
            user_input = input("\nPress Enter for new target (or 'q' to quit): ").strip().lower()

            if user_input == 'q':
                break
            color_img, depth_img = self.camera.capture_aligned_rgbd()
            pointmap = self.mesh_gen.depth_to_pointcloud(depth_img, self.camera_K)
            scene = self.mesh_gen.batch_reconstruct(image=color_img,
                                                    pointmap=pointmap,
                                                    intrinsic=self.camera_K,
                                                    prompts=["marker"],
                                                    output_dir="output")
            mesh_id, mesh_centroid = self.load_mesh_object(scene)
            pick_pos = np.array(mesh_centroid)
            pick_euler = np.array([3.1416, 0.0, 0.0])
            pick_quat = R.from_euler('xyz', pick_euler).as_quat()
            q_pick = self.robot.ik(pick_pos, pick_quat)
            # if not self.cc.is_state_valid(q_pick):
            #     continue
            path = self.planner.plan(self.robot.get_q(), q_pick)
            self.robot.execute_path(path)
            self.robot.close_gripper()

            path = self.planner.plan(self.robot.get_q(), self.q_place)
            self.robot.execute_path(path)
            self.robot.open_gripper()

        print("Demo finished!")
        self.world.disconnect()
        self.camera.stop()


if __name__ == '__main__':
    config = {
        "gui": True,
        "real": True,
        "xarm_path": "urdf/xarm/xarm7_with_gripper.urdf",
    }
    demo = Demo(**config)
    demo.run_realworld_demo()