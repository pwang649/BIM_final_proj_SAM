import os
import sys
import cv2
import numpy as np
import torch
import trimesh
from typing import List, Dict, Optional
from PIL import Image

from pytorch3d.transforms import quaternion_to_matrix, Transform3d

from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
sys.path.insert(0, "sam-3d-objects/notebook")
from inference import Inference   # from sam-3d-objects repo


class RGBD3DReconstructor:
    """
    End-to-end class for:
        - RGB-D → point cloud
        - SAM3 segmentation
        - SAM-3D-Objects reconstruction
        - Mesh export
    """

    def __init__(self, sam3d_config_path: str, device="cuda:0"):
        self.device = device
        self.sam3_model = None
        self.sam3_processor = None
        self.sam3d = None

        # geometric transform constants
        self.R_zup_to_yup = torch.tensor(
            [[-1, 0, 0],
             [0, 0, 1],
             [0, 1, 0]], dtype=torch.float32
        )
        self.R_yup_to_zup = self.R_zup_to_yup.T

        self.R_flip_z = torch.tensor(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, -1]], dtype=torch.float32
        )

        # Undo pointmap convention [-X, -Y, Z]
        self.R_pytorch3d_to_cam = torch.tensor(
            [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]], dtype=torch.float32
        )

        # Load SAM-3D-Objects
        self.sam3d = Inference(sam3d_config_path, compile=False)
        self.sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(self.sam3_model, confidence_threshold=0.5)

    # ---------------------------------------------------------
    # Depth → Point Cloud
    # ---------------------------------------------------------
    def depth_to_pointcloud(self, depth_image: np.ndarray, K: np.ndarray) -> torch.Tensor:
        depth = torch.from_numpy(depth_image.astype(np.float32))
        K = torch.from_numpy(K.astype(np.float32))
        H, W = depth.shape

        u = torch.arange(W).unsqueeze(0).repeat(H, 1)
        v = torch.arange(H).unsqueeze(1).repeat(1, W)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = depth
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        # SAM-3D pointmap format is [-x, -y, z]
        return torch.stack((-x, -y, z), dim=-1)

    # ---------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------
    def segment_image(self, image: Image, prompt: str) -> np.ndarray:
        if self.sam3_processor is None:
            raise RuntimeError("SAM3 not initialized. Call initialize_sam3().")

        state = self.sam3_processor.set_image(image)
        output = self.sam3_processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"]

        if masks.shape[0] == 0:
            return np.zeros((image.height, image.width), dtype=bool)

        mask_data = masks[0][0].cpu().numpy()
        return mask_data > 0.5

    # ---------------------------------------------------------
    # Geometry transform
    # ---------------------------------------------------------
    def transform_mesh_vertices(self, vertices, rotation, translation, scale):
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices, dtype=torch.float32)

        vertices = vertices.unsqueeze(0)

        vertices = vertices @ self.R_flip_z
        vertices = vertices @ self.R_yup_to_zup

        R_mat = quaternion_to_matrix(rotation)
        tfm = Transform3d(dtype=vertices.dtype)
        tfm = tfm.scale(scale).rotate(R_mat).translate(*translation)

        vertices_world = tfm.transform_points(vertices)

        vertices = vertices @ self.R_zup_to_yup
        return vertices_world[0]

    # ---------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------
    def reconstruct(
        self,
        image: Image,
        mask: np.ndarray,
        pointmap: torch.Tensor,
        intrinsic: np.ndarray,
        output_path: str,
        seed: int = 42
    ) -> Dict:
        """
        Run SAM-3D-Objects and export mesh.
        """
        image_np = np.array(image).astype(np.uint8)

        output = self.sam3d(
            image_np,
            mask.astype(np.uint8),
            seed=seed,
            pointmap=pointmap,
            intrinsic=intrinsic
        )

        mesh = output["glb"]
        vertices = mesh.vertices

        S = output["scale"][0].cpu().float()
        T = output["translation"][0].cpu().float()
        R = output["rotation"].squeeze().cpu().float()

        vtx_transformed = self.transform_mesh_vertices(vertices, R, T, S)
        vtx_transformed = vtx_transformed @ self.R_pytorch3d_to_cam

        mesh.vertices = vtx_transformed.cpu().numpy().astype(np.float32)
        mesh.export(output_path)

        return output

    # ---------------------------------------------------------
    # Batch Processing
    # ---------------------------------------------------------
    def batch_reconstruct(
        self,
        image: Image,
        pointmap: torch.Tensor,
        intrinsic: np.ndarray,
        prompts: List[str],
        output_dir="outputs",
        seed: int = 42
    ) -> trimesh.Scene:

        os.makedirs(output_dir, exist_ok=True)
        scene = trimesh.Scene()

        for prompt in prompts:
            mask = self.segment_image(image, prompt)
            mesh_out = self.reconstruct(
                image=image,
                mask=mask,
                pointmap=pointmap,
                intrinsic=intrinsic,
                output_path=os.path.join(output_dir, f"{prompt.replace(' ', '_')}.stl"),
                seed=seed
            )
            scene.add_geometry(mesh_out["glb"], node_name=prompt)

        scene.export(os.path.join(output_dir, "scene.stl"))
        return scene

if __name__ == "__main__":
    recon = RGBD3DReconstructor(
        sam3d_config_path="sam-3d-objects/checkpoints/hf/pipeline.yaml"
    )

    rgb = Image.open("rgb.png")
    depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED) * 0.001

    intrinsic = np.array([
        [927.17, 0, 651.3],
        [0, 927.36, 349.62],
        [0, 0, 1]
    ])

    pointmap = recon.depth_to_pointcloud(depth, intrinsic)

    prompts = ["pitcher", "red bowl", "yellow bowl"]

    scene = recon.batch_reconstruct(
        image=rgb,
        pointmap=pointmap,
        intrinsic=intrinsic,
        prompts=prompts,
        output_dir="outputs"
    )

    print("✓ Reconstruction complete")