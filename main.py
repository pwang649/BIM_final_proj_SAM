import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from typing import List, Dict, Optional

from numba.cuda.extending import intrinsic
from trimesh import Scene

from gpt4o import read_image_as_base64, call_openai_chat_completion
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
sys.path.insert(0, "sam-3d-objects/notebook")
from inference import Inference
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SAM3_MODEL = None
SAM3_PROCESSOR = None
SAM3D_INFERENCE = None

def transform_mesh_vertices(vertices, rotation, translation, scale):
	if isinstance(vertices, np.ndarray):
		vertices = torch.tensor(vertices, dtype=torch.float32)

	vertices = vertices.unsqueeze(0)  #  batch dimension [1, N, 3]

	# Flip Z-axis
	vertices = vertices @ R_flip_z.to(vertices.device)

	# Convert mesh from Y-up (GLB) → Z-up (canonical PyTorch3D)
	vertices = vertices @ R_yup_to_zup.to(vertices.device)

	# apply gaussian splatting transformations
	R_mat = quaternion_to_matrix(rotation.to(vertices.device))
	tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
	tfm = (
		tfm.scale(scale)
		   .rotate(R_mat)
		   .translate(translation[0], translation[1], translation[2])
	)
	vertices_world = tfm.transform_points(vertices)

	# convert back to Y-up so GLB is saved correctly
	vertices = vertices @ R_zup_to_yup.to(vertices.device)

	# remove batch dimension
	return vertices_world[0]

def depth_to_pointcloud(depth_image, intrinsics_K):
    # Convert inputs to torch tensors
    depth = torch.from_numpy(depth_image.astype(np.float32))  # (H, W)
    K = torch.from_numpy(intrinsics_K.astype(np.float32))     # (3, 3)

    H, W = depth.shape

    # Create pixel coordinate grid
    u = torch.arange(W).unsqueeze(0).repeat(H, 1)   # (H, W)
    v = torch.arange(H).unsqueeze(1).repeat(1, W)   # (H, W)

    # Intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Compute x,y,z in camera coordinates
    z = depth
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    # Stack into (H, W, 3)
    points = torch.stack((-x, -y, z), dim=-1)

    return points

def initialize_sam3():
    global SAM3_MODEL, SAM3_PROCESSOR
    SAM3_MODEL = build_sam3_image_model()
    SAM3_PROCESSOR = Sam3Processor(SAM3_MODEL, confidence_threshold=0.5)

def initialize_sam3d_objects():
    global SAM3D_INFERENCE
    config_path = "sam-3d-objects/checkpoints/hf/pipeline.yaml"
    SAM3D_INFERENCE = Inference(config_path, compile=False)

def segment_image(image: Image, text_prompt: str) -> Optional[np.ndarray]:
    inference_state = SAM3_PROCESSOR.set_image(image)
    output = SAM3_PROCESSOR.set_text_prompt(state=inference_state, prompt=text_prompt)
    masks = output["masks"]  # Shape: (num_masks, H, W)

    if masks.shape[0] == 0:
        print(f"    WARNING: No objects found matching '{text_prompt}'")
        return np.zeros((image.height, image.width), dtype=bool)

    # Use the mask with highest confidence
    mask_data = masks[0][0].cpu().numpy()  # Shape: (H, W)
    binary_mask = mask_data > 0.5
    np.save("yellow_bowl_mask.npy", binary_mask)
    return binary_mask

def reconstruct_to_3d(image: Image, mask: np.ndarray, pointmap, intrinsic, output_path: str, seed: int = 42) -> Optional[Dict]:
    image = np.array(image)
    image_array = image.astype(np.uint8)

    output = SAM3D_INFERENCE(image_array, mask, seed=seed, pointmap=pointmap, intrinsic=intrinsic)

    # Save the gaussian splat as PLY mesh
    print(f"    Saving mesh to {output_path}...")

    mesh = output["glb"]
    vertices = mesh.vertices

    vertices_tensor = torch.tensor(vertices)

    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()

    # Transform vertices
    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)

    # --- Convert vertices from pointmap frame back to true camera frame ---
    # (undoing the earlier pointmap conversion: [-X, -Y, Z])
    vertices_transformed = vertices_transformed @ R_pytorch3d_to_cam.to(vertices_transformed.device)

    # Update mesh vertices
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    # Export mesh
    mesh.export(output_path)
    # output["glb"].export(output_path)
    return output

def batch_process(image: Image, pointmap, intrinsic, prompts: List[str], output_dir: str = "outputs", seed: int = 42) -> Scene:
    abs_output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    masks = [np.load("pitcher_mask.npy"),
             np.load("red_bowl_mask.npy"),
             np.load("yellow_bowl_mask.npy"),
             np.load("green_bowl_mask.npy"),]
    scene = trimesh.Scene()
    for i, prompt in enumerate(prompts):
        output_path = f"{prompt}.stl"
        # mask = segment_image(image, prompt)
        mesh = reconstruct_to_3d(image, masks[i], pointmap, intrinsic, output_path, seed=seed)
        scene.add_geometry(mesh["glb"], node_name=prompt)
    scene.export("scene.stl")

    return scene

if __name__ == "__main__":
    image_path = "rgb.png"
    depth_path = "depth.png"
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) * 0.001
    # intrinsic = np.load("intrinsics.npy")
    intrinsic = np.array([[9.271697387695312500e+02, 0.000000000000000000e+00, 6.513150634765625000e+02],
                            [0.000000000000000000e+00, 9.273668823242187500e+02, 3.496213378906250000e+02],
                            [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    point_map = depth_to_pointcloud(depth_img, intrinsic)
    R_zup_to_yup = torch.tensor([
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=torch.float32)

    R_yup_to_zup = R_zup_to_yup.T

    # flip Z-axis
    R_flip_z = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ], dtype=torch.float32)

    # Convert from pointmap convention [-X, -Y, Z] back to true
    R_pytorch3d_to_cam = torch.tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    api_key = os.environ.get("OPENAI_API_KEY")
    model = "gpt-4o"
    user_prompt = (
        "Describe the objects in the image as prompts (one per line) that's useful for an image segmentation model like SAM. "
        "Don't include the prompts background and table. Don't include any extra symbols in the response. "
        "Return the objects in an order that is best for sequencing remove to clear the table.")
    max_tokens = 300

    # initialize_sam3()
    initialize_sam3d_objects()
    image = Image.open(image_path)

    # base64_image = read_image_as_base64(image_path)
    # completion_text = call_openai_chat_completion(api_key=api_key, model=model, user_prompt=user_prompt, base64_image=base64_image, max_tokens=max_tokens)
    # prompts_list = [line.strip() for line in completion_text.split("\n") if line.strip()]
    prompts_list = ["pitcher", "red bowl", "yellow bowl", "green bowl"]
    print(prompts_list)

    results = batch_process(image, point_map, intrinsic, prompts_list, output_dir="batch_outputs")
    scene = trimesh.Scene()

    print("\n✓ Pipeline execution complete!")
