import sys
import cv2
import numpy as np
import torch

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

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
    points = torch.stack((x, y, z), dim=-1)

    return points

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("../rgb.png")
# mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
mask = np.load("../pitcher_mask.npy")
depth_img = cv2.imread("../depth.png", cv2.IMREAD_UNCHANGED)
intrinsic = np.load("../intrinsics.npy")
point_map = depth_to_pointcloud(depth_img, intrinsic)
depth = torch.from_numpy(depth_img).unsqueeze(0)

# run model
output = inference(image, mask, seed=42, pointmap=point_map, intrinsic=intrinsic)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
