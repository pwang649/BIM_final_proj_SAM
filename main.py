import os
import sys
import time

import numpy as np
import trimesh
from PIL import Image
from typing import List, Dict, Optional

from trimesh import Scene

from gpt4o import read_image_as_base64, call_openai_chat_completion
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor
sys.path.insert(0, "sam-3d-objects/notebook")
from inference import Inference
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SAM3_MODEL = None
SAM3_PROCESSOR = None
SAM3D_INFERENCE = None

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
    return binary_mask

def reconstruct_to_3d(image: Image, mask: np.ndarray, output_path: str, seed: int = 42) -> Optional[Dict]:
    image = np.array(image)
    image_array = image.astype(np.uint8)

    output = SAM3D_INFERENCE(image_array, mask, seed=seed)

    # Save the gaussian splat as PLY mesh
    print(f"    Saving mesh to {output_path}...")
    output["mesh"].export(output_path)
    return output["mesh"]

def batch_process(image: Image, prompts: List[str], output_dir: str = "outputs", seed: int = 42) -> Scene:
    abs_output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    scene = trimesh.Scene()
    for prompt in prompts:
        output_path = f"{prompt}.stl"
        mask = segment_image(image, prompt)
        mesh = reconstruct_to_3d(image, mask, output_path, seed=seed)
        scene.add_geometry(mesh, node_name="prompt")
    scene.export("scene.stl")

    return scene

if __name__ == "__main__":
    image_path = "rgb.png"
    api_key = "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"
    model = "gpt-4o"
    user_prompt = (
        "Describe the objects in the image as prompts (one per line) that's useful for an image segmentation model like SAM. "
        "Don't include the prompts background and table. Don't include any extra symbols in the response. "
        "Return the objects in an order that is best for sequencing remove to clear the table.")
    max_tokens = 300

    initialize_sam3()
    initialize_sam3d_objects()
    image = Image.open(image_path)

    base64_image = read_image_as_base64(image_path)
    completion_text = call_openai_chat_completion(api_key=api_key, model=model, user_prompt=user_prompt, base64_image=base64_image, max_tokens=max_tokens)
    prompts_list = [line.strip() for line in completion_text.split("\n") if line.strip()]
    print(prompts_list)

    results = batch_process(image, prompts_list, output_dir="batch_outputs")
    scene = trimesh.Scene()

    print("\n✓ Pipeline execution complete!")
