"""
Integrated Pipeline: SAM3 Segmentation + SAM 3D Objects Reconstruction

This script demonstrates how to:
1. Use SAM3 to segment objects of interest from an image using text prompts
2. Use SAM 3D Objects to reconstruct the 3D mesh from the segmented object

The pipeline is minimal and focused on showing the connection between the two models.
For production use, consider adding error handling, batching, and optimization.
"""

import os
import sys
import numpy as np
from PIL import Image

# Store the root directory where this script lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# INPUT VARIABLES - Modify these to test with different images and objects
# ============================================================================

# Path to input image (relative to root directory)
INPUT_IMAGE_PATH = "rgb.png"

# Text prompt for SAM3 segmentation (e.g., "chair", "person", "lamp")
SEGMENTATION_PROMPT = "pitcher"

# Output path for the 3D mesh (relative to root directory)
OUTPUT_MESH_PATH = "output_mesh.ply"

# ============================================================================
# STAGE 1: Object Segmentation with SAM3
# ============================================================================

def segment_with_sam3(image_path: str, text_prompt: str) -> np.ndarray:
    """
    Segment object of interest from image using SAM3 with text prompt.
    
    Args:
        image_path: Path to input image (absolute path)
        text_prompt: Text description of object to segment (e.g., "chair", "person")
    
    Returns:
        Binary mask array (H, W) where True indicates the segmented object
    """
    print(f"\n[STAGE 1] Segmenting '{text_prompt}' from image...")
    
    # Save current directory and change to sam3 project directory
    original_dir = os.getcwd()
    sam3_dir = os.path.join(ROOT_DIR, "sam3")
    os.chdir(sam3_dir)
    
    try:
        # Add sam3 module directory to path so imports work naturally
        if sam3_dir not in sys.path:
            sys.path.insert(0, sam3_dir)
        
        # Import SAM3 - now that we're in the sam3 directory, sam3.sam3 refers to the correct module
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Load model
        print("  - Loading SAM3 model...")
        model = build_sam3_image_model()
        processor = Sam3Processor(model, confidence_threshold=0.5)
        
        # Load and process image (use absolute path since we changed directory)
        print(f"  - Loading image from {image_path}...")
        image = Image.open(image_path)
        
        # Set image and get inference state
        inference_state = processor.set_image(image)
        
        # Get segmentation with text prompt
        print(f"  - Running segmentation for prompt: '{text_prompt}'...")
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # Extract mask - take the highest confidence mask
        masks = output["masks"]  # Shape: (num_masks, H, W)
        
        if masks.shape[0] == 0:
            print(f"  WARNING: No objects found matching '{text_prompt}'")
            binary_mask = np.zeros((image.height, image.width), dtype=bool)
        else:
            # Use the mask with highest confidence (first one typically)
            # SAM3 returns masks with shape (1, H, W), we need to squeeze to (H, W)
            mask_data = masks[0].cpu().numpy()  # Shape: (H, W)
            binary_mask = mask_data > 0.5
        
        print(f"  - Segmentation complete. Mask shape: {binary_mask.shape}")
        return binary_mask
    
    finally:
        # Always restore original directory
        os.chdir(original_dir)


# ============================================================================
# STAGE 2: 3D Reconstruction with SAM 3D Objects
# ============================================================================

def reconstruct_3d_with_sam3d_objects(
    image_path: str, 
    mask: np.ndarray, 
    output_path: str
) -> dict:
    """
    Reconstruct 3D mesh from segmented object using SAM 3D Objects.
    
    Args:
        image_path: Path to input image (absolute path)
        mask: Binary segmentation mask (H, W) from SAM3
        output_path: Path to save output PLY mesh (absolute path)
    
    Returns:
        Dictionary with reconstruction outputs (gaussian splat, mesh, etc.)
    """
    print(f"\n[STAGE 2] Reconstructing 3D mesh from segmented object...")
    
    # Validate and reshape mask to ensure it's 2D (H, W)
    print(f"  - Mask input shape: {mask.shape}, dtype: {mask.dtype}")
    
    # If mask has extra dimensions, squeeze them
    if mask.ndim > 2:
        mask = np.squeeze(mask)
        print(f"  - Squeezed mask to shape: {mask.shape}")
    
    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    # Save current directory and change to sam-3d-objects project directory
    original_dir = os.getcwd()
    sam3d_obj_dir = os.path.join(ROOT_DIR, "sam-3d-objects")
    sam3d_notebook_dir = os.path.join(sam3d_obj_dir, "notebook")
    os.chdir(sam3d_notebook_dir)
    
    try:
        # Add sam3d_objects module to path so imports work naturally
        if sam3d_obj_dir not in sys.path:
            sys.path.insert(0, sam3d_obj_dir)
        if sam3d_notebook_dir not in sys.path:
            sys.path.insert(0, sam3d_notebook_dir)
        
        # Import SAM 3D Objects inference code from the notebook directory
        from inference import Inference, load_image
        
        # Load image (use absolute path)
        print(f"  - Loading image from {image_path}...")
        image_array = load_image(image_path)
        
        # Initialize inference pipeline
        print("  - Loading SAM 3D Objects model...")
        tag = "hf"
        config_path = os.path.join(sam3d_obj_dir, f"checkpoints/{tag}/pipeline.yaml")
        inference = Inference(config_path, compile=False)
        
        # Run reconstruction with the segmented mask
        print("  - Running 3D reconstruction...")
        output = inference(image_array, mask, seed=42)
        
        # Save the gaussian splat as PLY mesh (use absolute path for output)
        print(f"  - Saving mesh to {output_path}...")
        output["gs"].save_ply(output_path)
        
        print(f"  - 3D reconstruction complete!")
        return output
    
    finally:
        # Always restore original directory
        os.chdir(original_dir)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_integrated_pipeline(
    image_path: str, 
    prompt: str, 
    output_mesh_path: str
) -> None:
    """
    Run the complete integrated pipeline: segmentation -> reconstruction
    
    Args:
        image_path: Path to input image (relative to root directory)
        prompt: Text prompt for object segmentation
        output_mesh_path: Path to save output 3D mesh (relative to root directory)
    """
    print("=" * 70)
    print("SAM3 + SAM 3D Objects Integrated Pipeline")
    print("=" * 70)
    
    # Convert relative paths to absolute paths
    abs_image_path = os.path.join(ROOT_DIR, image_path)
    abs_output_path = os.path.join(ROOT_DIR, output_mesh_path)
    
    # Validate input
    if not os.path.exists(abs_image_path):
        print(f"ERROR: Image not found at {abs_image_path}")
        return
    
    try:
        # Stage 1: Segment object with SAM3
        segmentation_mask = segment_with_sam3(abs_image_path, prompt)
        
        # Check if segmentation was successful
        if not segmentation_mask.any():
            print(f"ERROR: Segmentation failed - no mask generated for '{prompt}'")
            return
        
        # Stage 2: Reconstruct 3D with SAM 3D Objects
        output = reconstruct_3d_with_sam3d_objects(
            abs_image_path, 
            segmentation_mask, 
            abs_output_path
        )
        
        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print(f"Output mesh saved to: {abs_output_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the pipeline with the specified inputs
    run_integrated_pipeline(
        image_path=INPUT_IMAGE_PATH,
        prompt=SEGMENTATION_PROMPT,
        output_mesh_path=OUTPUT_MESH_PATH
    )
