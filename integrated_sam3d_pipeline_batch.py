"""
Integrated Pipeline with Batch Inference: SAM3 Segmentation + SAM 3D Objects Reconstruction

This script initializes all models once at startup, then enables efficient batch inference.
Ideal for processing multiple images with the same pipeline - model initialization overhead paid once.

Architecture:
1. Initialize SAM3 model and processor
2. Initialize SAM 3D Objects inference pipeline
3. Run batch inference on multiple (image, prompt) pairs

Benefits:
- Models loaded only once at startup
- Much faster for multiple images (no repeated model loading)
- Memory efficient (models stay in VRAM)
- Clean function interface for batch processing
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

# Store the root directory where this script lives
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global model instances (initialized once at startup)
SAM3_MODEL = None
SAM3_PROCESSOR = None
SAM3D_INFERENCE = None

# ============================================================================
# MODEL INITIALIZATION - Run once at startup
# ============================================================================

def initialize_sam3():
    """
    Initialize SAM3 model and processor.
    Called once at pipeline startup.
    """
    global SAM3_MODEL, SAM3_PROCESSOR
    
    print("\n[INIT] Initializing SAM3 model...")
    
    # Save current directory and change to sam3 project directory
    original_dir = os.getcwd()
    sam3_dir = os.path.join(ROOT_DIR, "sam3")
    os.chdir(sam3_dir)
    
    try:
        # Add sam3 module directory to path
        if sam3_dir not in sys.path:
            sys.path.insert(0, sam3_dir)
        
        # Import SAM3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Load model
        print("  - Building SAM3 model...")
        SAM3_MODEL = build_sam3_image_model()
        SAM3_PROCESSOR = Sam3Processor(SAM3_MODEL, confidence_threshold=0.5)
        
        print("  ✓ SAM3 model initialized successfully")
        return True
    
    except Exception as e:
        print(f"  ✗ Failed to initialize SAM3: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.chdir(original_dir)


def initialize_sam3d_objects():
    """
    Initialize SAM 3D Objects inference pipeline.
    Called once at pipeline startup.
    """
    global SAM3D_INFERENCE
    
    print("\n[INIT] Initializing SAM 3D Objects model...")
    
    # Save current directory and change to sam-3d-objects project directory
    original_dir = os.getcwd()
    sam3d_obj_dir = os.path.join(ROOT_DIR, "sam-3d-objects")
    sam3d_notebook_dir = os.path.join(sam3d_obj_dir, "notebook")
    os.chdir(sam3d_notebook_dir)
    
    try:
        # Add sam3d_objects module to path
        if sam3d_obj_dir not in sys.path:
            sys.path.insert(0, sam3d_obj_dir)
        if sam3d_notebook_dir not in sys.path:
            sys.path.insert(0, sam3d_notebook_dir)
        
        # Import SAM 3D Objects inference code
        from inference import Inference
        
        # Initialize inference pipeline
        print("  - Building SAM 3D Objects pipeline...")
        tag = "hf"
        config_path = os.path.join(sam3d_obj_dir, f"checkpoints/{tag}/pipeline.yaml")
        SAM3D_INFERENCE = Inference(config_path, compile=False)
        
        print("  ✓ SAM 3D Objects model initialized successfully")
        return True
    
    except Exception as e:
        print(f"  ✗ Failed to initialize SAM 3D Objects: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.chdir(original_dir)


def initialize_pipeline():
    """
    Initialize all models for the pipeline.
    Call this once at the start of your script.
    """
    print("=" * 70)
    print("Initializing SAM3 + SAM 3D Objects Pipeline")
    print("=" * 70)
    
    # Initialize models
    sam3_ok = initialize_sam3()
    sam3d_ok = initialize_sam3d_objects()
    
    if not (sam3_ok and sam3d_ok):
        print("\n✗ Pipeline initialization failed!")
        return False
    
    print("\n" + "=" * 70)
    print("✓ Pipeline fully initialized and ready for batch inference!")
    print("=" * 70)
    return True


# ============================================================================
# BATCH INFERENCE FUNCTIONS
# ============================================================================

def segment_image(image_path: str, text_prompt: str) -> Optional[np.ndarray]:
    """
    Segment object from image using SAM3 (requires initialize_pipeline() called first).
    
    Args:
        image_path: Path to input image (absolute path)
        text_prompt: Text description of object to segment (e.g., "chair", "person")
    
    Returns:
        Binary mask array (H, W) where True indicates the segmented object, or None on failure
    """
    if SAM3_MODEL is None or SAM3_PROCESSOR is None:
        print(f"ERROR: SAM3 not initialized. Call initialize_pipeline() first.")
        return None
    
    try:
        print(f"  Segmenting '{text_prompt}' from image...")
        
        # Load and process image
        image = Image.open(image_path)
        
        # Set image and get inference state
        inference_state = SAM3_PROCESSOR.set_image(image)
        
        # Get segmentation with text prompt
        output = SAM3_PROCESSOR.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # Extract mask
        masks = output["masks"]  # Shape: (num_masks, H, W)
        
        if masks.shape[0] == 0:
            print(f"    WARNING: No objects found matching '{text_prompt}'")
            return np.zeros((image.height, image.width), dtype=bool)
        
        # Use the mask with highest confidence
        mask_data = masks[0].cpu().numpy()  # Shape: (H, W)
        binary_mask = mask_data > 0.5
        
        print(f"    ✓ Segmentation complete. Mask shape: {binary_mask.shape}")
        return binary_mask
    
    except Exception as e:
        print(f"    ✗ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def reconstruct_to_3d(image_path: str, mask: np.ndarray, output_path: str, seed: int = 42) -> Optional[Dict]:
    """
    Reconstruct 3D mesh from segmented object using SAM 3D Objects 
    (requires initialize_pipeline() called first).
    
    Args:
        image_path: Path to input image (absolute path)
        mask: Binary segmentation mask (H, W) from segmentation
        output_path: Path to save output PLY mesh (absolute path)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with reconstruction outputs, or None on failure
    """
    if SAM3D_INFERENCE is None:
        print(f"ERROR: SAM 3D Objects not initialized. Call initialize_pipeline() first.")
        return None
    
    try:
        print(f"  Reconstructing 3D mesh...")
        
        # Validate and reshape mask
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Change to sam-3d-objects notebook directory for relative paths to work
        original_dir = os.getcwd()
        sam3d_notebook_dir = os.path.join(ROOT_DIR, "sam-3d-objects", "notebook")
        os.chdir(sam3d_notebook_dir)
        
        try:
            # Load image utilities
            from inference import load_image
            
            # Load image
            image_array = load_image(image_path)
            
            # Run reconstruction with the segmented mask
            output = SAM3D_INFERENCE(image_array, mask, seed=seed)
            
            # Save the gaussian splat as PLY mesh
            print(f"    Saving mesh to {output_path}...")
            output["gs"].save_ply(output_path)
            
            print(f"    ✓ 3D reconstruction complete!")
            return output
        
        finally:
            os.chdir(original_dir)
    
    except Exception as e:
        print(f"    ✗ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_image_to_mesh(image_path: str, text_prompt: str, output_path: str, seed: int = 42) -> bool:
    """
    Complete pipeline: segment image and reconstruct to 3D mesh.
    
    Args:
        image_path: Path to input image (relative to ROOT_DIR or absolute)
        text_prompt: Text description of object to segment
        output_path: Path to save output PLY mesh (relative to ROOT_DIR or absolute)
        seed: Random seed for reproducibility
    
    Returns:
        True if successful, False otherwise
    """
    # Convert relative paths to absolute
    if not os.path.isabs(image_path):
        image_path = os.path.join(ROOT_DIR, image_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(ROOT_DIR, output_path)
    
    # Validate input
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return False
    
    # Segment
    mask = segment_image(image_path, text_prompt)
    if mask is None or not mask.any():
        print(f"ERROR: Segmentation failed for '{text_prompt}'")
        return False
    
    # Reconstruct
    result = reconstruct_to_3d(image_path, mask, output_path, seed=seed)
    return result is not None


# ============================================================================
# BATCH INFERENCE UTILITIES
# ============================================================================

def batch_process(
    image_prompt_pairs: List[Tuple[str, str]], 
    output_dir: str = "outputs",
    seed: int = 42
) -> List[bool]:
    """
    Process multiple (image, prompt) pairs in batch.
    
    Args:
        image_prompt_pairs: List of (image_path, text_prompt) tuples
        output_dir: Directory to save output meshes (relative to ROOT_DIR)
        seed: Random seed for reproducibility
    
    Returns:
        List of success/failure booleans for each image
    """
    # Create output directory
    abs_output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    results = []
    
    print("\n" + "=" * 70)
    print(f"Batch processing {len(image_prompt_pairs)} images")
    print("=" * 70)
    
    for idx, (image_path, prompt) in enumerate(image_prompt_pairs, 1):
        print(f"\n[{idx}/{len(image_prompt_pairs)}] Processing: {image_path} -> '{prompt}'")
        
        # Generate output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(abs_output_dir, f"{base_name}_{prompt}.ply")
        
        # Process
        success = process_image_to_mesh(image_path, prompt, output_path, seed=seed)
        results.append(success)
        
        if success:
            print(f"  ✓ Successfully saved: {output_path}")
        else:
            print(f"  ✗ Failed to process this image")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Batch processing complete: {sum(results)}/{len(results)} successful")
    print("=" * 70)
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline once at startup
    if not initialize_pipeline():
        print("Failed to initialize pipeline. Exiting.")
        sys.exit(1)
    
    # Example 1: Single image processing
    print("\n\n[EXAMPLE 1] Single image processing:")
    print("-" * 70)
    success = process_image_to_mesh(
        image_path="rgb.png",  # Input image
        text_prompt="pitcher",  # What to segment
        output_path="output_single.ply",  # Output mesh
        seed=42
    )
    
    # Example 2: Batch processing multiple images
    print("\n\n[EXAMPLE 2] Batch processing multiple images:")
    print("-" * 70)
    
    # Define multiple images and prompts to process
    image_prompt_pairs = [
        ("rgb.png", "pitcher"),
        ("rgb.png", "person"),
        ("rgb.png", "background"),
        # Add more (image, prompt) pairs here
    ]
    
    # Process all in batch (models stay initialized, just inference runs)
    results = batch_process(image_prompt_pairs, output_dir="batch_outputs", seed=42)
    
    print("\n✓ Pipeline execution complete!")
