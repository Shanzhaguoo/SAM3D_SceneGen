# Multi-View 3D Gaussian Scene Generation

A pipeline for generating 3D Gaussian Splatting scenes from multi-view images using text-guided segmentation.

## Features

- **Text-Guided Segmentation**: Uses Grounding DINO + SAM2 to segment objects based on text prompts
- **Camera Pose Estimation**: Estimates camera poses from multi-view images using VGGT
- **3D Gaussian Generation**: Generates 3D Gaussians for each detected object
- **Multi-View Fusion**: Merges Gaussians from different views into a unified scene
- **Visualization**: Outputs segmentation masks, detection results, and camera trajectory

## Pipeline Overview

```
Input Images → Object Detection (GDINO) → Segmentation (SAM2) → Gaussian Generation → Pose Estimation (VGGT) → Scene Merging → PLY Output
```

## Output Structure

```
outputs/
├── merged_scene.ply          # Final merged 3D Gaussian scene
├── metadata.json             # Scene metadata
└── visualization/
    ├── detections/           # Detection bounding boxes
    ├── masks/                # Segmentation masks
    └── camera_trajectory.png # Estimated camera poses
```

## Usage

```python
python scene_generation.py
```

Configure parameters in `main()`:
- `IMAGE_DIR`: Input image directory
- `TEXT_PROMPT`: Object detection prompt
- `OUTPUT_DIR`: Output directory

## Installation
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
git clone https://github.com/xxx/vggt.git
Follow the env setup guide: https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md