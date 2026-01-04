import os
import sys
import glob
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.ops import box_convert, nms
from copy import deepcopy

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append("notebook")
sys.path.append("Grounded-SAM-2")
sys.path.append("vggt")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import (
    load_model as load_gdino_model,
    load_image as load_image_gdino,
    predict as gdino_predict,
)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_invert

from inference import (
    Inference,
    load_image as load_image_gs,
    make_scene,
    interactive_visualizer,
)


class Config:
    SAM2_CHECKPOINT = "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "sam2.1/sam2.1_hiera_l"
    GROUNDING_DINO_CONFIG = "./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "./Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.45
    TEXT_THRESHOLD = 0.30
    NMS_THRESHOLD = 0.5
    MIN_MASK_PIXELS = 1000
    GS_CONFIG_PATH = "checkpoints/hf/checkpoints/pipeline.yaml"
    VGGT_CHECKPOINT = "vggt/checkpoint/model.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.det_dir = os.path.join(output_dir, "visualization", "detections")
        self.mask_dir = os.path.join(output_dir, "visualization", "masks")
        os.makedirs(self.det_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
    
    def save_detection(self, image: np.ndarray, boxes: torch.Tensor, labels: list, 
                       confidences: torch.Tensor, save_name: str):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        h, w = image.shape[:2]
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(boxes), 1)))
        
        for idx, (box, label, conf) in enumerate(zip(boxes, labels, confidences)):
            box_pixel = box * torch.Tensor([w, h, w, h])
            cx, cy, bw, bh = box_pixel.numpy()
            x1, y1 = cx - bw/2, cy - bh/2
            
            rect = plt.Rectangle((x1, y1), bw, bh, fill=False, 
                                  edgecolor=colors[idx], linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{label}: {conf:.2f}", 
                    color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8))
        
        ax.axis('off')
        ax.set_title(f"Detections: {save_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.det_dir, f"{save_name}_detection.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_masks(self, image: np.ndarray, masks: np.ndarray, labels: list, save_name: str):
        n_masks = len(masks)
        if n_masks == 0:
            return
        
        cols = min(3, n_masks + 1)
        rows = (n_masks + 1 + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.array(axes).flatten() if n_masks > 0 else [axes]
        
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_masks, 1)))
        
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            ax = axes[idx + 1]
            ax.imshow(image)
            
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0] = [*colors[idx][:3], 0.5]
            ax.imshow(colored_mask)
            
            ax.set_title(f"{label}")
            ax.axis('off')
        
        for idx in range(n_masks + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.mask_dir, f"{save_name}_masks.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        combined_mask = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
        for idx, mask in enumerate(masks):
            color = (np.array(colors[idx][:3]) * 255).astype(np.uint8)
            combined_mask[mask > 0] = color
        
        overlay = (image * 0.6 + combined_mask * 0.4).astype(np.uint8)
        cv2.imwrite(os.path.join(self.mask_dir, f"{save_name}_overlay.png"), 
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    def save_camera_trajectory(self, extrinsics: np.ndarray, intrinsics: np.ndarray):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = []
        directions = []
        
        for i, ext in enumerate(extrinsics):
            if ext.shape == (3, 4):
                ext = np.vstack([ext, [0, 0, 0, 1]])
            
            c2w = np.linalg.inv(ext)
            pos = c2w[:3, 3]
            positions.append(pos)
            
            forward = c2w[:3, 2]
            directions.append(forward)
        
        positions = np.array(positions)
        directions = np.array(directions)
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c=np.arange(len(positions)), cmap='viridis', s=100, marker='o')
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', alpha=0.5, linewidth=2)
        
        scale = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.1
        for i, (pos, dir) in enumerate(zip(positions, directions)):
            ax.quiver(pos[0], pos[1], pos[2], 
                      dir[0], dir[1], dir[2], 
                      length=scale, color='red', alpha=0.6)
        
        for i, pos in enumerate(positions):
            ax.text(pos[0], pos[1], pos[2], f'{i}', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualization", "camera_trajectory.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        self._save_camera_top_view(positions, directions)
    
    def _save_camera_top_view(self, positions: np.ndarray, directions: np.ndarray):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.scatter(positions[:, 0], positions[:, 2], 
                   c=np.arange(len(positions)), cmap='viridis', s=100)
        ax.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.5)
        
        scale = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.1
        for pos, dir in zip(positions, directions):
            ax.arrow(pos[0], pos[2], dir[0] * scale, dir[2] * scale, 
                     head_width=scale * 0.1, head_length=scale * 0.05, fc='red', ec='red')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('Camera Trajectory (Top View)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualization", "camera_trajectory_top.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()


class GroundedSAMSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        
        GlobalHydra.instance().clear()
        sam2_config_dir = os.path.abspath("./Grounded-SAM-2/sam2/configs")
        initialize_config_dir(config_dir=sam2_config_dir, version_base=None)
        
        sam2_model = build_sam2(config.SAM2_MODEL_CONFIG, config.SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.gdino_model = load_gdino_model(
            model_config_path=config.GROUNDING_DINO_CONFIG,
            model_checkpoint_path=config.GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )
    
    def segment(self, image_path: str, text_prompt: str) -> dict:
        image_source, image = load_image_gdino(image_path)
        h, w, _ = image_source.shape
        
        boxes, confidences, labels = gdino_predict(
            model=self.gdino_model, image=image, caption=text_prompt,
            box_threshold=self.config.BOX_THRESHOLD,
            text_threshold=self.config.TEXT_THRESHOLD,
            device=self.device
        )
        
        if len(boxes) == 0:
            return {"masks": [], "labels": [], "image": image_source, 
                    "boxes": boxes, "confidences": confidences}
        
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy")
        keep_indices = nms(boxes_xyxy, confidences, iou_threshold=self.config.NMS_THRESHOLD)
        
        boxes_pixel = boxes_pixel[keep_indices]
        boxes_kept = boxes[keep_indices]
        confidences_kept = confidences[keep_indices]
        labels = [labels[i] for i in keep_indices]
        input_boxes = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        self.sam2_predictor.set_image(image_source)
        with torch.autocast(device_type=self.device, dtype=self.config.DTYPE):
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
            )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        valid_indices = [i for i, mask in enumerate(masks) if mask.sum() >= self.config.MIN_MASK_PIXELS]
        if valid_indices:
            masks = masks[valid_indices]
            labels = [labels[i] for i in valid_indices]
            boxes_kept = boxes_kept[valid_indices]
            confidences_kept = confidences_kept[valid_indices]
        else:
            return {"masks": [], "labels": [], "image": image_source,
                    "boxes": torch.tensor([]), "confidences": torch.tensor([])}
        
        return {"masks": masks, "labels": labels, "image": image_source,
                "boxes": boxes_kept, "confidences": confidences_kept}


class VGGTPoseEstimator:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        self.model = self._load_model()
    
    def _load_model(self):
        from vggt.vggt.models.aggregator import Aggregator
        from vggt.vggt.heads.camera_head import CameraHead
        
        vggt_cfg = dict(
            img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4.0, num_register_tokens=4, patch_embed="dinov2_vitl14_reg",
            aa_order=['frame', 'global'], aa_block_size=1, qk_norm=True, rope_freq=100, init_values=0.01,
        )
        
        aggregator = Aggregator(**vggt_cfg).to(self.device)
        token_dim = 2 * vggt_cfg['embed_dim']
        camera_head = CameraHead(dim_in=token_dim).to(self.device)
        
        if os.path.exists(self.config.VGGT_CHECKPOINT):
            ckpt = torch.load(self.config.VGGT_CHECKPOINT, map_location=self.device, weights_only=True)
            model_w = ckpt.get('model', ckpt)
            
            agg_w = {k.replace('aggregator.', ''): v for k, v in model_w.items() if k.startswith('aggregator.')}
            aggregator.load_state_dict(agg_w, strict=False)
            
            cam_w = {k.replace('camera_head.', ''): v for k, v in model_w.items() if k.startswith('camera_head.')}
            if cam_w:
                camera_head.load_state_dict(cam_w, strict=False)
        
        aggregator.eval()
        camera_head.eval()
        return {'aggregator': aggregator, 'camera_head': camera_head}
    
    def estimate_poses(self, images: np.ndarray) -> dict:
        imgs = torch.from_numpy(images.copy()).float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.config.DTYPE):
                agg_tokens, ps_idx = self.model['aggregator'](imgs)
                pose_enc = self.model['camera_head'](agg_tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
        
        return {
            "extrinsic": extrinsic.squeeze(0).cpu().numpy(),
            "intrinsic": intrinsic.squeeze(0).cpu().numpy(),
        }


class GaussianGenerator:
    def __init__(self, config: Config):
        self.inference = Inference(config.GS_CONFIG_PATH, compile=False)
    
    def generate(self, image_path: str, mask: np.ndarray, seed: int = 42):
        image = load_image_gs(image_path)
        if mask.dtype == bool:
            mask_np = mask.astype(np.uint8)
        elif mask.max() > 1:
            mask_np = (mask / 255).astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
        return self.inference(image, np.ascontiguousarray(mask_np), seed=seed)


def flip_yz(matrix):
    flip = np.diag([1, -1, -1, 1]).astype(np.float32)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    return flip @ matrix @ flip


def merge_gaussians_with_camera_poses(outputs_per_image: list, extrinsics: np.ndarray):
    all_outputs = []
    
    for img_idx, outputs in enumerate(outputs_per_image):
        if len(outputs) == 0:
            continue
        
        w2c = extrinsics[img_idx]
        if w2c.shape == (3, 4):
            w2c = np.vstack([w2c, [0, 0, 0, 1]])
        
        w2c = flip_yz(w2c)
        c2w = np.linalg.inv(w2c)
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        
        for output in outputs:
            output = deepcopy(output)
            device = output["translation"].device
            
            T_cam = output["translation"].cpu().numpy()[0]
            T_new = R_c2w @ T_cam + t_c2w
            output["translation"] = torch.from_numpy(T_new[None].astype(np.float32)).to(device)
            
            R_c2w_torch = torch.from_numpy(R_c2w.astype(np.float32)).to(device)
            q_c2w = matrix_to_quaternion(R_c2w_torch.unsqueeze(0))
            q_c2w_inv = quaternion_invert(q_c2w)
            output["rotation"] = quaternion_multiply(output["rotation"], q_c2w_inv)
            
            all_outputs.append(output)
    
    return all_outputs


def process_scene(
    image_paths: list,
    text_prompt: str,
    output_dir: str,
    config: Config = None,
    target_size: tuple = (518, 518)
):
    if config is None:
        config = Config()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing models...")
    segmenter = GroundedSAMSegmenter(config)
    pose_estimator = VGGTPoseEstimator(config)
    gs_generator = GaussianGenerator(config)
    visualizer = Visualizer(output_dir)
    
    print(f"Loading {len(image_paths)} images...")
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        images.append(img_resized)
    images = np.stack(images)
    
    print("Estimating camera poses with VGGT...")
    poses = pose_estimator.estimate_poses(images)
    
    print("Saving camera trajectory visualization...")
    visualizer.save_camera_trajectory(poses["extrinsic"], poses["intrinsic"])
    
    outputs_per_image = []
    all_labels = []
    
    for idx, img_path in enumerate(image_paths):
        print(f"Processing image {idx + 1}/{len(image_paths)}: {img_path}")
        
        seg_result = segmenter.segment(img_path, text_prompt)
        masks = seg_result["masks"]
        labels = seg_result["labels"]
        image = seg_result["image"]
        boxes = seg_result["boxes"]
        confidences = seg_result["confidences"]
        
        img_name = Path(img_path).stem
        
        if len(boxes) > 0:
            visualizer.save_detection(image, boxes, labels, confidences, img_name)
        
        if len(masks) == 0:
            print(f"  No valid objects detected")
            outputs_per_image.append([])
            continue
        
        print(f"  Detected {len(masks)} objects: {labels}")
        visualizer.save_masks(image, masks, labels, img_name)
        
        image_outputs = []
        for mask_idx, (mask, label) in enumerate(zip(masks, labels)):
            print(f"  Generating Gaussian for '{label}'...")
            try:
                raw_output = gs_generator.generate(img_path, mask, seed=42 + mask_idx)
                image_outputs.append(raw_output)
                all_labels.append(f"{label}_img{idx}")
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        outputs_per_image.append(image_outputs)
    
    total_outputs = sum(len(outputs) for outputs in outputs_per_image)
    if total_outputs == 0:
        print("No Gaussians generated!")
        return None
    
    print(f"Merging {total_outputs} Gaussians with camera poses...")
    all_outputs = merge_gaussians_with_camera_poses(outputs_per_image, poses["extrinsic"])
    
    scene_gs = make_scene(*all_outputs)
    
    ply_path = os.path.join(output_dir, "merged_scene.ply")
    scene_gs.save_ply(ply_path)
    print(f"Saved PLY to: {ply_path}")
    
    metadata = {
        "image_paths": image_paths,
        "text_prompt": text_prompt,
        "objects": all_labels,
        "num_images": len(image_paths),
        "num_objects": total_outputs,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nVisualization outputs saved to: {os.path.join(output_dir, 'visualization')}")
    
    return ply_path


def main():
    IMAGE_DIR = "./data/class1"
    TEXT_PROMPT = "desk."
    OUTPUT_DIR = "outputs/merged_scene"
    
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    
    print(f"Found {len(image_paths)} images")
    
    ply_path = process_scene(
        image_paths=image_paths,
        text_prompt=TEXT_PROMPT,
        output_dir=OUTPUT_DIR,
    )
    
    if ply_path:
        interactive_visualizer(ply_path)


if __name__ == "__main__":
    main()