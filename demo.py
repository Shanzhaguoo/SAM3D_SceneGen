import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_masks

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
PATH = os.getcwd()
print("Current working directory:", PATH)
IMAGE_PATH = f"{PATH}/images/shutterstock_stylish_kidsroom_1640806567/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))


image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)
# masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")


# run model
output = inference(image, mask, seed=42)
# outputs = [inference(image, mask, seed=42) for mask in masks]
print(output.keys())
# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
