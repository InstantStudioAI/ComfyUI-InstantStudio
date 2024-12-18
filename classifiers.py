import os
import folder_paths
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return

def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(0), 0, 255).astype(np.uint8))

def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")

class HFClassify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_classifier_model": ("TRANSFORMERS_CLASSIFIER",),
                "image_to_classify": ("IMAGE",),
                "target_label": (["male", "female"],),  # Toggle for selecting 'male' or 'female'
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOL")  # Added BOOL for the new boolean output
    RETURN_NAMES = ("raw_score", "rounded_score", "is_above_threshold")  # Added name for the boolean output
    FUNCTION = "classify"

    def classify(self, hf_classifier_model, image_to_classify, target_label):
        # Convert image to PIL format
        image = to_pil(image_to_classify)

        # Get classification result
        result = hf_classifier_model(image)

        # Extract the score for the target label
        target_result = next((entry for entry in result if entry["label"] == target_label), None)
        if target_result:
            score = target_result["score"]
        else:
            # If the target label isn't in the results, default score to 0
            score = 0.0

        # Round the score to 0 or 1
        rounded_score = int(round(score))

        # Determine if the score is above the threshold of 0.5
        is_above_threshold = score > 0.5

        # Return the raw score, rounded score, and the boolean indicating if above threshold
        return (score, rounded_score, is_above_threshold)
