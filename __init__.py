from .upload_images import UploadImages
from .classifiers import *


NODE_CLASS_MAPPINGS = {
    "UploadImagesToInstantStudio": UploadImages,
    'HuggingFace Classify': HFClassify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadImagesToInstantStudio": "Upload Images to Instant Studio"
}
