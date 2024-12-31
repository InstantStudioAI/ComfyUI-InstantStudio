from .upload_images import UploadImages
from .classifiers import *
from .moondream import Moondream

NODE_CLASS_MAPPINGS = {
    "UploadImagesToInstantStudio": UploadImages,
    'HuggingFace Classify': HFClassify,
    'Moondream': Moondream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadImagesToInstantStudio": "Upload Images to Instant Studio",
    "Moondream": "Moondream (Vision LM to analyze images)"
}
