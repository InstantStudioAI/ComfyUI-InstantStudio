from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

class Moondream:
    def __init__(self):
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-08-26"
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                revision=self.revision,
                torch_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision
            )
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Image Description",)

    FUNCTION = "describe_image"

    CATEGORY = "image/utils"

    def describe_image(self, image):
        self.load_model()
        
        pil_image = Image.fromarray((image[0] * 255).numpy().astype('uint8'))
        
        with torch.inference_mode():
            enc_image = self.model.encode_image(pil_image)
            return (self.model.answer_question(enc_image, "Describe this image.", self.tokenizer),)
