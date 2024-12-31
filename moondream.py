from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

class Moondream:
    def __init__(self):
        pass
    
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
        # change tensor to PIL Image
        pil_image = Image.fromarray((image[0] * 255).numpy().astype('uint8'))
        
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

        enc_image = model.encode_image(pil_image)
        return (model.answer_question(enc_image, "Describe this image.", tokenizer),)
