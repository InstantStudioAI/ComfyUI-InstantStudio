from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import codecs
import numpy
import gc

class Moondream:
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": """
What is the gender of the person in the image? like: boy, girl, man, woman
What is the age range of the person in the image? like: 3-6 years old
What is the hair color of the person in the image? like: dark hair, light hair
What is the hair style of the person in the image? like: straight hair, curly hair, pageboy cut
"""},),
                "separator": ("STRING", {"multiline": False, "default": r"\n"},),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)

    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "image/utils"

    def interrogate(self, image:torch.Tensor, prompt:str, separator:str):
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"

        if (self.model == None) or (self.tokenizer == None):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    revision=revision
                ).to('cuda')
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            except RuntimeError:
                raise ValueError(f"Error loading model {model_id} revision {revision}")
        
        prompts = list(filter(lambda x: x!="", [s.lstrip() for s in prompt.splitlines()]))
        if len(prompts) == 0:
            prompts = [""]

        try:
            i = 255. * image[0].cpu().numpy()
            img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))
            enc_image = self.model.encode_image(img)

            descriptions = []
            sep = codecs.decode(separator, 'unicode_escape')
            for p in prompts:
                answer = self.model.answer_question(enc_image, p, self.tokenizer, temperature=None, do_sample=None)
                answer = answer.lower().strip()
                descriptions.append(answer)
        except RuntimeError:
            raise ValueError(f"Error loading model {model_id} revision {revision}")
        
        return ", ".join(descriptions)