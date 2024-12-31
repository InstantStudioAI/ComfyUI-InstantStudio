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
                "prompt": ("STRING", {"multiline": True, "default": "Please provide a detailed description of this image."},),
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
        
        descriptions = ""
        prompts = list(filter(lambda x: x!="", [s.lstrip() for s in prompt.splitlines()])) # make a prompt list and remove unnecessary whitechars and empty lines
        if len(prompts) == 0:
            prompts = [""]

        try:
            for im in image:
                i = 255. * im.cpu().numpy()
                img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8))
                enc_image = self.model.encode_image(img)

                descr = ""
                sep = codecs.decode(separator, 'unicode_escape')
                for p in prompts:
                    answer = self.model.answer_question(enc_image, p, self.tokenizer, temperature=None, do_sample=None)
                    descr += f"{answer}{sep}"
                descriptions += f"{descr[0:-len(sep)]}\n"
        except RuntimeError:
            raise ValueError(f"Error loading model {model_id} revision {revision}")
        
        return(descriptions[0:-1],)
