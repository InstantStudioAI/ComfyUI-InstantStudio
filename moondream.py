"""
  This node is a comfyui node for the moondream visual LLM (https://huggingface.co/vikhyatk/moondream2)
  It is customized through the following project:  https://github.com/Hangover3832/ComfyUI-Hangover-Moondream
"""

from transformers import AutoModelForCausalLM as AutoModel
# Try to import multiple possible tokenizers to improve compatibility
try:
    from transformers import CodeGenTokenizerFast as Tokenizer
except ImportError:
    try:
        from transformers import AutoTokenizer as Tokenizer
        print("[Moondream] Use AutoTokenizer instead of CodeGenTokenizerFast")
    except ImportError:
        from transformers import PreTrainedTokenizerFast as Tokenizer
        print("[Moondream] Use PreTrainedTokenizerFast instead of CodeGenTokenizerFast")

from PIL import Image
import torch
import gc
import numpy as np
import codecs
import subprocess
import os
import requests
import sys
import traceback

def Run_git_status(repo:str) -> list[str]:
    """resturns a list of all model tag references for this huggingface repo"""
    url = f"https://huggingface.co/{repo}"
    process = subprocess.Popen(['git', 'ls-remote', url], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    result = []
    if process.returncode == 0:
        revs = stdout.decode().splitlines()
        revs = [r.replace('main', 'latest') for r in revs if ('/tags' in r) or ('/main' in r)]
        for line in revs:
            rev = line.split('\t')
            result.append(f"{rev[-1].split('/')[-1]} -> {rev[0]}")
    return result

class Moondream:
    HUGGINGFACE_MODEL_NAME = "vikhyatk/moondream2"
    DEVICES = ["gpu", "cpu"] if torch.cuda.is_available() else  ["cpu"]
    Versions = 'versions.txt'
    Model_Revisions_URL = f"https://huggingface.co/{HUGGINGFACE_MODEL_NAME}/raw/main/{Versions}"
    current_path = os.path.abspath(os.path.dirname(__file__))
    try:
        print("[Moondream] trying to update model versions...", end='')
        response = requests.get(Model_Revisions_URL)
        if response.status_code == 200:
            with open(f"{current_path}/{Versions}", 'w') as f:
                f.write(response.text)
            print('ok')
    except Exception as e:
        if hasattr(e, 'message'):
            msg = e.message
        else:
            msg = e
        print(f'failed ({msg})')

    with open(f"{current_path}/{Versions}", 'r') as f:
        versions = f.read()
    
    MODEL_REVISIONS = [v for v in versions.splitlines() if v.strip()]
    print(f"[Moondream] found model versions: {', '.join(MODEL_REVISIONS)}")
    MODEL_REVISIONS.insert(0,'ComfyUI/models/moondream2')

    try:
        print('\033[92m\033[4m[Moondream] model revsion references:\033[0m\033[92m')
        git_status = Run_git_status(HUGGINGFACE_MODEL_NAME)
        for s in git_status:
            print(s)
        # return ("",)
    except:
        pass
    finally:
        print('\033[0m')


    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.revision = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "What is the gender, the age range, hair color, and hairstyle of the person in the image? Respond with only these four attributes."},),
                "separator": ("STRING", {"multiline": False, "default": r", "},),
                "model_revision": (s.MODEL_REVISIONS, {"default": s.MODEL_REVISIONS[-1]},),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.},),
                "device": (s.DEVICES, {"default": s.DEVICES[0]},)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "interrogate"
    OUTPUT_NODE = False
    CATEGORY = "InstantStudio"

    def interrogate(self, image:torch.Tensor, prompt:str, separator:str, model_revision:str, temperature:float, device:str):

        dev = "cuda" if device.lower() == "gpu" else "cpu"
        if temperature < 0.01:
            temperature = None
            do_sample = None
        else:
            do_sample = True

        if (self.model == None) or (self.tokenizer == None) or (device != self.device) or (model_revision != self.revision):
            del self.model
            del self.tokenizer
            gc.collect()
            if (device == "cpu") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.revision = model_revision

            print(f"[Moondream] loading model moondream2 revision '{model_revision}', please stand by....")
            if model_revision == Moondream.MODEL_REVISIONS[0]:
                model_name = model_revision
                model_revision = None
            else:
                model_name = Moondream.HUGGINGFACE_MODEL_NAME

            try:
                self.model = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    revision=model_revision
                ).to(dev)

                try:
                    # Try to load the tokenizer
                    self.tokenizer = Tokenizer.from_pretrained(model_name)
                except Exception as e:
                    # Try to load the tokenizer using AutoTokenizer if the default one fails
                    from transformers import AutoTokenizer
                    print(f"[Moondream] Failed to load tokenizer, trying to use AutoTokenizer... ({str(e)})")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                import transformers
                error_msg = f"[Moondream] Failed to load model. transformers version: {transformers.__version__}\nError details: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                raise ValueError(f"[Moondream] Please check if the transformers package meets the requirements. "
                                "Note: older models may no longer be compatible with newer versions of the package.\n"
                                f"Error details: {str(e)}")
            print(f"[Moondream] model moondream2 revision '{model_revision}' loaded successfully")

            self.device = device

        descriptions = ""
        prompts = list(filter(lambda x: x!="", [s.lstrip() for s in prompt.splitlines()])) # make a prompt list and remove unnecessary whitechars and empty lines
        if len(prompts) == 0:
            prompts = [""]

        try:
            for im in image:
                i = 255. * im.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                enc_image = self.model.encode_image(img)
                descr = ""
                sep = codecs.decode(separator, 'unicode_escape')
                for p in prompts:
                    answer = self.model.answer_question(enc_image, p, self.tokenizer, temperature=temperature, do_sample=do_sample)
                    descr += f"{answer}{sep}"
                descriptions += f"{descr[0:-len(sep)]}\n"
        except Exception as e:
            error_msg = f"[Moondream] Model execution failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            raise ValueError(f"[Moondream] Model execution failed: {str(e)}")
        
        return(descriptions[0:-1],)