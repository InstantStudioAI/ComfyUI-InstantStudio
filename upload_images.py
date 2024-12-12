import os
import json
from comfy.cli_args import args
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import requests

class UploadImages:
    def __init__(self):
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to upload."}),
                "note": ("STRING", {"tooltip": "A note to associate with the image."}),
                "email": ("STRING", {"tooltip": "The email to use for authentication."}),
                "url": ("STRING", {"default": "https://toolkit.instantstudio.ai/api/v1/creations", "tooltip": "The URL to post the image to."}),
                "api_key": ("STRING", {"default": "TODO", "tooltip": "The API Key to use for authentication."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "upload_images"
    OUTPUT_NODE = True
    CATEGORY = "Instant Studio"
    DESCRIPTION = "Uploads the input images to Instant Studio."

    def upload_images(self, images, note=None, email=None, url=None, api_key=None, prompt=None, extra_pnginfo=None):
        uuids = None

        succeeded_count = 0
        failed_count = 0
        temp_file_paths = list()

        try:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        for key in prompt:
                            if "inputs" in prompt[key] and "api_key" in prompt[key]["inputs"]:
                                prompt[key]["inputs"]["api_key"] = "TODO"
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        if "workflow" in extra_pnginfo and "nodes" in extra_pnginfo["workflow"]:
                            for node in extra_pnginfo['workflow']['nodes']:
                                if node['type'] == 'UploadImagesToInstantStudio':
                                    node['widgets_values'] = []
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                temp_file = None

                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file_path = temp_file.name
                    img.save(temp_file_path, pnginfo=metadata, compress_level=self.compress_level)

                    temp_file_paths.append(temp_file_path)

            files = [("files[]", open(temp_file_path, "rb")) for temp_file_path in temp_file_paths]
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-User-Email": email
            }
            params = {
                "note": note,
            }
            response = requests.post(url, headers=headers, params=params, files=files)
            if response.status_code != 200:
                failed_count += 1
                raise Exception(f"Failed to post image: {response.status_code} - {response.text}")
            else:
                succeeded_count += 1
                response_data = response.json()
                uuids = response_data.get("uuids")

        finally:
            for temp_file_path in temp_file_paths:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        print(f"Successfully uploaded {succeeded_count} images to Instant Studio: {uuids}")
        return { "ui": { "uuids": uuids }, "result": (uuids,) }
