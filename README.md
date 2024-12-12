# ComfyUI-InstantStudio Node Pack

A collection of nodes to enhance your experience with ComfyUI.

## Nodes in This Pack

### 1. **Upload Images Node**
The `upload_images` node allows you to upload images to the **InstantStudio Toolkit**. It requires:
- **Username**
- **Toolkit Endpoint**
- **API Code**

### 2. **Classifier Node**
The `classifier` node works with the **HF Transformers Classifier Provider**, a feature provided by the [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) by **ltdrdata**. This node:
- Utilizes **HuggingFace's Transformers models** for classification.
- Currently supports classifying images as **male** or **female**.
- Enables workflows based on the gender classification of a user's input image.

---
## Install

To install the ComfyUI-InstantStudio Node Pack:

### Manual Installation
1. Navigate to your `custom_nodes` directory:
   ```bash
   cd custom_nodes
   ```

### Clone and install requirements:
  ```bash
  git clone https://github.com/InstantStudioAI/ComfyUI-InstantStudio.git
  cd comfy_instant_studio
  pip install -r requirements.txt
  ```

### Restart ComfyUI
