# Object Wizard
Object Wizard is a Python-based project that leverages the power of computer vision models and image inpainting techniques to interactively edit and enhance images. This project allows you to seamlessly manipulate images by selecting pixels of interest, applying inpainting techniques, and generating creative outputs. The core components of the project include the Segment Anything Model (SAM) for pixel-wise image segmentation and the Stable Diffusion Inpainting Pipeline for image inpainting.

# Prerequisites
Before you get started, make sure you have the following dependencies installed:

Python (3.6 or higher)
gradio library (pip install gradio)
numpy library (pip install numpy)
torch library (pip install torch)
diffusers library (install instructions: Diffusers GitHub)
PIL (Python Imaging Library) (pip install pillow)
segment_anything library (install instructions: Segment Anything GitHub)
# Setup and Usage
Clone or download the Object Wizard repository to your local machine.

Install the required libraries mentioned in the prerequisites.

Run the object_wizard.py script to launch the interactive GUI.


# code
```
pip install -r ./requirements.txt
python3 ./app.py
```
python3 object_wizard.py
The GUI will provide you with the following components:

# Input Image: Upload an image you want to edit.
Mask Image: Select pixels on the input image to create a mask using the Segment Anything Model (SAM).
Output Image: View the enhanced output image after applying inpainting.
Additionally, you can provide a text prompt in the Type your prompt here textbox to guide the inpainting process.

Click the Submit button to generate the enhanced output image using the Stable Diffusion Inpainting Pipeline.

# How It Works
The project uses the Segment Anything Model (SAM) to segment and label pixels in the input image. You can select pixels of interest to create a mask.

The Stable Diffusion Inpainting Pipeline is employed to inpaint the selected region of the image, resulting in a visually appealing and coherent output.

