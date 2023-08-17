# gradio is used to uild a ui
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry



device="cuda"
sam_checkpoint="weights/sam_vit_h_4b8939.pth"
model_type="vit_h"

sam=sam_model_registry[model_type](checkpoint=sam_checkpoint)

sam.to(device)
# initiating the predictor class
predictor=SamPredictor(sam)

# now we need stable diffusion inpaint pipeline

pipe=StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",  # model name
    torch_dtype=torch.float16
)
# now sending the pipeline to cuda

pipe=pipe.to(device)

selected_pixels=[]

with gr.Row as demo:
    
    # this is one row in which i get these three components
    with gr.Row():
        input_img=gr.Image(label="Input Image")
        mask_img=gr.Image(label="Mask image")
        output_img=gr.Image(label="Output_Image")
    
    # in other row we will have promt text
    with gr.Row():
        prompt_text=gr.Textbox(lines=1,label="Type your prompt here")
    
    # in other row we will have a button
    with gr.Row():
         submit=gr.Button(label="Submit")

    def generate_mask(image,event:gr.SelectData):
        # so this will store the current selected pixels
        selected_pixels.append(event.index)

        # now we will tel the sam predictor to
        predictor.set_image(image)

        input_points=np.array(selected_pixels)

        input_labels=np.ones(input_points.shape[0])
        mask,_,_=predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        # (n,sz,sz)
        # covertig mask to a pil image
        mask=np.logical_not(mask)  # use it to change the background
        mask=Image.fromarray(mask[0,:,:])
        return mask




# so inpaint function will take a image,mask and a prompt and will return a output image and we will use stable diffusion 2.1
    def inpaint(image,mask,prompt): # these are numpy arrays
        # converting the image to pil image
        image=Image.fromarray(image)
        mask=Image.fromarray(mask)

        # resizing the images to 512,512
        image=image.resize((512,512))
        mask=mask.resize((512,512))

        output=pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask
                    ).images[0]
        return output

# now defining click event 
    input_img.select(generate_mask,[input_img],[mask_img])
    # gettig the imapinting thing

    submit.click(inpaint,inputs=[input_img,mask_img,prompt_text],outputs=[output_img],) 


if __name__=="__main__":
    demo.launch()


# Working of Sampredictor
# we have to pass the x,y cordinate values for an image and one label, we can increase the no of points keeping the
# label same

    

