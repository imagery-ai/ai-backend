# Auxilary methods and imports
import torch
import intel_extension_for_pytorch as ipex
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from typing import Union
import PIL
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import  StableDiffusionPipeline_LEDITS

def load_image(image: Union[str, PIL.Image.Image]):
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def image_grid(imgs, rows, cols, spacing = 20):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size

    grid = Image.new('RGBA', size=(cols * w + (cols-1)*spacing, rows * h + (rows-1)*spacing ), color=(255,255,255,0))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=( i // rows * (w+spacing), i % rows * (h+spacing)))
        #print(( i // rows * w, i % rows * h))
    return grid

anime = 0 #0

if anime:
    model = 'hakurei/waifu-diffusion'
    org = load_image('/home/jpark/hackathon/anime_girl.png').resize((512,512))
else:
    model = 'runwayml/stable-diffusion-v1-5'
    org = load_image('https://github.com/ml-research/ledits_pp/blob/main/examples/images/yann-lecun.jpg?raw=true').resize((512,512))

device = 'xpu'

pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model,safety_checker = None,)
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(model, subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)
pipe.to(device)

im = np.array(org)[:, :, :3]

gen = torch.manual_seed(42)
with torch.no_grad():
    _ = pipe.invert(im, num_inversion_steps=50, generator=gen, verbose=True, skip=0.15)
    out = pipe(editing_prompt=['angry face'],
               edit_threshold=[.9],
               edit_guidance_scale=[3],
               reverse_editing_direction=[False ,False, False],
               use_intersect_mask=True,)
image_grid((org, out.images[0]), 1, 2)
im1 = org.save("original.jpg")  
im2 = out.images[0].save("result.jpg") 
