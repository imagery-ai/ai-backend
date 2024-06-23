import base64
import sys
print("SYSPATH", sys.path)
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn
from leditspp import StableDiffusionPipeline_LEDITS
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
import PIL
import io
import numpy as np
import torch
from fastapi.responses import JSONResponse


# run with:
# python -m uvicorn main:app --reload


app = FastAPI()


class ImageInput(BaseModel):
    image: bytes


class PromptInput(BaseModel):
    prompt: str


@app.on_event("startup")
async def load_model():
    print("Loading model...")
    # self.model_name = "stabilityai/stable-diffusion-3-medium-diffusers"  # Model name
    model_name = "runwayml/stable-diffusion-v1-5"  # Model name
    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(
        model_name,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        model_name,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
    )
    pipe = pipe.to(
        "mps"
    )  # TODO: CHANGE THIS DEPENDING ON HARDWARE (mps, cuda, intel)

    app.state.pipe = pipe
    app.state.image = None

    print("Model loaded successfully.")


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    if not hasattr(app.state, "pipe"):
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    image = PIL.Image.open(io.BytesIO(contents))
    app.state.image = image
    print("Image received and stored.")
    return {"message": "Image uploaded successfully."}


@app.post("/generate_image/")
async def generate_image(editing_prompt: list[str], reverse_editing_direction: list[bool]):
    if not hasattr(app.state, "pipe"):
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if app.state.image is None:
        raise HTTPException(status_code=400, detail="Image not set.")

    pipe = app.state.pipe

    im = np.array(app.state.image)[:, :, :3]
    gen = torch.manual_seed(42)
    with torch.no_grad():
        _ = pipe.invert(
            im, num_inversion_steps=50, generator=gen, verbose=True, skip=0.15
        )
        edited_image = pipe(

            editing_prompt=editing_prompt,
            edit_threshold=[0.8] * len(editing_prompt),
            edit_guidance_scale=[6] * len(editing_prompt),
            reverse_editing_direction=reverse_editing_direction,
            use_intersect_mask=True,
        )
    app.state.image = edited_image.images[0]
    buffered = io.BytesIO()
    app.state.image.save(buffered, format="JPEG")
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return JSONResponse(content={"image": image_data})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
