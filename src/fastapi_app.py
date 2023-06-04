from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os
import base64
import numpy as np
from .utils.general_utils import load_model, predict_image
from .pipeline.modeling.models import (
    UNet,
    ImageSegmentationModel,
    UNetWithResnet50Encoder,
)

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Function to process the image
def get_mask(image):
    """Get the mask of the image"""
    # Load the model
    model = UNetWithResnet50Encoder(
        last_n_layers_to_unfreeze=2,
        n_classes=7,
    )
    checkpoint = load_model("models/UNetWithResnet50Encoder_unfreeze2.pth", cpu=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Convert image to numpy
    image_array = np.array(image)
    # Get the mask
    mask = predict_image(image=image_array, model=model, patch_size=256)
    # Convert mask to PIL image
    colormap = {
        0: (128, 128, 128), # Background (label 0) - Black
        1: (128, 0, 0),     # Building (Label 1) - Dark Red
        2: (255, 165, 0), # Road (Label 2) - Gray
        3: (0, 0, 255),     # Water (Label 3) - Blue
        4: (255, 255, 127),   # Barren (Label 4) - Cream Yellow
        5: (0, 88, 0),     # Forest (Label 5) - Dark Green
        6: (185, 255, 71)  # Agriculture (Label 6) - Lime Green
    }
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.convert('P')
    mask.putpalette([c for rgb in tuple(colormap.values()) for c in rgb])
    mask = mask.convert('RGBA')  # Convert to 'RGBA' mode
    mask.putalpha(180)

    return mask

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/")
async def upload_image(request: Request):
    formdata = await request.form()
    image = formdata["image"].file

    # Read the image file
    image = Image.open(image).convert('RGB')

    # Get image mask prediction
    mask = get_mask(image)

    # Overlay the mask on the original image
    overlayed_image = Image.alpha_composite(image.convert('RGBA'), mask)

    # Create an in-memory file to save the original and processed image
    # mask
    processed_file = io.BytesIO()
    overlayed_image.save(processed_file, format='PNG')
    processed_file.seek(0)
    # image
    original_file = io.BytesIO()
    image.save(original_file, format='PNG')
    original_file.seek(0)

    # Encode the original and processed image as base64
    encoded_image = base64.b64encode(original_file.getvalue()).decode('utf-8')
    encoded_mask = base64.b64encode(processed_file.getvalue()).decode('utf-8')

    area = "placeholder"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "original_image": encoded_image,
            "processed_image": encoded_mask,
            "area": area,
        },
    )