from flask import Flask, render_template, request, redirect
from PIL import Image
import io
import os
import base64
import numpy as np
from utils.general_utils import load_model, predict_image
from pipeline.modeling.models import (
    UNet,
    ImageSegmentationModel,
    UNetWithResnet50Encoder,
)

app = Flask(__name__)

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
    colormap = [
        (128, 128, 128), # Background (label 0) - Black
        (128, 0, 0),     # Building (Label 1) - Dark Red
        (255, 165, 0), # Road (Label 2) - Gray
        (0, 0, 255),     # Water (Label 3) - Blue
        (255, 255, 127),   # Barren (Label 4) - Cream Yellow
        (0, 128, 0),     # Forest (Label 5) - Dark Green
        (185, 255, 71)  # Agriculture (Label 6) - Lime Green
    ]
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.convert('P')
    mask.putpalette([c for rgb in colormap for c in rgb])
    mask = mask.convert('RGBA')  # Convert to 'RGBA' mode
    mask.putalpha(180)

    return mask

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']

        # Check if the user submitted an empty form
        if file.filename == '':
            return redirect(request.url)

        # Read the image file
        image = Image.open(file).convert('RGB')

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

        return render_template('result.html', original_image=encoded_image, processed_image=encoded_mask, area=area)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)