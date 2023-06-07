from flask import Flask, render_template, request, redirect
from PIL import Image
import io
import base64
import numpy as np
import logging
from utils.general_utils import load_model, predict_image, setup_logging
from pipeline.modeling.models import (
    UNet,
    UNetWithResnet50Encoder,
)

# import rasterio
# from rasterio.io import MemoryFile
from hydra import compose, initialize
import os

setup_logging()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Hydra config
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config")
        # set torch_home to the path where the models are stored
        os.environ["TORCH_HOME"] = "models/cache"
        # Load the model
        model = UNetWithResnet50Encoder(
            last_n_layers_to_unfreeze=2,
            n_classes=7,
        )
        # Check if model checkpoint exists
        if os.path.exists(cfg["api"]["MODEL_PATH"]):
            logging.info("Loading model from local")
            checkpoint = load_model(
                model_dir=cfg["api"]["MODEL_PATH"],
                cpu=cfg["api"]["CPU"],
                source="local",
            )
        else:
            logging.info(f"Loading model from {cfg['api']['MODEL_SOURCE']}")
            checkpoint = load_model(
                model_dir=cfg["api"]["MODEL_PATH"], 
                cpu=cfg["api"]["CPU"],
                source=cfg["api"]["MODEL_SOURCE"],
                gdrive_id=cfg["api"]["GDRIVE_ID"],
                )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Check if the POST request has the file part
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]

        # Check if the user submitted an empty form
        if file.filename == "":
            return redirect(request.url)

        # Read the image file
        image = Image.open(file).convert("RGB")

        # Get image mask prediction
        logging.info(f"Predicting mask for image: {file.filename}")
        mask = get_mask(
            image=image, 
            model=model,
            file=file, 
            cfg=cfg,
            )

        # Overlay the mask on the original image
        overlayed_image = Image.alpha_composite(image.convert("RGBA"), mask)

        # Create an in-memory file to save the original and processed image
        encoded_mask = save_in_memory(overlayed_image, format="PNG")
        encoded_image = save_in_memory(image, format="PNG")

        # area = "placeholder"

        return render_template(
            "result.html",
            original_image=encoded_image,
            processed_image=encoded_mask,
            # area=area,
        )

    return render_template("upload.html")


def get_mask(image, model, file, cfg):
    """Get the mask of the image and its area

    Args:
        image (PIL.Image): image
        cfg (DictConfig): hydra config

    Returns:
        PIL.Image: mask in RGBA format
    """
    # Convert image to numpy
    image_array = np.array(image)

    # Get the mask and set alpha channel
    mask = predict_image(
        image=image_array, model=model, patch_size=cfg["api"]["PATCH_SIZE"]
    )

    # # If file is of type '.tif', calculate the area of the mask
    # if file.filename.endswith(".tif"):
    #     area_map = calculate_area(mask=mask, file=file, area_map=cfg["api"]["AREA_MAP"])
    #     logging.info(area_map)

    colormap = cfg["api"]["COLORMAP"]
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.convert("P")
    mask.putpalette([c for rgb in tuple(colormap.values()) for c in rgb])
    mask = mask.convert("RGBA")  # Convert to 'RGBA' mode
    mask.putalpha(cfg["api"]["ALPHA_VALUE"])

    return mask


def save_in_memory(file_to_save, format="PNG"):
    """Save file in memory

    Args:
        file (str): file name

    Returns:
        str: file in memory
    """
    tmp_file = io.BytesIO()
    file_to_save.save(tmp_file, format=format)
    tmp_file.seek(0)
    encoded_file = base64.b64encode(tmp_file.getvalue()).decode("utf-8")

    return encoded_file


# def calculate_area(
#     mask,
#     file,
#     area_map,
# ):
#     """Calculate the area of the mask if the file is of type '.tif'

#     Args:
#         mask (np.array): mask
#         file (str): file name
#         area_map (dict): area map

#     Returns:
#         str: area
#     """
#     with MemoryFile(file) as memfile:
#         with memfile.open() as dataset:

#          return type(dataset)

if __name__ == "__main__":
    # app.jinja_env.auto_reload = True
    # app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(debug=True, host="0.0.0.0")
