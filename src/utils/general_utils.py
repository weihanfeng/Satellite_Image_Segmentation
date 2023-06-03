import logging
import logging.config
import os
import yaml
import torch
import cv2
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T


logger = logging.Logger(__name__)

def setup_logging(
    logging_config_path="./conf/base/logging.yaml", default_level=logging.INFO
):
    """Set up configuration for logging utilities.

    Args:
        logging_config_path : str, optional
            Path to YAML file containing configuration for Python logger,
            by default "./config/logging_config.yaml"
        default_level : logging object, optional, by default logging.INFO
    """

    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")

def get_num_files(path):
    """Get number of files in extracted folder"""
    return len(os.listdir(path))

def save_model(checkpoint, model_dir):
    """Save model checkpoint"""
    logging.info("Saving model checkpoint")
    torch.save(checkpoint, model_dir)

def load_model(model_dir, cpu=False):
    """Load model checkpoint"""
    logging.info("Loading model checkpoint")
    if cpu:
        model = torch.load(model_dir, map_location=torch.device("cpu"))
    else:
        model = torch.load(model_dir)
    return model

def _pad_img(img, window_size):
    """
    pad image on the right and bottom to make it divisible by window size
    """
    h, w, c = img.shape
    pad_h = 0
    pad_w = 0
    if h % window_size != 0:
        pad_h = window_size - (h % window_size)
    if w % window_size != 0:
        pad_w = window_size - (w % window_size)
    img = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    return img, pad_h, pad_w

def _unpad_mask(img, pad_h, pad_w):
    """
    unpad mask
    """
    h, w = img.shape
    img = img[: h - pad_h, : w - pad_w]
    return img

def predict_image(image, model, patch_size):
    """
    Predict image mask using sliding window.
    Image is a numpy array of shape (h, w, c).
    """
    # pad image
    image, pad_h, pad_w = _pad_img(image, patch_size)
    # patchify image
    patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
    patch_shape = patches.shape
    # predict patches
    patches = patches.reshape(-1, patch_size, patch_size, 3)
    patches = torch.stack(
        [
            T.ToTensor()(np.squeeze(np.expand_dims(image, axis=0), axis=0))
            for image in patches
        ]
    )
    predictions = model(patches)
    predictions = torch.permute(predictions, (0, 2, 3, 1))
    predictions = F.softmax(predictions, dim=3)
    predictions = torch.argmax(predictions, dim=3)
    predictions = predictions.numpy()
    # reshape into patch_shape
    predictions = predictions.reshape(patch_shape[0], patch_shape[1], patch_size, patch_size)
    # unpatchify
    predictions = unpatchify(predictions, image.shape[:-1])
    # unpad
    predictions = _unpad_mask(predictions, pad_h, pad_w)
    # prediction is a numpy array of shape (h, w)

    return predictions

def view_image_and_mask(image_path, mask_path):
    """
    View image and mask
    """
    if not os.path.exists(image_path):
        print("Image path does not exist")
        return None
    image = cv2.imread(image_path, 1)
    mask = cv2.imread(mask_path)
    fig, ax = plt.subplots(1,2, figsize=(10, 10))
    ax[0].imshow(image)
    ax[1].imshow(mask[:,:,0])
    plt.show()
