import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.models import UNet, ImageSegmentationModel
from pipeline.modeling.dataset import SegmentationDataset, Transform
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = False
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/selected_data_split/train/"
TRAIN_MASK_DIR = "data/selected_data_split/train/"
VAL_IMG_DIR = "data/selected_data_split/val/"
VAL_MASK_DIR = "data/selected_data_split/val/"

def main():

    # create train and validation dataset
    train_dataset = SegmentationDataset(
        image_dir=TRAIN_IMG_DIR,
        transform=Transform(),
    )
    val_dataset = SegmentationDataset(
        image_dir=VAL_IMG_DIR,
        transform=Transform(),
    )
    # create train and validation dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    model = UNet(in_channels=3, out_channels=5)
    # train model
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        segmentation_model = ImageSegmentationModel(
            model=model,
            in_channels=3,
            out_channels=5,
            num_classes=5,
            feature_nums=[64, 128, 256, 512],
            learning_rate=LEARNING_RATE,
            optimizer=optim.Adam,
            loss_fn=nn.CrossEntropyLoss(),
        )
        train_loss, train_iou = segmentation_model.train_val_epoch(train_loader, mode="train")
        val_loss, val_iou = segmentation_model.train_val_epoch(val_loader, mode="val")
        print(f"Training Loss: {train_loss:.4f} | Training IoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}")
        print("---------------------------------------")

if __name__ == "__main__":
    main()