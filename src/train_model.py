import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.models import (
    UNet,
    ImageSegmentationModel,
    UNetWithResnet50Encoder,
)
from pipeline.modeling.dataset import SegmentationDataset, Transform
from utils import setup_logging, save_model, load_model
from torch.utils.data import DataLoader
import torch.multiprocessing
import hydra
from omegaconf import DictConfig
import logging

torch.multiprocessing.set_sharing_strategy("file_system")

setup_logging()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # create train and validation dataset
    logging.info("Starting model training...")
    logging.info(f"Creating train and validation dataloader...")
    train_dataset = SegmentationDataset(
        image_dir=cfg["files"]["TRAIN_IMG_DIR"],
        transform=Transform(),
    )
    val_dataset = SegmentationDataset(
        image_dir=cfg["files"]["VAL_IMG_DIR"],
        transform=Transform(),
    )
    # create train and validation dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["dataloader"]["BATCH_SIZE"],
        num_workers=cfg["dataloader"]["NUM_WORKERS"],
        pin_memory=cfg["dataloader"]["PIN_MEMORY"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["dataloader"]["BATCH_SIZE"],
        num_workers=cfg["dataloader"]["NUM_WORKERS"],
        pin_memory=cfg["dataloader"]["PIN_MEMORY"],
        shuffle=False,
    )

    # Load model a
    # model = UNet(
    #     in_channels=cfg["model"]["IN_CHANNELS"],
    #     out_channels=cfg["model"]["OUT_CHANNELS"],
    # )
    model = UNetWithResnet50Encoder(
        n_classes=5,
    )
    optimizer = optim.Adam
    last_epoch = 0
    if cfg["model"]["LOAD_MODEL"]:
        logging.info("Loading model...")
        model_artifact = load_model(cfg["files"]["MODEL_READ_PATH"])
        model.load_state_dict(model_artifact["state_dict"])
        last_epoch = model_artifact["epoch"]
        # optimizer.load_state_dict(model_artifact["optimizer"])

    # train model
    logging.info(f"Model architecture: {model.__class__.__name__}")
    logging.info("Start training...")
    best_loss = float("inf")
    total_epoch = last_epoch + cfg["model"]["NUM_EPOCHS"]
    for epoch in range(last_epoch, total_epoch):
        logging.info(f"Epoch {epoch+1}/{total_epoch}")
        segmentation_model = ImageSegmentationModel(
            model=model,
            in_channels=3,
            out_channels=5,
            num_classes=5,
            feature_nums=cfg["model"]["FEATURE_NUMS"],
            learning_rate=cfg["model"]["LEARNING_RATE"],
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
        )
        # Get train and validation loss and iou
        train_loss, train_iou = segmentation_model.train_val_epoch(
            train_loader, mode="train"
        )
        val_loss, val_iou = segmentation_model.train_val_epoch(val_loader, mode="val")
        logging.info(f"Training Loss: {train_loss:.4f} | Training IoU: {train_iou:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}")

        # save checkpoint if validation loss decreases
        checkpoint = {
            "state_dict": segmentation_model.model.state_dict(),
            # "optimizer": segmentation_model.optimizer.state_dict(),
            "epoch": epoch,
            "losses": {"train_loss": train_loss, "val_loss": val_loss},
            "iou": {"train_iou": train_iou, "val_iou": val_iou},
        }
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(checkpoint, cfg["files"]["MODEL_SAVE_PATH"])
        logging.info("---------------------------------------")


if __name__ == "__main__":
    main()
