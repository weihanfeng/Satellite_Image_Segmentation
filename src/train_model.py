import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.models import (
    UNet,
    ImageSegmentationModel,
    UNetWithResnet50Encoder,
)
from pipeline.modeling.trainer import Trainer
from pipeline.modeling.dataset import SegmentationDataset, Transform
from utils.general_utils import setup_logging, save_model, load_model
from torch.utils.data import DataLoader
import torch.multiprocessing
import hydra
from omegaconf import DictConfig
import logging

torch.multiprocessing.set_sharing_strategy("file_system")

setup_logging()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(True)
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

    # Load model
    # model = UNet(
    #     in_channels=cfg["model"]["IN_CHANNELS"],
    #     out_channels=cfg["model"]["OUT_CHANNELS"],
    # )
    model = UNetWithResnet50Encoder(
        last_n_layers_to_unfreeze=cfg["model"]["LAST_N_LAYERS_TO_UNFREEZE"], 
        n_classes=cfg["model"]["OUT_CHANNELS"],
    )
    optimizer = optim.Adam
    last_epoch = 0
    if cfg["model"]["LOAD_MODEL"]:
        logging.info("Loading model...")
        model_artifact = load_model(cfg["files"]["MODEL_READ_PATH"])
        model.load_state_dict(model_artifact["state_dict"])
        last_epoch = model_artifact["epoch"]
        best_loss = model_artifact["losses"]["val_loss"]
        # optimizer.load_state_dict(model_artifact["optimizer"])
    else:
        best_loss = float("inf")

    # train model
    logging.info(f"Model architecture: {model.__class__.__name__}")
    logging.info("Start training...")
    trainer = Trainer(
        model=model,
        model_save_path=cfg["files"]["MODEL_SAVE_PATH"],
        num_classes=cfg["model"]["OUT_CHANNELS"],
        learning_rate=cfg["model"]["LEARNING_RATE"],
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        num_epochs=cfg["model"]["NUM_EPOCHS"],
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        last_epoch=last_epoch,
        reduce_lr_factor=cfg["model"]["REDUCE_LR_FACTOR"],
        reduce_lr_patience=cfg["model"]["REDUCE_LR_PATIENCE"],
        best_loss=best_loss,
    )
    logging.info("---------------------------------------")


if __name__ == "__main__":
    main()
