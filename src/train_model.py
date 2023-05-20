import torch
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.models import UNet, ImageSegmentationModel
from pipeline.modeling.dataset import SegmentationDataset, Transform
from torch.utils.data import DataLoader
import torch.multiprocessing
import hydra
from omegaconf import DictConfig

torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # create train and validation dataset
    train_dataset = SegmentationDataset(
        image_dir=cfg["files"]["TRAIN_IMG_DIR"],
        transform=Transform(),
    )
    print(cfg["files"]["TRAIN_IMG_DIR"])
    print(len(train_dataset))
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

    model = UNet(
        in_channels=cfg["model"]["IN_CHANNELS"],
        out_channels=cfg["model"]["OUT_CHANNELS"],
    )
    # train model
    for epoch in range(cfg["model"]["NUM_EPOCHS"]):
        print(f"Epoch {epoch+1}/{cfg['model']['NUM_EPOCHS']}")
        segmentation_model = ImageSegmentationModel(
            model=model,
            in_channels=3,
            out_channels=5,
            num_classes=5,
            feature_nums=cfg["model"]["FEATURE_NUMS"],
            learning_rate=cfg["model"]["LEARNING_RATE"],
            optimizer=optim.Adam,
            loss_fn=nn.CrossEntropyLoss(),
        )
        train_loss, train_iou = segmentation_model.train_val_epoch(
            train_loader, mode="train"
        )
        val_loss, val_iou = segmentation_model.train_val_epoch(val_loader, mode="val")
        print(f"Training Loss: {train_loss:.4f} | Training IoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}")
        print("---------------------------------------")


if __name__ == "__main__":
    main()
