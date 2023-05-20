import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.unet import UNet
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

def train_model(train_loader, val_loader, model, loss_fn, optimizer, scaler):
    """Train model for one epoch
    Args:
        loader (DataLoader): DataLoader for training dataset
        model (nn.Module): model
        optimizer (Optimizer): optimizer
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision training
    Returns:
        None
    """
    progress_bar = tqdm(train_loader) # loader is an iterable
    total_loss = 0.0
    total_iou = 0.0
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data = data.to(device=DEVICE)
        # targets = F.one_hot(targets, num_classes=5).permute(0, 3, 1, 2)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast(): # cast tensors to a smaller memory footprint
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward() # scale loss to prevent underflow when using low precision floats
        scaler.step(optimizer)
        scaler.update()

        # calculate IoU
        # predicted_labels = torch.argmax(predictions, dim=1)
        # F one hot the predicted labels
        # predicted_labels = F.one_hot(predicted_labels, num_classes=5).permute(0, 3, 1, 2)
        iou = compute_iou(predictions, targets)
        total_iou += iou.item()
        total_loss += loss.item()
        ongoing_iou = total_iou/(batch_idx+1)
        ongoing_loss = total_loss/(batch_idx+1)

        # update tqdm progress bar
        progress_bar.set_postfix({"loss": ongoing_loss, "IoU": ongoing_iou})
        progress_bar.update()
    
    train_loss = total_loss/len(train_loader)
    train_iou = total_iou/len(train_loader)
    val_loss, val_iou = validate_model(val_loader, model, loss_fn)
    
    print(f"Training Loss: {train_loss:.4f} | Training IoU: {train_iou:.4f}")
    print(f"Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}")
    print("---------------------------------------")
    
def compute_iou(pred, target):
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)
    iou = torch.sum(intersection) / torch.sum(union)

    return iou

def validate_model(loader, model, loss_fn):
    """Validate model on validation dataset
    Args:
        loader (DataLoader): DataLoader for validation dataset
        model (nn.Module): model
        loss_fn: loss function
    Returns:
        val_loss (float): validation loss
        val_iou (float): validation IoU
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    total_iou = 0.0
    total_batches = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=DEVICE)
            # targets = F.one_hot(targets, num_classes=5).permute(0, 3, 1, 2)
            targets = targets.float().to(device=DEVICE)

            predictions = model(data)
            loss = loss_fn(predictions, targets)

            val_loss += loss.item()

            # predicted_labels = torch.argmax(predictions, dim=1)
            # predicted_labels = F.one_hot(predicted_labels, num_classes=5).permute(0, 3, 1, 2)
            iou = compute_iou(predictions, targets)
            total_iou += iou.item()

            total_batches += 1

    val_loss /= total_batches
    val_iou = total_iou / total_batches

    return val_loss, val_iou

def main():
    # create model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    # create loss function
    # non binary cross entropy loss is used for multi-class classification
    loss_fn  = nn.CrossEntropyLoss()
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # create mixed precision training scaler
    scaler = torch.cuda.amp.GradScaler()

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

    # train model
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_model(train_loader, val_loader, model, loss_fn, optimizer, scaler)

if __name__ == "__main__":
    main()