import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from pipeline.modeling.unet import UNet
from pipeline.modeling.dataset import SegmentationDataset, Transform
from torch.utils.data import DataLoader
import torch.nn.functional as F

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/selected_data_split/train/"
TRAIN_MASK_DIR = "data/selected_data_split/train/"
VAL_IMG_DIR = "data/selected_data_split/val/"
VAL_MASK_DIR = "data/selected_data_split/val/"

def train_model(loader, model, loss_fn, optimizer, scaler):
    """Train model for one epoch
    Args:
        loader (DataLoader): DataLoader for training dataset
        model (nn.Module): model
        optimizer (Optimizer): optimizer
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision training
    Returns:
        None
    """
    progress_bar = tqdm(loader) # loader is an iterable
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data = data.to(device=DEVICE)
        targets = F.one_hot(targets, num_classes=5).permute(0, 3, 1, 2)
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

        total_loss += loss.item()

        # update tqdm progress bar
        progress_bar.set_postfix({"training_loss": total_loss})
        progress_bar.update()
    
    epoch_loss = total_loss/len(loader)
    print(f"Epoch loss: {epoch_loss}")
    

def main():
    # create model
    model = UNet(in_channels=3, out_channels=5).to(DEVICE)
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
        train_model(train_loader, model, loss_fn, optimizer, scaler)

if __name__ == "__main__":
    main()