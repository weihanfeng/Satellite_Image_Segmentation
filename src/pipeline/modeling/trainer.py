"""
Trainer class for training models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from utils.general_utils import setup_logging, save_model, load_model
import logging

setup_logging()

class Trainer():
    def __init__(
        self,
        model,
        num_classes=5,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
        loss_fn=nn.CrossEntropyLoss(),
        num_epochs=10,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_epochs = num_epochs

    def _get_prediction_result(self, pred, target):
        """Get prediction result
        Args:
            pred (torch.Tensor): prediction
            target (torch.Tensor): target
        Returns:
            loss (torch.Tensor): loss
            iou (torch.Tensor): IoU
            pred_labels (torch.Tensor): predicted labels
        """
        loss = self.loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        pred_labels = F.one_hot(pred_labels, num_classes=self.num_classes).permute(0, 3, 1, 2)
        iou = self._compute_iou(pred_labels, target)

        return loss, iou, pred_labels

    def _compute_iou(self, pred, target):
        """Compute IoU
        Args:
            pred (torch.Tensor): prediction
            target (torch.Tensor): target
        Returns:
            iou (torch.Tensor): IoU
        """
        intersection = torch.logical_and(pred, target)
        union = torch.logical_or(pred, target)
        iou = torch.sum(intersection) / torch.sum(union)

        return iou
    
    def train_single_epoch(self, loader):
        """A training and validation epoch
        Args:
            loader (DataLoader): DataLoader
        Returns:
            train_loss (float): training loss
            train_iou (float): training IoU
        """
        self.model.train()

        progress_bar = tqdm(loader, desc="Training epoch")
        total_loss = 0.0
        total_iou = 0.0
        for batch_idx, (data, targets) in enumerate(progress_bar):
            
            # Perform forward pass and update model
            with torch.cuda.amp.autocast(): # cast tensors to a smaller memory footprint
                data = data.to(device=self.device)
                targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
                targets = targets.float().to(device=self.device)
                self.optimizer.zero_grad()
                predictions = self.model(data)
                loss, iou, _ = self._get_prediction_result(predictions, targets)
                self.scaler.scale(loss).backward() # scale loss to prevent underflow when using low precision floats
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.step()
            
            # get result for current batch
            total_iou += iou.item()
            total_loss += loss.item()
            ongoing_iou = total_iou/(batch_idx+1)
            ongoing_loss = total_loss/(batch_idx+1)

            # update tqdm progress bar
            progress_bar.set_postfix({"loss": ongoing_loss, "IoU": ongoing_iou})
            progress_bar.update()

        epoch_loss = total_loss/len(loader)
        epoch_iou = total_iou/len(loader)

        return epoch_loss, epoch_iou
    
    def val_single_epoch(self, loader):
        """A validation epoch"""
        self.model.eval()

        progress_bar = tqdm(loader, desc="Validation epoch")
        total_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(progress_bar):
                # Perform forward pass and update model
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        data = data.to(device=self.device)
                        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)
                        targets = targets.float().to(device=self.device)
                        predictions = self.model(data)
                        loss, iou, _ = self._get_prediction_result(predictions, targets)

                # get result for current batch
                total_iou += iou.item()
                total_loss += loss.item()
                ongoing_iou = total_iou/(batch_idx+1)
                ongoing_loss = total_loss/(batch_idx+1)

                # update tqdm progress bar
                progress_bar.set_postfix({"loss": ongoing_loss, "IoU": ongoing_iou})
                progress_bar.update()

        epoch_loss = total_loss/len(loader)
        epoch_iou = total_iou/len(loader)

        return epoch_loss, epoch_iou
    
    def train(self, loader):
        """Run training and validation epochs"""
        for epoch in range(self.num_epochs):
            
            train_loss, train_iou = self.train_batch(loader)
            val_loss, val_iou = self.val_batch(loader)
            logging.info(f"Training Loss: {train_loss:.4f} | Training IoU: {train_iou:.4f}")
            logging.info(f"Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}")

