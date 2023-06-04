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


class Trainer:
    def __init__(
        self,
        model,
        model_save_path,
        num_classes=5,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
        loss_fn=nn.CrossEntropyLoss(),
        num_epochs=10,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_save_path = model_save_path
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_epochs = num_epochs

    def train(
        self,
        train_loader,
        val_loader,
        last_epoch=0,
        reduce_lr_factor=0.1,
        reduce_lr_patience=2,
        best_loss=float("inf"),
    ):
        """Run training and validation epochs
        
        Args:
            train_loader (DataLoader): training DataLoader
            val_loader (DataLoader): validation DataLoader
            last_epoch (int, optional): last epoch. Defaults to 0.
            reduce_lr_factor (float, optional): factor to reduce learning rate. Defaults to 0.1.
            reduce_lr_patience (int, optional): number of epochs to wait before reducing learning rate. Defaults to 2.
            best_loss (float, optional): best loss recorded. Defaults to float("inf").
        """
        total_epoch = last_epoch + self.num_epochs
        for epoch in range(last_epoch, total_epoch):
            logging.info(f"Epoch {epoch+1}/{total_epoch}")
            for param_group in self.optimizer.param_groups:
                logging.info(f"Learning rate: {param_group['lr']}")

            # Run training and validation epoch
            train_loss, train_iou = self._run_single_epoch(train_loader, mode="train")
            val_loss, val_iou = self._run_single_epoch(val_loader, mode="val")

            # Reduce learning rate if validation loss does not improve
            reduce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                verbose=True,
            )
            reduce_lr_scheduler.step(val_loss)

            # Save model checkpoint if validation loss is lower than best loss
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint(
                    epoch=epoch,
                    losses={"train_loss": train_loss, "val_loss": val_loss},
                    iou={"train_iou": train_iou, "val_iou": val_iou},
                )

            logging.info("---------------------------------------")

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
        pred_labels = F.one_hot(pred_labels, num_classes=self.num_classes).permute(
            0, 3, 1, 2
        )
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

    def _run_single_epoch(self, loader, mode):
        """A training and validation epoch
        Args:
            loader (DataLoader): DataLoader
            mode (str): train or val
        Returns:
            train_loss (float): training loss
            train_iou (float): training IoU
        """
        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()
        else:
            raise ValueError("Mode must be either train or val")

        progress_bar = tqdm(loader, desc="Training epoch")
        total_loss = 0.0
        total_iou = 0.0
        for batch_idx, (data, targets) in enumerate(progress_bar):
            # Perform forward pass and update model
            if mode == "train":
                loss, iou = self._train_step(data, targets)
            elif mode == "val":
                loss, iou = self._val_step(data, targets)

            # get result for current batch
            total_iou += iou.item()
            total_loss += loss.item()
            ongoing_iou = total_iou / (batch_idx + 1)
            ongoing_loss = total_loss / (batch_idx + 1)

            # update tqdm progress bar
            progress_bar.set_postfix({"loss": ongoing_loss, "IoU": ongoing_iou})
            progress_bar.update()

        epoch_loss = total_loss / len(loader)
        epoch_iou = total_iou / len(loader)

        return epoch_loss, epoch_iou

    def _train_step(self, data, targets):
        """A training step

        Args:
            data (torch.Tensor): input batch training data
            targets (torch.Tensor): batch training target

        Returns:
            loss (torch.Tensor): batch training loss
            iou (torch.Tensor): batch training IoU
        """
        with torch.cuda.amp.autocast():
            data = data.to(device=self.device)
            targets = F.one_hot(targets, num_classes=self.num_classes).permute(
                0, 3, 1, 2
            )
            targets = targets.float().to(device=self.device)
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss, iou, _ = self._get_prediction_result(predictions, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.step()

        return loss, iou

    def _val_step(self, data, targets):
        """A validation step

        Args:
            data (torch.Tensor): input batch validation data
            targets (torch.Tensor): batch validation target

        Returns:
            loss (torch.Tensor): batch validation loss
            iou (torch.Tensor): batch validation IoU
        """
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                data = data.to(device=self.device)
                targets = F.one_hot(targets, num_classes=self.num_classes).permute(
                    0, 3, 1, 2
                )
                targets = targets.float().to(device=self.device)
                predictions = self.model(data)
                loss, iou, _ = self._get_prediction_result(predictions, targets)

        return loss, iou

    def _save_checkpoint(self, epoch, losses, iou):
        """Save model checkpoint
        Args:
            epoch (int): current epoch
            losses (dict): training and validation losses
            iou (dict): training and validation IoU
        """
        checkpoint = {
            "state_dict": self.model.state_dict(),
            # "optimizer": segmentation_model.optimizer.state_dict(),
            "epoch": epoch,
            "losses": losses,
            "iou": iou,
        }
        save_model(checkpoint, self.model_save_path)
        logging.info("Model saved")
