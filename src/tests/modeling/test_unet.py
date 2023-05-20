import pytest
import torch
import pipeline as pl

def test_UNET():
    x = torch.randn((5, 3, 256, 256))
    model = pl.modeling.models.UNet(in_channels=3, out_channels=1)
    preds = model(x)
    
    assert preds.shape == (5, 1, 256, 256), "Output shape is incorrect"