import numpy as np

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


def lrs_lambda(epoch):
    if epoch == 0:
        return 1.0
    prod = 1.0
    prod *= 0.5 ** (epoch // 2)
    if epoch % 2 > 0:
        prod *= 0.75
    return prod


class MyBatchNorm(nn.Module):
    def forward(self, x):
        m = x.mean(dim=[1, 2, 3], keepdim=True)
        s = s = torch.clip(x.std(dim=[1, 2, 3], keepdim=True), min=1e-6)
        x = (x - m) / s
        return x


class DilConv2(L.LightningModule):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        channels = 96

        self.net = nn.ModuleList((
            nn.Conv2d(4, channels, kernel_size, padding="same", dilation=1),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 2),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 3),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 4),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 5),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, 12, 1, padding="same"),
            # nn.ReLU(),
            MyBatchNorm(),
        ))

    def forward(self, x):
        x = x.to(device=self.net[0].weight.device).squeeze(2)

        # handling negatives
        x = nn.functional.relu(x, inplace=True)

        # norm
        m = x.mean(dim=[1, 2, 3], keepdim=True)
        s = s = torch.clip(x.std(dim=[1, 2, 3], keepdim=True), min=1e-6)
        x = (x - m) / s

        for lay in self.net:
            x = lay(x)

        # unnorm
        x = x * s + m

        return x.unsqueeze(2)
    
    def training_step(self, batch):
        x, y = batch
        out = self.forward(x)
        loss = (((out -  y) * (y !=  - 1)) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-3)
        lrs = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs_lambda)
        return [optimizer], [lrs]
    

class DilConv2Extend(L.LightningModule):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        channels = 48

        self.main_model = DilConv2().load_from_checkpoint("../models/main_model_epoch3.ckpt")
        for param in self.main_model.parameters():
            param.requires_grad = False

        self.net = nn.ModuleList((
            MyBatchNorm(),
            nn.Conv2d(4 * 20, channels, kernel_size, padding="same", dilation=1),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 2),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 3),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 4),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, channels, kernel_size, padding="same", dilation=kernel_size ** 5),
            nn.ReLU(),
            MyBatchNorm(),
            nn.Conv2d(channels, 12, 1, padding="same")
        ))

    def main_model_forecast(self, x):
        self.main_model.eval()
        with torch.no_grad():
            main_preds = self.main_model(x)
        return main_preds

    def forward(self, x):
        main_preds = self.main_model_forecast(x[:, :, 0:1])
        x = x[:, :, 1:]

        x = x.to(device=self.net[1].weight.device).reshape(x.shape[0], x.shape[1] * x.shape[2], 252, 252)

        # handling negatives
        x = nn.functional.relu(x, inplace=True)

        for lay in self.net:
            x = lay(x)

        return x.unsqueeze(2) + main_preds
    
    def training_step(self, batch):
        x, y = batch
        out = self.forward(x)
        loss = (((out -  y) * (y !=  - 1)) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-3)
        lrs = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs_lambda)
        return [optimizer], [lrs]