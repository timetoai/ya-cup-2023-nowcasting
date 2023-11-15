import argparse

import h5py
import lightning as L
import numpy as np
import torch
import torch.utils.data as data
import tqdm

from datasets import RadarDataset
from models import DilConv2


def prepare_data_loaders(train_batch_size=6, valid_batch_size=1, test_batch_size=1, mode="overlap",
                         num_workers=0):
    use_dataset = RadarDataset

    train_dataset = use_dataset([
        '../train/2021-01-train.hdf5', '../train/2021-03-train.hdf5', '../train/2021-04-train.hdf5',
        '../train/2021-06-train.hdf5', '../train/2021-07-train.hdf5', '../train/2021-09-train.hdf5',
        '../train/2021-10-train.hdf5', '../train/2021-12-train.hdf5',
        '../train/2021-02-train.hdf5', '../train/2021-05-train.hdf5', '../train/2021-08-train.hdf5',
        '../train/2021-11-train.hdf5'], mode=mode)
    valid_dataset = use_dataset([
        '../train/2021-02-train.hdf5', '../train/2021-05-train.hdf5', '../train/2021-08-train.hdf5',
        '../train/2021-11-train.hdf5'
    ])
    test_dataset = use_dataset(['../2022-test-public.hdf5'], out_seq_len=0, with_time=True)
    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def main(epochs, batch_size, num_workers):
    L.seed_everything(42)

    train_loader, valid_loader, test_loader = prepare_data_loaders(train_batch_size=batch_size,
                                                                   valid_batch_size=batch_size,
                                                                   test_batch_size=1,
                                                                   num_workers=num_workers)
    
    model = DilConv2()
    trainer = L.Trainer(
        logger=L.pytorch.loggers.TensorBoardLogger(save_dir="tensorboard_logs"),
        max_epochs=epochs,
        callbacks=[L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')]
    )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.num_workers)
