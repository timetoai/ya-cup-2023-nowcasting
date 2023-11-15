import argparse

import h5py
import lightning as L
import numpy as np
import torch
import torch.utils.data as data
import tqdm

from datasets import RadarDataset2
from models import DilConv2Extend


def prepare_data_loaders(train_batch_size=6, valid_batch_size=1, test_batch_size=1, mode="overlap",
                         num_workers=0):
    use_dataset = RadarDataset2

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

def process_test(model, test_loader, output_file='../output.hdf5'):
    model.eval()
    with torch.no_grad():
        for index, item in tqdm.tqdm(enumerate(test_loader)):
            (inputs, last_input_timestamp), _ = item
            output = model(inputs)
            with h5py.File(output_file, mode='a') as f_out:
                for index in range(output.shape[1]):
                    timestamp_out = str(int(last_input_timestamp[-1]) + 600 * (index + 1))
                    f_out.create_group(timestamp_out)
                    f_out[timestamp_out].create_dataset(
                        'intensity',
                        data=output[0, index, 0].detach().cpu().numpy().astype(np.float16)
                    )

def main():
    L.seed_everything(42)

    train_loader, valid_loader, test_loader = prepare_data_loaders(train_batch_size=8,
                                                                   valid_batch_size=8,
                                                                   test_batch_size=1,
                                                                   num_workers=2)
    
    model = DilConv2Extend().load_from_checkpoint("../models/extend_model_epoch2.ckpt")
    process_test(model, test_loader)


if __name__ == '__main__':
    main()
