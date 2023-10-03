import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import Dataset, RandomSampler, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from einops import rearrange
from tqdm import tqdm
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
from dataset import SpeechAudioDataset

    
def collate_fn(batch):
    inputs, targets, length_ratio = [], [], []
    for noisy_input, clean_target in batch:
        inputs.append(noisy_input)
        targets.append(clean_target)
        length_ratio.append(len(noisy_input))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    length_ratio = torch.tensor(length_ratio, dtype=torch.long) / inputs.shape[1]

    return inputs, targets, length_ratio


class AudioDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, target_rate, max_length, batch_size, num_workers):
        super().__init__()

        self.data_dir = data_dir
        self.target_rate = target_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_filenames, valid_filenames, test_filenames = self.prepare_data()
        self.train_noisy = train_filenames[0]
        self.train_clean = train_filenames[1]

        self.valid_noisy = valid_filenames[0]
        self.valid_clean = valid_filenames[1]

        self.test_noisy = test_filenames[0]
        self.test_clean = test_filenames[1]

    def prepare_data(self):

        train_filenames = pd.read_csv(f"{self.data_dir}/train_filenames.csv")
        valid_filenames = pd.read_csv(f"{self.data_dir}/valid_filenames.csv")
        test_filenames = pd.read_csv(f"{self.data_dir}/test_filenames.csv")
        train_filenames = train_filenames.to_numpy().squeeze()
        train_filenames = rearrange(train_filenames, "(n c) -> n c", c=22)

        valid_filenames = valid_filenames.to_numpy().squeeze()
        valid_filenames = rearrange(valid_filenames, "(n c) -> n c", c=22)

        test_filenames = test_filenames.to_numpy().squeeze()
        test_filenames = rearrange(test_filenames, "(n c) -> n c", c=22)

        train_noisy_filenames, train_clean_filenames = self.create_pair_filenames(train_filenames)
        valid_noisy_filenames, valid_clean_filenames = self.create_pair_filenames(valid_filenames)
        test_noisy_filenames, test_clean_filenames = self.create_pair_filenames(test_filenames)

        return (train_noisy_filenames, train_clean_filenames), (valid_noisy_filenames, valid_clean_filenames), (test_noisy_filenames, test_clean_filenames)

    def create_pair_filenames(self,filenames):
        clean_filenames = []
        noisy_filenames = []

        for chunk_filenames in tqdm(filenames):

            for filename in chunk_filenames:
                snr_level = filename.split(os.path.sep)[-2]
                if snr_level == "InfdB":
                    break
            clean_filenames += [filename] * len(chunk_filenames)
            noisy_filenames += chunk_filenames.tolist()

        return noisy_filenames, clean_filenames
    
    def train_dataloader(self):

        train_dataset = SpeechAudioDataset(self.train_noisy,
                                 self.train_clean,
                                 self.data_dir,
                                 self.target_rate,
                                 self.max_length,
                                )
        Sampler = RandomSampler(train_dataset, replacement=False, num_samples=300_000)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=Sampler,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=False,
            )
        
        return train_loader

    def valid_dataloader(self):
        
        valid_dataset = SpeechAudioDataset(self.valid_noisy,
                                 self.valid_clean,
                                 self.data_dir,
                                 self.target_rate,
                                 self.max_length,
                                )
        Sampler = None

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=Sampler,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=False,
            )
        
        return valid_loader
        
    def test_dataloader(self):
        
        test_dataset = SpeechAudioDataset(self.test_noisy,
                                 self.test_clean,
                                 self.data_dir,
                                 self.target_rate,
                                 self.max_length,
                                )
        Sampler = None

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=Sampler,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=False,
            )
        return test_loader



    


