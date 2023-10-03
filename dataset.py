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

class SpeechAudioDataset(Dataset):
    def __init__(self,
                 noisy_files,
                 clean_files,
                 base_path,
                 target_rate,
                 max_length,
                ):

        # list of files
        self.base_path = base_path
        self.noisy_files = noisy_files
        self.clean_files = clean_files

        # fixed len
        self.target_rate = target_rate
        self.max_length = max_length


    def __len__(self):
        return len(self.noisy_files)


    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        x_clean, x_noisy = self.same_length(x_clean, x_noisy)

        x_clean = self.crop_audio(x_clean)
        x_noisy = self.crop_audio(x_noisy)

        return torch.from_numpy(x_noisy), torch.from_numpy(x_clean)


    def load_sample(self, filename):
        filename = os.path.join(self.base_path, filename)
        waveform, _ = librosa.load(filename, sr=self.target_rate)
        return waveform


    def same_length(self, x_clean, x_noisy):
        clean_length = len(x_clean)
        noisy_length = len(x_noisy)

        if clean_length > noisy_length:
            x_clean = x_clean[:noisy_length]
        else:
            x_noisy = x_noisy[:clean_length]

        return x_clean, x_noisy

    def crop_audio(self, x):
        if len(x) > self.max_length:
            x = x[:self.max_length]

        return x