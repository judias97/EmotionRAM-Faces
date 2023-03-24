'''
data_loaders.py
Created on 2023 03 13 16:25:06
Description: Arquivo responsavel por carregar os dados do dataset e prepara-los para utilização.

Author: Will <wlc2@cin.ufpe.br> and Julia <jdts@cin.ufpe.br>
'''

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageDraw
import math
from utils import get_transform
import numpy as np

np.random.seed(0)

class CAERSDataset(Dataset):
    def __init__(self, root, input_file, transforms=None):
        self.root = root
        self.input_file = input_file
        self.transforms = transforms
        self.data = self.read_input_file(self.input_file)

    def read_input_file(self, file):
        return [line.rstrip('\n') for line in open(file)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].split(',')
        path = os.path.join(self.root, sample[0])
        label = int(float(sample[1]))
        x1, y1, x2, y2 = math.ceil(float(sample[2])), math.ceil(float(sample[3])), math.ceil(float(sample[4])), math.ceil(float(sample[5]))

        im = Image.open(path)

        face = im.crop((x1, y1, x2, y2))

        data = {
            'face': face
        }

        if self.transforms is not None:
            data = self.transforms(data)
            data['path'] = path

        return data, label

def main():
    train_dataset = CAERSDataset("data/CAER-S", "data/train.txt", get_transform(train=True))
    test_dataset = CAERSDataset("data/CAER-S", "data/test.txt", get_transform(train=False))
    val_dataset = CAERSDataset("data/CAER-S", "data/val.txt", get_transform(train=False))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(next(iter(train_dataloader)))

if __name__ == '__main__':
    main()

