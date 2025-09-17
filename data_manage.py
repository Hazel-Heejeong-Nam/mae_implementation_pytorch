import os
import torch
from  torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk



class patch_dataset(Dataset):
    def __init__(self, data_dir, mode ="train", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.train = self._load_data(self.data_dir, self.mode)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _load_data(self, data_dir, mode):
        if os.path.isdir(data_dir):
            dataset = load_from_disk("./galaxy_dataset")
            dataset.save_to_disk("./galaxy_dataset")
        else: 
            dataset = load_from_disk("./galaxy_dataset")

        return dataset[mode]

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if self.transform:
            img = self.transform(img)

        return img, label