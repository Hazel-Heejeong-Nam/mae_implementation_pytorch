import os
from  torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset
from torchvision import transforms



class patch_dataset(Dataset):
    def __init__(self, data_dir, mode ="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.data = self._load_data(self.data_dir, self.mode)
        if mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def _load_data(self, data_dir, mode):
        if os.path.isdir(data_dir):
            dataset = load_from_disk(data_dir)
        else: 
            dataset = load_dataset("matthieulel/galaxy10_decals")
            dataset.save_to_disk(data_dir)
        return dataset[mode]

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        label = self.data[idx]["label"]

        if self.transform:
            img = self.transform(img)

        return img, label