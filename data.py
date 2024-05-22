from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, ImageNet
from lightning.pytorch import LightningDataModule
import torchvision.transforms as tt
from typing import Tuple
from torch import Tensor

class mnist(Dataset):
    def __init__(self, train: bool = True) -> None:
        self.erase_transform = tt.RandomErasing(1)

        if train:
            self.dataset = MNIST('./data/train', True, tt.Compose([
                 tt.ToTensor(),
                 tt.Resize((256, 256), antialias = True),
                 tt.Normalize((0.1307,), (0.3081,))
            ]), download = True)
        else:
            self.dataset = MNIST('./data/valid', True, tt.Compose([
                 tt.ToTensor(),
                 tt.Resize((256, 256), antialias = True),
                 tt.Normalize((0.1307,), (0.3081,))
            ]), download = True)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        return self.dataset[idx], self.erase_transform(self.dataset[idx])

class imagenet(Dataset):
    def __init__(self, train: bool = True) -> None:
        self.erase_transform = tt.RandomErasing(1)
        if train:
            self.dataset = ImageNet('./data/train', "train", transform = tt.Compose([
                 tt.ToTensor(),
                 tt.Resize((256, 256), antialias = True),
                 tt.Normalize((0.1307,), (0.3081,))
            ]))
        else:
            self.dataset = ImageNet('./data/valid', "val", tt.Compose([
                 tt.ToTensor(),
                 tt.Resize((256, 256), antialias = True),
                 tt.Normalize((0.1307,), (0.3081,))
            ]))

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        return self.dataset[idx], self.erase_transform(self.dataset[idx])

VALID_DATASETS = {
    'mnist': mnist,
    'imagenet': imagenet
}

class NormalModule(LightningDataModule):
    def __init__(self, type_dataset: str, batch_size: int, num_workers: int = 8, pin_memory: bool = True) -> None:
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.type_dataset = VALID_DATASETS[type_dataset]

    def prepare_data(self) -> None:
        
        self.train_ds = self.type_dataset(True)
        self.val_ds = self.type_dataset(False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            self.batch_size, 
            True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            self.batch_size, 
            True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )