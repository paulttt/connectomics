import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


class TracesDataset(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', mode='fit', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size

    def prepare_data(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.mode == 'fit':
            X = np.load(self.data_dir + "X_train.npy")
            y = np.load(self.data_dir + "y_train.npy")
            X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=y)
            self.x_train_tensor = torch.from_numpy(X_train).float().to(device)
            self.y_train_tensor = torch.from_numpy(y_train).float().to(device)
            self.x_val_tensor = torch.from_numpy(X_val).float().to(device)
            self.y_val_tensor = torch.from_numpy(y_val).float().to(device)

        if self.mode == 'test':
            X = np.load(self.data_dir + "X_test.npy")
            y = np.load(self.data_dir + "y_test.npy")
            self.x_test_tensor = torch.from_numpy(X).float().to(device)
            self.y_test_tensor = torch.from_numpy(y).float().to(device)


    def setup(self, stage='fit'):
        if stage == 'fit' or stage is None:
            self.train_data = TensorDataset(self.x_train_tensor, self.y_train_tensor)
            self.val_data = TensorDataset(self.x_val_tensor, self.y_val_tensor)

        if stage == 'test' or stage is None:
            self.test_data = TensorDataset(self.x_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
