import numpy as np
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMinMax
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sktime.utils.data_processing import from_2d_array_to_nested


def NumpyDataloader(dataset='Haleh', train_val_split=0.25, test_only=False):
    path = 'data/'
    if dataset == 'Haleh':
        path = 'data/Haleh/'

    elif dataset == 'John':
        labels = np.loadtxt(path + "John/inhibitory_cells.txt", delimiter=',')
        signals = np.load(path + "John/signals.npy")
        labels = labels[(labels[:, 0] == 1) * (labels[:, 0] == 2)]
        signals = signals[np.ma.mask_or((labels[:, 0] == 1), (labels[:, 0] == 2))]
    else:
        raise NotImplementedError
    if test_only:
        X = np.load(path + 'X_test.npy')
        y = np.load(path + 'y_test.npy')
        train_val_split = 0
    else:
        X = np.load(path + 'X_train.npy')
        y = np.load(path + 'y_train.npy')
    if train_val_split > 0 and not test_only:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_split,
                                                            random_state=42, stratify=y)
        return (X_train, y_train), (X_test, y_test)
    else:
        return X, y


def SktDataloader(dataset='Haleh', train_val_split=0.25, test_only=False):
    if train_val_split > 0 and not test_only:
        (X_train, y_train), (X_val, y_val) = NumpyDataloader(dataset, train_val_split, test_only)
        train_nested = from_2d_array_to_nested(X_train)
        val_nested = from_2d_array_to_nested(X_val)
        return (train_nested, y_train), (val_nested, y_val)
    else:
        X, y = NumpyDataloader(dataset, train_val_split, test_only)
        nested = from_2d_array_to_nested(X)
        return nested, y


class LitDataloader(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str = './',
                 mode='fit',
                 batch_size=64,
                 indices=None,
                 one_hot=True,
                 num_classes=2,
                 imbalanced_sampler=True,
                 normalize=False):
        super(LitDataloader, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.imbalanced_sampler = imbalanced_sampler
        self.indices = indices
        self.normalize = normalize
        if self.normalize:
            self.scaler = TimeSeriesScalerMinMax(value_range=(0, 1))

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.mode == 'fit':
            X = np.load(self.data_dir + "X_train.npy")
            if self.normalize:
                X = self.scaler.fit_transform(X)
            y = np.load(self.data_dir + "y_train.npy")
            if self.indices is None:
                X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=y)
            else:
                X_train, X_val = X[self.indices[0]], X[self.indices[1]]
                y_train, y_val = y[self.indices[0]], y[self.indices[1]]
            self.x_train_tensor = torch.from_numpy(X_train).float().to(device)
            self.x_val_tensor = torch.from_numpy(X_val).float().to(device)
            if self.one_hot:
                self.y_train_tensor = F.one_hot(torch.from_numpy(y_train).to(torch.int64),
                                                num_classes=self.num_classes).to(device)
                self.y_val_tensor = F.one_hot(torch.from_numpy(y_val).to(torch.int64),
                                              num_classes=self.num_classes).to(device)
            else:
                self.y_train_tensor = torch.from_numpy(y_train).to(device)
                self.y_val_tensor = torch.from_numpy(y_val).to(device)

        if self.mode == 'test':
            X = np.load(self.data_dir + "X_test.npy")
            if self.normalize:
                X = self.scaler.fit_transform(X)
            y = np.load(self.data_dir + "y_test.npy")
            self.x_test_tensor = torch.from_numpy(X).float().to(device)

            if self.one_hot:
                self.y_test_tensor = F.one_hot(torch.from_numpy(y).to(torch.int64), num_classes=self.num_classes).to(device)
            else:
                self.y_test_tensor = torch.from_numpy(y).to(device)

    def setup(self):
        if self.mode == 'fit' or self.mode is None:
            self.train_data = TraceDataset(self.x_train_tensor, self.y_train_tensor)
            self.val_data = TraceDataset(self.x_val_tensor, self.y_val_tensor)

        if self.mode == 'test' or self.mode is None:
            self.test_data = TraceDataset(self.x_test_tensor, self.y_test_tensor)

    def train_dataloader(self):
        if self.imbalanced_sampler:
            return DataLoader(self.train_data, batch_size=self.batch_size,
                              sampler=ImbalancedDatasetSampler(self.train_data))
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.imbalanced_sampler:
            return DataLoader(self.val_data, batch_size=self.batch_size,
                              sampler=ImbalancedDatasetSampler(self.val_data))
        else:
            return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class TraceDataset(TensorDataset):
    def __init__(self, X, y):
        super(TraceDataset, self).__init__()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.data = Tensor(X.float())
        self.labels = Tensor(y.float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target_val = self.labels[index]
        data_val = self.data[index]
        return data_val, target_val


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(ImbalancedDatasetSampler)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            for l in label:
                if l in label_to_count:
                    label_to_count[l] += 1
                else:
                    label_to_count[l] = 1

        # weight for each sample
        weights = [1.0 / min([label_to_count[l] for l in self._get_label(dataset, idx)])
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx].data.cpu().numpy()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples