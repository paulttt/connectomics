from torch.utils import data
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np


def calc_epochs(desired_iter, batch_size, num_data):
    steps_per_epoch = num_data // batch_size
    epochs_needed = desired_iter // steps_per_epoch
    return epochs_needed


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def normalize_dataset(x):
    x = x.data.cpu().numpy()
    scaler = MinMaxScaler()
    scaler.fit(x)
    x_norm = scaler.transform(x)
    return torch.from_numpy(x_norm)


def print_curr_lr(every, curr_epoch, optimizer):
    if curr_epoch % every == 0:
        print('Current Learning Rate is: {:.4f}'.format(optimizer.param_groups[0]['lr']))


def corrupt_input(x, corrupt=0.2):
    num_elements = len(x)
    k = int(num_elements * corrupt)
    corrupted = x.detach().clone()
    indices = torch.randperm(len(x))[:k]
    corrupted[indices] = 0
    return corrupted


def corrupt_signal(signal, corrupt=0.5, mean=0.0, var=0.5, normalize=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_masks = []
    for i in range(signal.shape[0]):
        if normalize:
            rms = np.sqrt(np.mean(signal[i].data.cpu().numpy()**2, axis=0))
            normalized_var = var * rms
        noise = torch.from_numpy(np.random.normal(mean, normalized_var, signal[i].shape)).float()
        signal_dim = signal[i].shape[0]
        indices = torch.randint(0, signal_dim, (1, int(signal_dim * corrupt))).data.cpu().numpy()
        noise_mask = torch.zeros((1, signal_dim))
        noise_mask[0, indices] = noise[indices]
        noise_masks.append(noise_mask.data.cpu())
    noise = torch.cat(noise_masks, dim=0)
    noise = noise.to(device)
    return signal+noise


def target_distribution(q):
    p_nom = q ** 2 / torch.sum(q, dim=0)
    p = (p_nom.t() / torch.sum(p_nom, dim=1)).t()
    return p


def kullback_leibler(p, q):
    kl = torch.mean(torch.sum(p * torch.log(p/(q+1e-6)), dim=1))
    return kl


def accuracy(y_pred, y_true):
    dim = max(y_pred.max(), y_true.max()) + 1
    counts = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        counts[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = np.asarray(linear_sum_assignment(counts.max() - counts))
    acc = counts[row_ind, col_ind].sum() / y_pred.shape[0]
    return acc, counts


def f1_binary(y_pred, y_true):
    sz_pred = np.unique(y_pred).size
    sz_gt = np.unique(y_true).size
    assert (sz_pred == 2) * (sz_gt == 2), "Class Labels should be the same and binary. " \
                                          "Got {} and {} instead.".format(sz_pred, sz_gt)
    dim = max(y_pred.max(), y_true.max()) + 1
    counts = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        counts[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = np.asarray(linear_sum_assignment(counts.max() - counts))
    if row_ind[0] != col_ind[0]:
        temp = y_pred
        y_pred[temp == 0] = 1
        y_pred[temp == 1] = 0
    cm = confusion_matrix(y_true, y_pred)
    (tn, fp, fn, tp) = cm.ravel()
    precision = tp / (tp + fp + np.spacing(1))
    recall = tp / (tp + fn + np.spacing(1))
    f1 = (2 * precision * recall) / (precision + recall + np.spacing(1))
    return f1


class Dataset(data.Dataset):
    def __init__(self, inputs, target, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.data = inputs
        self.labels = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        datapoint = self.data[idx]
        target = self.labels[idx]
        if self.transform:
            datapoint = self.transform(datapoint)
        if self.target_transform:
            target = self.target_transform(target)
        return datapoint, target
