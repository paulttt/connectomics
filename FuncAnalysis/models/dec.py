import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from collections import OrderedDict
from utils.utils import target_distribution, accuracy, kullback_leibler, f1_binary
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback


class DEC(nn.Module):
    def __init__(self, input_dim=784, latent_dim=10, n_clusters=10, hidden_dims=None,
                 dropout=0.0, alpha=1.0, acc='accuracy'):
        super().__init__()

        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        if hidden_dims is None:
            hidden_dims = [500, 50, 2000]
        self.accuracy = acc


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layers = [input_dim] + hidden_dims + [latent_dim]
        modules = []
        input_channels = input_dim
        for h_dim in hidden_dims:
            submodules = [("linear", nn.Linear(input_channels, h_dim)), ("activation", nn.ReLU())]
            if dropout > 0:
                submodules.append(("dropout", nn.Dropout(dropout)))
            input_channels = h_dim
            modules.append(nn.Sequential(OrderedDict(submodules)))
        modules.append(nn.Sequential(OrderedDict([("linear", nn.Linear(hidden_dims[-1], latent_dim))])))
        self.encoder = nn.Sequential(*modules)
        # Cluster centers, with feature dimension equal to latent/embedding space.
        self.mu = nn.Parameter(torch.Tensor(n_clusters, latent_dim))

    def load_model(self, path):
        curr_model_dict = self.state_dict()
        pretrained_model_dict = torch.load(path, map_location=self.device)
        update_dict = {param_name: param for param_name, param in pretrained_model_dict.items()
                       if param_name in curr_model_dict}
        curr_model_dict.update(update_dict)
        self.load_state_dict(curr_model_dict)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def soft_assign(self, z):
        squared_norm = torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2)
        base = 1.0 / (1.0 + (squared_norm / self.alpha))
        power = float(self.alpha + 1.0) / 2.0
        numerator = base ** power
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        q = numerator/denominator
        return q

    def encode(self, x):
        z = self.encoder(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        q = self.soft_assign(z)
        return q, z

    def fit(self, data, labels=None, num_clusters=10, lr=0.01, batch_size=256, num_epochs=10, stop_criterion=0.001):
        torch.autograd.set_detect_anomaly(True)
        if torch.cuda.is_available():
            self.cuda()
        data = data.type(torch.FloatTensor)
        data = data.to(self.device)
        data = data.view(-1, self.layers[0])
        z_init = self.encode(data)
        kmeans = KMeans(n_clusters=num_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z_init.data.cpu().numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        self.mu.data.copy_(cluster_centers)
        y_pred_previous = y_pred
        print('==========Training Deep Embedding Clustering Model==========')
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        '''
        if labels is not None:
            labels = labels.cpu().numpy()
            print('Clustering Accuracy: {:.3f}, NMI Score: {:.3f}'.format(accuracy(y_pred, labels)[0],
                                                                          nmi(labels, y_pred)))
        '''

        n = data.shape[0]
        num_batches = int(math.ceil(n/batch_size))
        self.train()
        for epoch in range(num_epochs):
            q, z = self.forward(data)
            p = target_distribution(q).data
            _, y_pred = torch.max(q, dim=1)
            y_pred = y_pred.data.cpu().numpy()
            error = np.sum(y_pred != y_pred_previous) / n
            print('Current Error: {:.3f}'.format(error))
            if error < stop_criterion and epoch > 0:
                print('Difference in Cluster predictions: '
                      'error={:.3f} < stop_criterion={:.3f}'.format(error, stop_criterion))
                print('Convergence Threshold is reached. Training will be stopped.')
                break

            # batch-wise training
            loss = 0.0
            for batch_num in range(num_batches):
                x_batch = data[batch_num*batch_size:min((batch_num+1)*batch_size, n)]
                p_batch = p[batch_num*batch_size:min((batch_num+1)*batch_size, n)]

                optimizer.zero_grad()

                q_batch, z_batch = self.forward(x_batch)
                kl_loss = kullback_leibler(p_batch, q_batch)
                loss += kl_loss.data * len(x_batch)
                kl_loss.backward(retain_graph=True)
                optimizer.step()
            if labels is not None:
                if self.accuracy == 'accuracy':
                    print('Epoch No. {}: Loss: {:.3f}, Accuracy: {:.3f}, NMI Score: {:.3f}'
                          .format(epoch, loss / n, accuracy(y_pred, labels)[0], nmi(labels, y_pred)))
                elif self.accuracy == 'f1':
                    print('Epoch No. {}: Loss: {:.3f}, Accuracy: {:.3f}, NMI Score: {:.3f}'
                          .format(epoch, loss / n, f1_binary(y_pred, labels), nmi(labels, y_pred)))
            else:
                print('Epoch No. {}: Loss: {:.3f}'.format(epoch, loss / n))

    def predict(self, x, batch_size=256):
        dataloader = DataLoader(x, batch_size, shuffle=False)
        embeddings = []
        for batch in tqdm(dataloader, unit='batch'):
            batch.to(self.device)
            embeddings.append(self.forward(batch).detach().cpu())
        return torch.cat(embeddings).max(dim=1)
