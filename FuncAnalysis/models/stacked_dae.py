import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.dae import DenoisingAE, LitDenoisingAE
from utils.utils import Dataset, corrupt_signal, calc_epochs, set_dropout, corrupt_input
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


class StackedDenoisingAE(nn.Module):
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: int = 0,
                 corruption_type: str = 'mask'):
        super(StackedDenoisingAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.corruption_type = corruption_type
        if hidden_dims is None:
            hidden_dims = [500, 200]

        self.layers = [input_dim] + hidden_dims + [latent_dim]
        enc_modules = []
        input_channels = input_dim
        for h_dim in hidden_dims:
            submodule = [("linear", nn.Linear(input_channels, h_dim)), ("activation", nn.ReLU())]
            if self.dropout > 0:
                submodule.append(("dropout", nn.Dropout(self.dropout)))
            input_channels = h_dim
            enc_modules.append(nn.Sequential(OrderedDict(submodule)))
        enc_modules.append(nn.Sequential(OrderedDict([("linear", nn.Linear(hidden_dims[-1], latent_dim))])))
        self.encoder = nn.Sequential(*enc_modules)

        dec_modules = [nn.Sequential(OrderedDict([("linear", nn.Linear(latent_dim, hidden_dims[-1])),
                                                  ("activation", nn.ReLU())]))]
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            submodule = [("linear", nn.Linear(hidden_dims[i], hidden_dims[i + 1])), ("activation", nn.ReLU())]
            if self.dropout > 0:
                submodule.append(("dropout", nn.Dropout(self.dropout)))
            dec_modules.append(nn.Sequential(OrderedDict(submodule)))

        dec_modules.append(nn.Sequential(OrderedDict([("linear", nn.Linear(hidden_dims[-1], input_dim))])))
        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))

    def get_layer_pairs(self, index):
        assert (index >= 0) and (index <= len(self.layers)-2), 'Layer Index out of range.'
        encoder_layer = self.encoder[index].linear
        decoder_layer = self.decoder[-(1+index)].linear
        return encoder_layer, decoder_layer

    def pretrain(self, train_loader, val_loader, lr=0.1, num_epochs=12, dropout=0.2, corrupt=0.2, batch_size=256):
        for layer in range(len(self.layers) - 1):
            in_channels = self.layers[layer]
            out_channels = self.layers[layer + 1]
            if layer == 0:
                dae = DenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                  enc_activation='relu', dec_activation=None, corruption_type=self.corruption_type)
            elif layer == len(self.layers) - 1:
                dae = DenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                  enc_activation=None, dec_activation='relu', corruption_type=self.corruption_type)
            else:
                dae = DenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                  enc_activation='relu', dec_activation='relu', corruption_type=self.corruption_type)

            dae.fit(train_loader, val_loader, lr=lr, num_epochs=num_epochs, corrupt=corrupt, batch_size=batch_size)
            encoder, decoder = self.get_layer_pairs(index=layer)
            dae.copy_weights(encoder, decoder)

            transformed_train = dae.transform_data(train_loader)
            transformed_val = dae.transform_data(val_loader)
            training_dataset = Dataset(transformed_train, transformed_train)
            validation_dataset = Dataset(transformed_val, transformed_val)
            train_loader = DataLoader(training_dataset, batch_size=256, shuffle=True, num_workers=0)
            val_loader = DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=0)

    def fit(self, train_loader, val_loader, lr=0.1, num_epochs=12, corrupt=0.2, batch_size=256, dropout=0.0):
        if dropout != self.dropout:
            self.dropout = dropout
            set_dropout(self, drop_rate=dropout)
        if torch.cuda.is_available():
            self.cuda()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        update_every_epochs = calc_epochs(20000, batch_size=batch_size, num_data=int(batch_size * len(train_loader)))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=update_every_epochs, gamma=0.1)
        criterion = nn.MSELoss()
        print('==========Training Stacked Denoising Autoencoder (SDAE)==========')
        self.train()
        for epoch in range(num_epochs):
            training_loss = 0.0
            for batch_num, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                x = x.view(x.shape[0], -1).float()
                if corrupt > 0:
                    if self.corruption_type == 'mask':
                        x_corrupt = corrupt_input(x, corrupt)
                    elif self.corruption_type == 'gaussian':
                        x_corrupt = corrupt_signal(x, corrupt=corrupt, mean=0.0, var=0.4, normalize=True)
                    else:
                        raise KeyError('Type of corruption is not known.')
                else:
                    x_corrupt = x.detach().clone()
                optimizer.zero_grad()
                x_recon = self.forward(x_corrupt)
                recon_loss = criterion(x, x_recon)
                recon_loss.backward()
                optimizer.step()
                training_loss += recon_loss.data * len(x)

            validation_loss = 0.0
            for batch_num, (x, _) in enumerate(val_loader):
                x = x.to(self.device)
                x = x.view(x.shape[0], -1).float()
                x_recon = self.forward(x)
                recon_loss = criterion(x, x_recon)
                validation_loss += recon_loss.data * len(x)

            scheduler.step()

            print('Epoch No. {} --- Training Loss: {:.3f}, Validation Loss: {:.3f}'
                  .format(epoch,
                          training_loss / len(train_loader.dataset),
                          validation_loss / len(val_loader.dataset)))


class LitStackedDenoisingAE(pl.LightningModule):
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: int = 0,
                 learning_rate: float = 0.1,
                 corruption_type: str = 'mask',
                 corrupt: float = 0.2,
                 optimizer: str = 'SGD'):
        super(LitStackedDenoisingAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optimizer
        self.corrupt = corrupt
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.corruption_type = corruption_type
        self.hidden_dims = hidden_dims
        self.layers = [input_dim] + self.hidden_dims + [self.latent_dim]
        enc_modules = []
        input_channels = input_dim
        hidden_dims = self.hidden_dims
        for h_dim in hidden_dims:
            submodule = [("linear", nn.Linear(input_channels, h_dim)), ("activation", nn.ReLU())]
            if self.dropout > 0:
                submodule.append(("dropout", nn.Dropout(self.dropout)))
            input_channels = h_dim
            enc_modules.append(nn.Sequential(OrderedDict(submodule)))
        enc_modules.append(nn.Sequential(OrderedDict([("linear", nn.Linear(hidden_dims[-1], self.latent_dim))])))
        self.encoder = nn.Sequential(*enc_modules)

        dec_modules = [nn.Sequential(OrderedDict([("linear", nn.Linear(self.latent_dim, hidden_dims[-1])),
                                                  ("activation", nn.ReLU())]))]
        hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            submodule = [("linear", nn.Linear(hidden_dims[i], hidden_dims[i + 1])), ("activation", nn.ReLU())]
            if self.dropout > 0:
                submodule.append(("dropout", nn.Dropout(self.dropout)))
            dec_modules.append(nn.Sequential(OrderedDict(submodule)))

        dec_modules.append(nn.Sequential(OrderedDict([("linear", nn.Linear(hidden_dims[-1], input_dim))])))
        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))

    def get_layer_pairs(self, index):
        assert (index >= 0) and (index <= len(self.layers)-2), 'Layer Index out of range.'
        encoder_layer = self.encoder[index].linear
        decoder_layer = self.decoder[-(1+index)].linear
        return encoder_layer, decoder_layer

    def pretrain(self, train_loader, val_loader, trial, lr=0.1, dropout=0.2, max_epochs=250, batch_size=64):
        for layer in range(len(self.layers) - 1):
            in_channels = self.layers[layer]
            out_channels = self.layers[layer + 1]
            if layer == 0:
                dae = LitDenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                     learning_rate=lr, enc_activation='relu', dec_activation=None,
                                     corruption_type=self.corruption_type, corrupt=self.corrupt,
                                     optimizer=self.optimizer)
            elif layer == len(self.layers) - 1:
                dae = LitDenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                     learning_rate=lr, enc_activation=None, dec_activation='relu',
                                     corruption_type=self.corruption_type, corrupt=self.corrupt,
                                     optimizer=self.optimizer)
            else:
                dae = LitDenoisingAE(in_channels=in_channels, out_channels=out_channels, dropout=dropout,
                                     learning_rate=lr, enc_activation='relu', dec_activation='relu',
                                     corruption_type=self.corruption_type, corrupt=self.corrupt,
                                     optimizer=self.optimizer)
            logger = TensorBoardLogger('tb_logs', 'DAE_Tune')
            trainer = pl.Trainer(gpus=1,
                                 max_epochs=max_epochs,
                                 logger=logger,
                                 auto_lr_find=True,
                                 callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')])
            trainer.fit(dae, train_loader, val_loader)
            encoder, decoder = self.get_layer_pairs(index=layer)
            dae.copy_weights(encoder, decoder)

            transformed_train = dae.transform_data(train_loader)
            transformed_val = dae.transform_data(val_loader)
            training_dataset = Dataset(transformed_train, transformed_train)
            validation_dataset = Dataset(transformed_val, transformed_val)
            train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        x, _ = batch
        x = x.to(self.device)
        x = x.view(x.shape[0], -1).float()
        if self.corrupt > 0:
            if self.corruption_type == 'mask':
                x_corrupt = corrupt_input(x, self.corrupt)
            elif self.corruption_type == 'gaussian':
                x_corrupt = corrupt_signal(x, 0.0, self.corrupt)
            else:
                raise KeyError('Type of corruption is not known.')
        else:
            x_corrupt = x.detach().clone()
        x_recon = self.forward(x_corrupt)
        recon_loss = criterion(x, x_recon)
        self.log('train_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        criterion = nn.MSELoss()
        x, _ = batch
        x = x.to(self.device)
        x = x.view(x.shape[0], -1).float()
        x_recon = self.forward(x)
        recon_loss = criterion(x, x_recon)
        self.log('val_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        return recon_loss

    '''
        def training_epoch_end(self, outputs):
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            tensorboard_logs = {'train_loss': avg_loss}
            return {'loss': avg_loss, 'log': tensorboard_logs}

        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'loss': avg_loss, 'log': tensorboard_logs}
        '''
