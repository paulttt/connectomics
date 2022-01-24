import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import corrupt_signal, calc_epochs, print_curr_lr, corrupt_input
import pytorch_lightning as pl


class DenoisingAE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.2,
                 enc_activation='relu',
                 dec_activation='relu',
                 corruption_type='mask'):
        super(DenoisingAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.corruption_type = corruption_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc_weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.enc_bias = nn.Parameter(torch.Tensor(out_channels))
        self.dec_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.dec_bias = nn.Parameter(torch.Tensor(in_channels))

        if enc_activation == 'relu':
            self.enc_activation = nn.ReLU()
        elif enc_activation == 'sigmoid':
            self.enc_activation = nn.Sigmoid()
        else:
            self.enc_activation = None

        if dec_activation == 'relu':
            self.dec_activation = nn.ReLU()
        elif dec_activation == 'sigmoid':
            self.dec_activation = nn.Sigmoid()
        else:
            self.dec_activation = None

        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def encode(self, x, train_mode=True):
        if train_mode:
            self.dropout.train()
        else:
            self.dropout.eval()

        if self.enc_activation is None:
            return self.dropout(F.linear(x, self.enc_weight, self.enc_bias))
        else:
            return self.dropout(self.enc_activation(F.linear(x, self.enc_weight, self.enc_bias)))

    def transform_data(self, data):
        transformed = []
        for i, (x, _) in enumerate(data):
            x = x.view(x.shape[0], -1)
            x = x.to(self.device)
            transform = self.encode(x, train_mode=False)
            transformed.append(transform.data.cpu())
        return torch.cat(transformed, dim=0)

    def decode(self, h, train_mode=True):
        if train_mode:
            self.dropout.train()
        else:
            self.dropout.eval()

        if self.dec_activation is None:
            return F.linear(h, self.dec_weight, self.dec_bias)
        else:
            return self.dec_activation(F.linear(h, self.dec_weight, self.dec_bias))

    def forward(self, x):
        return self.decode(self.encode(x))

    def init_weights(self):
        nn.init.normal_(self.enc_weight, 0, 0.01)
        nn.init.normal_(self.enc_bias, 0, 0.01)
        nn.init.normal_(self.dec_weight, 0, 0.01)
        nn.init.normal_(self.dec_bias, 0, 0.01)

    def copy_weights(self, encoder, decoder):
        encoder.weight.data.copy_(self.enc_weight)
        encoder.bias.data.copy_(self.enc_bias)
        decoder.weight.data.copy_(self.dec_weight)
        decoder.bias.data.copy_(self.dec_bias)

    def fit(self, train_loader, val_loader, lr=0.1, num_epochs=12, corrupt=0.2, batch_size=256):
        if torch.cuda.is_available():
            self.cuda()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        update_every_epochs = calc_epochs(20000, batch_size=batch_size, num_data=int(batch_size*len(train_loader)))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=update_every_epochs, gamma=0.1)
        criterion = nn.MSELoss()
        print('==========Training Denoising AE Layer==========')
        self.train()
        for epoch in range(num_epochs):
            training_loss = 0
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
                x_corrupt = x_corrupt.to(self.device)
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
            print_curr_lr(10, epoch, optimizer)

            print('Epoch No. {} --- Training Loss: {:.3f}, Validation Loss: {:.3f}'
                  .format(epoch,
                          training_loss / len(train_loader.dataset),
                          validation_loss / len(val_loader.dataset)))


class LitDenoisingAE(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.2,
                 learning_rate=0.1,
                 enc_activation='relu',
                 dec_activation='relu',
                 corruption_type='mask',
                 corrupt=0.2,
                 optimizer='Adam'):
        super(LitDenoisingAE, self).__init__()
        self.machine = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.corruption_type = corruption_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.corrupt = corrupt
        self.optimizer = optimizer

        self.enc_weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.enc_bias = nn.Parameter(torch.Tensor(out_channels))
        self.dec_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.dec_bias = nn.Parameter(torch.Tensor(in_channels))

        if enc_activation == 'relu':
            self.enc_activation = nn.ReLU()
        elif enc_activation == 'sigmoid':
            self.enc_activation = nn.Sigmoid()
        else:
            self.enc_activation = None

        if dec_activation == 'relu':
            self.dec_activation = nn.ReLU()
        elif dec_activation == 'sigmoid':
            self.dec_activation = nn.Sigmoid()
        else:
            self.dec_activation = None

        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def encode(self, x, train_mode=True):
        if train_mode:
            self.dropout.train()
        else:
            self.dropout.eval()

        if self.enc_activation is None:
            return self.dropout(F.linear(x, self.enc_weight, self.enc_bias))
        else:
            return self.dropout(self.enc_activation(F.linear(x, self.enc_weight, self.enc_bias)))

    def transform_data(self, data):
        transformed = []
        for i, (x, _) in enumerate(data):
            x = x.view(x.shape[0], -1)
            x = x.to(self.machine)
            transform = self.encode(x, train_mode=False)
            transformed.append(transform.data.cpu())
        return torch.cat(transformed, dim=0)

    def decode(self, h, train_mode=True):
        if train_mode:
            self.dropout.train()
        else:
            self.dropout.eval()

        if self.dec_activation is None:
            return F.linear(h, self.dec_weight, self.dec_bias)
        else:
            return self.dec_activation(F.linear(h, self.dec_weight, self.dec_bias))

    def forward(self, x):
        return self.decode(self.encode(x))

    def init_weights(self):
        nn.init.normal_(self.enc_weight, 0, 0.01)
        nn.init.normal_(self.enc_bias, 0, 0.01)
        nn.init.normal_(self.dec_weight, 0, 0.01)
        nn.init.normal_(self.dec_bias, 0, 0.01)

    def copy_weights(self, encoder, decoder):
        encoder.weight.data.copy_(self.enc_weight)
        encoder.bias.data.copy_(self.enc_bias)
        decoder.weight.data.copy_(self.dec_weight)
        decoder.bias.data.copy_(self.dec_bias)

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
        x = x.to(self.machine)
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
        x = x.to(self.machine)
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
