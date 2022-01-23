import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from torchmetrics import MetricCollection, AUROC, Accuracy, Precision, Recall, F1
import math
from models.resnet import ResidualBlock


class ResNet(pl.LightningModule):
    def __init__(self,
                 in_channels: int = 1,
                 output_dim: int = 512,
                 num_blocks: int = 3,
                 do_batchnorm: bool = True,
                 feature_maps: list = None,
                 kernel_sizes: list = None,
                 dropout: float = 0.3,
                 activation: nn.Module = None):
        super(ResNet, self).__init__()
        if feature_maps is None:
            self.feature_maps = [64, 128, 128]
        else:
            self.feature_maps = feature_maps

        if activation is None:
            self.activate = nn.ReLU()
        else:
            self.activate = activation

        self.num_channels = in_channels
        self.out_dim = output_dim
        self.kernel_sizes = kernel_sizes
        self.do_batchnorm = do_batchnorm
        self.p_drop = dropout
        self.activate = nn.ReLU()

        self.RESIDUAL_BLOCKS = self.build_blocks(in_channels=self.num_channels, num_blocks=num_blocks)
        self.Dropout = nn.Dropout(p=dropout)
        self.FINAL_LAYER = nn.Sequential(nn.AdaptiveAvgPool1d(self.feature_maps[-1]),
                                         nn.Flatten(),
                                         nn.Linear(self.feature_maps[-1] ** 2, self.out_dim))

    def forward(self, x):
        x = self.FINAL_LAYER(self.RESIDUAL_BLOCKS(x))
        x = self.Dropout(x)
        return x

    def build_blocks(self, in_channels, num_blocks):
        layers = []
        num_channels = in_channels
        for block_num in range(num_blocks):
            num_kernels = self.feature_maps[block_num]
            layers.append(("BLOCK_#{}".format(block_num + 1), ResidualBlock(in_channels=num_channels,
                                                                            batchnorm=self.do_batchnorm,
                                                                            n_feature_maps=num_kernels,
                                                                            kernel_sizes=self.kernel_sizes,
                                                                            activation=self.activate)))
            num_channels = num_kernels
        return nn.Sequential(OrderedDict(layers))


class ALSTM(pl.LightningModule):
    def __init__(self,
                 feature_dim: int = 1750,
                 num_heads: int = 8,
                 d_inner: int = 1500,
                 lstm_hidden: int = 500,
                 lstm_layer: int = 10,
                 num_channels: int = 1,
                 dropout: float = 0.1,
                 ):
        super(ALSTM, self).__init__()
        self.num_channels = num_channels
        self.lstm_layer_dim = lstm_layer
        self.lstm_hidden_dim = lstm_hidden
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=lstm_hidden, num_layers=lstm_layer, batch_first=True)
        #self.posEncoder = PositionalEncoding(d_model=lstm_hidden, dropout=dropout)
        #self.attention = nn.TransformerEncoderLayer(nhead=num_heads, d_model=lstm_hidden,
        #                                            dim_feedforward=d_inner, dropout=dropout)
        self.pooling = nn.AdaptiveAvgPool1d(256)
        self.fc_lstm = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

    def init_lstm(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(self.lstm_layer_dim, x.shape[0], self.lstm_hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.lstm_layer_dim, x.shape[0], self.lstm_hidden_dim).requires_grad_().to(device)
        return h0, c0

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.num_channels))
        h0, c0 = self.init_lstm(x)
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        #x = self.posEncoder(x)
        #x = self.attention(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], -1))
        x = self.pooling(x)
        x = self.fc_lstm(x)
        x = torch.squeeze(x)
        x = self.dropout(x)
        return x


class ALSTM_ResNet(pl.LightningModule):
    def __init__(self,
                 num_classes:   int = 2,
                 feature_dim:   int = 1750,
                 num_heads:     int = 5,
                 d_inner:       int = 1500,
                 lstm_hidden:   int = 500,
                 lstm_layer:    int = 10,
                 num_channels:  int = 1,
                 num_blocks:    int = 3,
                 lr:            float = 0.001,
                 dropout:       float = 0.1,
                 lstm_drop:     float = 0.1,
                 resnet_drop:   float = 0.1,
                 weight_decay:  float = 0.0,
                 do_batchnorm:  bool = True,
                 feature_maps:  list = None,
                 kernel_sizes:  list = None,
                 class_weights: list = None,
                 optimizer:     str = 'Adam'
                 ):
        super(ALSTM_ResNet, self).__init__()
        self.dropout = dropout
        self.activate = nn.ReLU()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.optimizer = optimizer

        self.ALSTM = ALSTM(feature_dim=feature_dim,
                           num_heads=num_heads,
                           d_inner=d_inner,
                           lstm_hidden=lstm_hidden,
                           lstm_layer=lstm_layer,
                           num_channels=num_channels,
                           dropout=lstm_drop)

        self.ResNet = ResNet(in_channels=num_channels,
                             output_dim=lstm_hidden,
                             num_blocks=num_blocks,
                             do_batchnorm=do_batchnorm,
                             feature_maps=feature_maps,
                             kernel_sizes=kernel_sizes,
                             dropout=resnet_drop,
                             activation=nn.ReLU())

        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden*2, num_classes)
        self.softmax = nn.Softmax(dim=1)

        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)

        metrics = MetricCollection({
            'F1': F1(num_classes=num_classes, ignore_index=0),
            'Precision': Precision(num_classes=num_classes, ignore_index=0),
            'Recall': Recall(num_classes=num_classes, ignore_index=0),
            'Accuracy': Accuracy(num_classes=num_classes),
            'AUROC': AUROC(num_classes=num_classes)
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], self.num_channels, -1))
        #x = torch.reshape(x, (x.shape[0], self.num_channels))
        x_lstm = self.ALSTM(x)
        x_cnn = self.ResNet(x)
        x = torch.cat((x_lstm, x_cnn), dim=1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                  weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                      weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y = y.type_as(logits)
        loss = self.loss(logits, y)
        y_hat = self.softmax(logits)
        y = torch.argmax(y, dim=1).int()
        metrics = self.train_metrics(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_f1', metrics['train_F1'], on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y = y.type_as(logits)
        loss = self.loss(logits, y)
        y_hat = self.softmax(logits)
        y = torch.argmax(y, dim=1).int()
        metrics = self.val_metrics(y_hat, y)
        metrics['val_loss'] = loss
        return OrderedDict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y = y.type_as(logits)
        loss = self.loss(logits, y)
        y = torch.argmax(y, dim=1).int()
        y_hat = self.softmax(logits)
        metrics = self.test_metrics(y_hat, y)
        metrics['test_loss'] = loss
        return OrderedDict(metrics)

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['val_Accuracy'] for x in outputs]).mean()
        precision = torch.stack([x['val_Precision'] for x in outputs]).mean()
        recall = torch.stack([x['val_Recall'] for x in outputs]).mean()
        f1 = torch.stack([x['val_F1'] for x in outputs]).mean()
        auroc = torch.stack([x['val_AUROC'] for x in outputs]).mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.log("val_auroc", auroc, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['test_Accuracy'] for x in outputs]).mean()
        precision = torch.stack([x['test_Precision'] for x in outputs]).mean()
        recall = torch.stack([x['test_Recall'] for x in outputs]).mean()
        f1 = torch.stack([x['test_F1'] for x in outputs]).mean()
        auroc = torch.stack([x['test_AUROC'] for x in outputs]).mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)
        self.log("test_auroc", auroc)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        x = x.view(x.size(0), -1)
        return self.forward(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoding(pl.LightningModule):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

