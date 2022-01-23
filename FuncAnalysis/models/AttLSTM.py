import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, AUROC
import math


class ALSTM(pl.LightningModule):
    def __init__(self,
                 num_classes = 2,
                 feature_dim: int = 1750,
                 optimizer: str = 'Adam',
                 num_heads: int = 8,
                 d_inner: int = 1024,
                 pooling_dim: int = 256,
                 lstm_hidden: int = 128,
                 lstm_layer: int = 10,
                 fc_layer: int = 2048,
                 num_channels: int = 1,
                 lr: float = 0.001,
                 lstm_drop: float = 0.1,
                 dropout: float = 0.4,
                 weight_decay: float = 0.0,
                 class_weights: list = None
                 ):
        super(ALSTM, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lstm_layer_dim = lstm_layer
        self.lstm_hidden_dim = lstm_hidden
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=lstm_hidden, num_layers=lstm_layer, batch_first=True)
        self.posEncoder = PositionalEncoding(d_model=lstm_hidden)
        self.attention = nn.TransformerEncoderLayer(nhead=num_heads, d_model=lstm_hidden,
                                                    dim_feedforward=d_inner)
        self.FINAL_LAYER = nn.Sequential(nn.AdaptiveAvgPool1d(pooling_dim),
                                         nn.Flatten(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(pooling_dim * lstm_hidden, self.num_classes))
        self.softmax = nn.Softmax(dim=1)
        self.lstm_drop = nn.Dropout(lstm_drop)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = lr
        self.weight_decay = weight_decay

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

    def init_lstm(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(self.lstm_layer_dim, x.shape[0], self.lstm_hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.lstm_layer_dim, x.shape[0], self.lstm_hidden_dim).requires_grad_().to(device)
        return h0, c0

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.num_channels))
        h0, c0 = self.init_lstm(x)
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.posEncoder(x)
        x = self.attention(x)
        x = torch.reshape(x, (x.shape[0], x.shape[2], -1))
        x = self.FINAL_LAYER(x)
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
        x = self.forward(x)
        return self.softmax(x)


class PositionalEncoding(nn.Module):

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
