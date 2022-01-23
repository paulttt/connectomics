import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
import math


class FeatureAttention(pl.LightningModule):
    def __init__(self,
                 num_intervals: int = 26,
                 num_classes: int = 2,
                 feature_dim: int = 773,
                 num_heads: int = 5,
                 d_inner: int = 1500,
                 conv_maps: list = None,
                 lr: float = 0.001,
                 dropout: float = 0.1,
                 weight_decay: float = 1e-5,
                 class_weights: list = None
                 ):
        super(FeatureAttention, self).__init__()
        self.position = PositionalEncoding(d_model=feature_dim)
        self.attention = nn.TransformerEncoderLayer(nhead=num_heads, d_model=feature_dim,
                                                    dim_feedforward=d_inner, dropout=dropout)
        if conv_maps is None:
            self.conv_maps = [16]
        else:
            self.conv_maps = conv_maps

        conv_layers = []
        channel_in = num_intervals
        for i, map in enumerate(self.conv_maps):
            conv_layers.append(('Conv_{}'.format(i+1), nn.Conv1d(channel_in, map, 1)))
            conv_layers.append(('ReLu', nn.ReLU()))
            channel_in = map
        conv_layers.append(('Conv_{}'.format(len(self.conv_maps)+1), nn.Conv1d(channel_in, 1, 1)))
        conv_layers.append(('ReLu', nn.ReLU()))
        self.conv = nn.Sequential(OrderedDict(conv_layers))
        self.fc = nn.Linear(feature_dim, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.weight_decay = weight_decay
        self.learning_rate = lr
        self.num_classes = num_classes

        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        metrics = MetricCollection({
            'F1': F1(num_classes=num_classes, ignore_index=0),
            'Precision': Precision(num_classes=num_classes, ignore_index=0),
            'Recall': Recall(num_classes=num_classes, ignore_index=0),
            'Accuracy': Accuracy(num_classes=num_classes)
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        x = self.position(x)
        x = self.attention(x)
        x = self.conv(x)
        x = self.fc(x)
        return torch.squeeze(x, 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
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
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['test_Accuracy'] for x in outputs]).mean()
        precision = torch.stack([x['test_Precision'] for x in outputs]).mean()
        recall = torch.stack([x['test_Recall'] for x in outputs]).mean()
        f1 = torch.stack([x['test_F1'] for x in outputs]).mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_f1", f1)


class PositionalEncoding(nn.Module):
# source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
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
