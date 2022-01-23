import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, F1, Accuracy, Precision, Recall, AUROC
import pytorch_lightning as pl
from collections import OrderedDict


class ResidualBlock(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 batchnorm=True,
                 n_feature_maps: int = 64,
                 kernel_sizes: list = None,
                 activation: nn = None):
        super(ResidualBlock, self).__init__()
        self.batchnorm = batchnorm

        if activation is None:
            self.activate = nn.ReLU()
        else:
            self.activate = activation

        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
            padding = [3, 2, 1]
        else:
            padding = []
            for kernel in kernel_sizes:
                padding.append((kernel+1)/2)
        submodule = []
        input_size = in_channels

        for i, kernel_size in enumerate(kernel_sizes):
            submodule.append(("Conv1D_{}".format(i+1), nn.Conv1d(input_size, n_feature_maps, kernel_size,
                                                                 padding=padding[i], padding_mode='reflect')))
            if batchnorm:
                submodule.append(("BatchNorm_{}".format(i+1), nn.BatchNorm1d(n_feature_maps)))
            if i < len(kernel_sizes)-1:
                submodule.append(("activation_{}".format(i+1), self.activate))
            input_size = n_feature_maps

        self.conv_part = nn.Sequential(OrderedDict(submodule))
        if in_channels != n_feature_maps:
            self.shortcut = nn.Sequential(OrderedDict(
                [("Conv1D", nn.Conv1d(in_channels, n_feature_maps, kernel_size=1, padding_mode='reflect')),
                 ("BatchNorm", nn.BatchNorm1d(n_feature_maps))]
            ))
        else:
            self.shortcut = nn.Sequential(OrderedDict([("BatchNorm", nn.BatchNorm1d(n_feature_maps))]))

    def forward(self, x):
        x_1 = self.conv_part(x)
        x_2 = self.shortcut(x)
        out = x_1 + x_2
        out = self.activate(out)
        return out


class ResNet(pl.LightningModule):
    def __init__(self,
                 num_channels: int = 1,
                 num_classes: int = 2,
                 num_blocks: int = 3,
                 feature_maps: list = None,
                 kernel_sizes: list = None,
                 activation: str = 'relu',
                 optimizer: str = 'Adam',
                 lr: float = 0.1,
                 class_weights: list = None,
                 weight_decay: float = 1e-5,
                 dropout: float = 0.0,
                 batchnorm: bool = True
                 ):
        super(ResNet, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = lr
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.do_batchnorm = batchnorm
        self.weight_decay = weight_decay

        if feature_maps is None:
            self.feature_maps = [64, 128, 128]
        else:
            self.feature_maps = feature_maps

        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = torch.ones(num_classes, dtype=torch.float32)

        if activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'prelu':
            self.activate = nn.PReLU()
        else:
            self.activate = nn.ReLU()

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

        self.RESIDUAL_BLOCKS = self.build_blocks(in_channels=num_channels, num_blocks=num_blocks)
        self.FINAL_LAYER = nn.Sequential(nn.AdaptiveAvgPool1d(self.feature_maps[-1]),
                                         nn.Flatten(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(self.feature_maps[-1]**2, self.num_classes))
        self.softmax = nn.Softmax(dim=1)

    def build_blocks(self, in_channels, num_blocks):
        layers = []
        num_channels = in_channels
        for block_num in range(num_blocks):
            num_kernels = self.feature_maps[block_num]
            layers.append(("BLOCK_#{}".format(block_num+1), ResidualBlock(in_channels=num_channels,
                                                                          batchnorm=self.do_batchnorm,
                                                                          n_feature_maps=num_kernels,
                                                                          kernel_sizes=self.kernel_sizes,
                                                                          activation=self.activate)))
            num_channels = num_kernels
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.FINAL_LAYER(self.RESIDUAL_BLOCKS(x))

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], 1, -1).float()
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
        x = x.view(x.shape[0], 1, -1).float()
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
        x = x.view(x.shape[0], 1, -1).float()
        logits = self.forward(x)
        y = y.type_as(logits)
        loss = self.loss(logits, y)
        y = torch.argmax(y, dim=1).int()
        y_hat = self.softmax(logits)
        metrics = self.test_metrics(y_hat, y)
        metrics['test_loss'] = loss
        return OrderedDict(metrics)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        x = x.view(x.shape[0], 1, -1).float()
        return self.forward(x)

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
        self.log("test_auroc", auroc, on_epoch=True, prog_bar=True)