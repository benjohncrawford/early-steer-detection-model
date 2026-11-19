import torch
import torch.nn as nn
from models.deep_neural_network import DeepNeuralNetwork


class _CNNWithMetadata(nn.Module):
    """Applies a Conv2d backbone to spatial maps and fuses tabular metadata in the dense head."""

    def __init__(self, conv_backbone, head):
        super().__init__()
        self.conv_backbone = conv_backbone
        self.head = head

    def forward(self, spatial_x, meta_x=None):
        features = self.conv_backbone(spatial_x)
        if meta_x is not None:
            features = torch.cat([features, meta_x], dim=1)
        return self.head(features)


class ConvolutionalNeuralNetwork(DeepNeuralNetwork):
    def __init__(
        self,
        seed=0,
        conv_channels=(32, 64, 128),
        kernel_size=3,
        pool_output_size=4,
        hidden_layers=(64, 32),
        dropout=0.2,
        standardize=True,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        patience=5,
        val_fraction=0.1,
    ):
        """
        conv_channels: tuple of ints specifying output channels for each conv block.
        kernel_size: size of the convolving kernel (e.g., 3 for a 3x3 filter).
        pool_output_size: spatial H and W output size of AdaptiveAvgPool2d before flattening.
        hidden_layers: tuple of ints specifying the dense layers after flattening.
        dropout: dropout probability applied in the dense classification head.
        """
        super().__init__(
            seed=seed,
            hidden_layers=hidden_layers,
            standardize=standardize,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            val_fraction=val_fraction,
        )
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_output_size = pool_output_size
        self.dropout = dropout

    def _build_network(self, in_channels, meta_dim=0):
        conv_layers = []
        prev_channels = in_channels
        padding = self.kernel_size // 2

        # Convolutional feature extractor: e.g., (48, 48, 48) -> (32, 24, 24) -> (64, 12, 12) -> (128, 6, 6)
        for out_channels in self.conv_channels:
            conv_layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=self.kernel_size, padding=padding)
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        # Pool to a fixed (pool_output_size x pool_output_size) grid so any input HxW works
        conv_layers.append(nn.AdaptiveAvgPool2d((self.pool_output_size, self.pool_output_size)))
        conv_layers.append(nn.Flatten())
        conv_backbone = nn.Sequential(*conv_layers)

        # Fully connected classification head combining pooled spatial features and metadata
        head_layers = []
        prev_dim = prev_channels * self.pool_output_size * self.pool_output_size + meta_dim
        for h_dim in self.hidden_layers:
            if self.dropout > 0:
                head_layers.append(nn.Dropout(self.dropout))
            head_layers.append(nn.Linear(prev_dim, h_dim))
            head_layers.append(nn.ReLU())
            prev_dim = h_dim

        head_layers.append(nn.Linear(prev_dim, 1))
        head = nn.Sequential(*head_layers)

        return _CNNWithMetadata(conv_backbone, head).to(self.device)