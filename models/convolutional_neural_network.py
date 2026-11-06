import torch.nn as nn
from models.deep_neural_network import DeepNeuralNetwork


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
        val_fraction=0.1
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
            val_fraction=val_fraction
        )
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_output_size = pool_output_size
        self.dropout = dropout

    def _build_network(self, in_channels):
        layers = []
        prev_channels = in_channels
        padding = self.kernel_size // 2

        # Convolutional feature extractor: (C, 128, 128) -> (32, 64, 64) -> (64, 32, 32) -> (128, 16, 16)
        for out_channels in self.conv_channels:
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=self.kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        # Pool to a fixed (pool_output_size x pool_output_size) grid so any input HxW works
        layers.append(nn.AdaptiveAvgPool2d((self.pool_output_size, self.pool_output_size)))
        layers.append(nn.Flatten())

        # Fully connected classification head
        prev_dim = prev_channels * self.pool_output_size * self.pool_output_size
        for h_dim in self.hidden_layers:
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)