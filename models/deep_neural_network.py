import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel


class _MLPWithMetadata(nn.Module):
    """Concatenates flat primary features and metadata before passing into an MLP."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, meta=None):
        if meta is not None:
            x = torch.cat([x, meta], dim=1)
        return self.mlp(x)


class DeepNeuralNetwork(BaseModel):
    def __init__(
        self,
        seed=0,
        hidden_layers=(64, 32),
        standardize=True,
        lr=1e-3,
        epochs=100,
        batch_size=32,
        patience=5,
        val_fraction=0.1,
    ):
        """
        hidden_layers: tuple of ints specifying hidden layer neuron counts (e.g., (128, 64, 32)).
        standardize: whether to standardize inputs using training-split mean and std.
        lr: learning rate for the Adam optimizer.
        epochs: maximum number of training passes over the dataset.
        batch_size: mini-batch size for training.
        patience: epochs without validation loss improvement before early stopping.
        seed: random seed for reproducibility.
        val_fraction: fraction of training data reserved for early stopping.
        """
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.standardize = standardize
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_ = None
        self.mean_ = None
        self.std_ = None
        self.meta_mean_ = None
        self.meta_std_ = None

    def _to_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.detach().to(device=self.device, dtype=dtype)
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def _unpack_inputs(self, x):
        """Accept either a single feature tensor/array or a (main_features, metadata) tuple."""
        if isinstance(x, tuple):
            x_main, x_meta = x
            return self._to_tensor(x_main), self._to_tensor(x_meta)
        return self._to_tensor(x), None

    @staticmethod
    def _compute_norm_stats(tensor, norm_dims):
        """Compute mean and safe std (constant features keep scale 1.0 to prevent blowup)."""
        mean = tensor.mean(dim=norm_dims, keepdim=True)
        std = tensor.std(dim=norm_dims, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return mean, std

    def _forward(self, x_main, x_meta=None):
        if x_meta is None:
            return self.model(x_main)
        return self.model(x_main, x_meta)

    def _build_network(self, input_dim, meta_dim=0):
        layers = []
        prev_dim = input_dim + meta_dim

        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        mlp = nn.Sequential(*layers)

        if meta_dim > 0:
            return _MLPWithMetadata(mlp).to(self.device)
        return mlp.to(self.device)

    def fit(self, x, y):
        """
        Fit the binary classification model.

        x: array/tensor of shape (n_samples, n_features) or (n_samples, C, H, W),
           or a tuple (x_main, x_meta) where x_meta has shape (n_samples, n_meta).
        y: array/tensor of shape (n_samples,) with binary labels.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_t, meta_t = self._unpack_inputs(x)
        y_t = self._to_tensor(y)

        self.classes_, y_idx = torch.unique(y_t, sorted=True, return_inverse=True)
        y_bin = y_idx.float().unsqueeze(1)

        val_size = max(1, int(len(x_t) * self.val_fraction))
        indices = torch.randperm(len(x_t))
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        if self.standardize:
            # Standardize per-feature for 2D inputs, or per-channel across (N, H, W) for 4D inputs
            norm_dims = 0 if x_t.ndim == 2 else (0, 2, 3)
            self.mean_, self.std_ = self._compute_norm_stats(x_t[train_idx], norm_dims)
            x_t = (x_t - self.mean_) / self.std_

            if meta_t is not None:
                self.meta_mean_, self.meta_std_ = self._compute_norm_stats(meta_t[train_idx], 0)
                meta_t = (meta_t - self.meta_mean_) / self.meta_std_

        x_train, y_train = x_t[train_idx], y_bin[train_idx]
        x_val, y_val = x_t[val_idx], y_bin[val_idx]

        if meta_t is None:
            meta_val = None
            dataset = TensorDataset(x_train, y_train)
        else:
            meta_train = meta_t[train_idx]
            meta_val = meta_t[val_idx]
            dataset = TensorDataset(x_train, meta_train, y_train)

        meta_dim = meta_t.shape[1] if meta_t is not None else 0
        self.model = self._build_network(x_t.shape[1], meta_dim=meta_dim)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        epochs_without_improvement = 0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            self.model.train()
            running_train_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                if meta_t is None:
                    batch_x, batch_y = batch
                    logits = self._forward(batch_x)
                else:
                    batch_x, batch_meta, batch_y = batch
                    logits = self._forward(batch_x, batch_meta)

                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(loader)

            self.model.eval()
            with torch.no_grad():
                val_logits = self._forward(x_val, meta_val)
                val_loss = criterion(val_logits, y_val).item()

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights.")
                    break

        self.model.load_state_dict(best_model_weights)

    def predict(self, x):
        """
        Predict binary class labels for samples in x.

        x: array/tensor of shape (n_samples, n_features) or (n_samples, C, H, W),
           or a tuple (x_main, x_meta).
        """
        is_tensor = (
            isinstance(x[0], torch.Tensor)
            if isinstance(x, tuple)
            else isinstance(x, torch.Tensor)
        )
        x_t, meta_t = self._unpack_inputs(x)

        if self.standardize:
            x_t = (x_t - self.mean_) / self.std_
            if meta_t is not None:
                meta_t = (meta_t - self.meta_mean_) / self.meta_std_

        self.model.eval()
        with torch.no_grad():
            logits = self._forward(x_t, meta_t).squeeze(1)
            preds = self.classes_[(logits > 0.0).long()]

        return preds if is_tensor else preds.cpu().numpy()

    def score(self, x, y):
        """Return classification accuracy on provided data and labels."""
        preds = self._to_tensor(self.predict(x))
        targets = self._to_tensor(y)
        return (preds == targets).float().mean().item()