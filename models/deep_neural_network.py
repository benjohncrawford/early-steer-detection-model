from math import floor
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.base_model import BaseModel
import copy


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
        val_fraction=0.1
    ):
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

    def _to_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.detach().to(device=self.device, dtype=dtype)
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def _build_network(self, input_dim):
        layers = []
        prev_dim = input_dim

        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, x, y):
        """
        Fit the model for later predictions

        x: numpy array or torch tensor of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.

        y: numpy array or torch tensor of shape (n_samples,)
        Target vector relative to X.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_t = self._to_tensor(x)
        y_t = self._to_tensor(y)

        self.classes_, y_idx = torch.unique(y_t, sorted=True, return_inverse=True)
        y_bin = y_idx.float().unsqueeze(1)

        if self.standardize:
            self.mean_ = x_t.mean(dim=0, keepdim=True)
            self.std_ = x_t.std(dim=0, keepdim=True) + 1e-8
            x_t = (x_t - self.mean_) / self.std_

        self.model = self._build_network(x_t.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        val_size = max(1, int(len(x_t) * self.val_fraction))
        indices = torch.randperm(len(x_t))
        val_idx, train_idx = indices[:val_size], indices[val_size:]
        
        x_train, y_train = x_t[train_idx], y_bin[train_idx]
        x_val, y_val = x_t[val_idx], y_bin[val_idx]


        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_loss = float('inf')
        epochs_without_improvement = 0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                
            avg_train_loss = running_train_loss / len(loader)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x_val)
                val_loss = criterion(val_logits, y_val).item()

            print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:

                    print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights.")
                    self.model.load_state_dict(best_model_weights)
                    break


    def predict(self, x):
        """
        Predict class labels for samples in x.

        x: numpy array or torch tensor of shape (n_samples, n_features)
        """
        is_tensor = isinstance(x, torch.Tensor)
        x_t = self._to_tensor(x)

        if self.standardize:
            x_t = (x_t - self.mean_) / self.std_

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_t).squeeze(1)
            preds = self.classes_[(logits > 0.0).long()]

        return preds if is_tensor else preds.cpu().numpy()

    def k_fold_val(self, k, x, y):
        """
        Perform K-Fold Cross Validation returns the average accuracy over k folds
        """
        tot_samples = len(y)
        val_samples_per_split = floor(tot_samples / k)

        acc_by_fold = []
        for i in range(k):
            start = i * val_samples_per_split
            end = (i + 1) * val_samples_per_split

            # Create mask of True values covering each row, flip the rows we are not including
            mask = np.ones(x.shape[0], dtype=bool)
            mask[start:end] = False

            # Train on all rows except those in val set
            fold_train_x = x[mask]
            fold_train_labels = y[mask]

            # Test on rows in the current fold
            fold_test_x = x[~mask]
            fold_test_labels = y[~mask]

            self.fit(fold_train_x, fold_train_labels)

            acc_by_fold.append(self.score(fold_test_x, fold_test_labels))

        return sum(acc_by_fold) / k

    def score(self, x, y):
        """
        Return accuracy on provided data and labels.

        x: numpy array or torch tensor of shape (n_samples, n_features)
        y: numpy array or torch tensor of shape (n_samples,)
        """
        preds = self._to_tensor(self.predict(x))
        targets = self._to_tensor(y)
        return (preds == targets).float().mean().item()