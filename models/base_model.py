import numpy as np
from sklearn.model_selection import StratifiedKFold


# Serves as a unified interface for both sklearn and PyTorch models
class BaseModel:
    def __init__(self, **kwargs):
        pass

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x, **kwargs):
        pass

    @staticmethod
    def _slice_inputs(x, mask):
        """Slice either a single array/tensor or a tuple of (spatial, metadata) inputs."""
        if isinstance(x, tuple):
            return tuple(part[mask] for part in x)
        return x[mask]

    def k_fold_val(self, k, x, y, seed=0):
        """
        Perform stratified K-Fold Cross Validation and return average accuracy over k folds.
        """
        tot_samples = len(y)
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)

        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        acc_by_fold = []

        for i, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(tot_samples), y_np)):
            # Boolean mask works identically for numpy arrays and CPU torch tensors
            val_mask = np.zeros(tot_samples, dtype=bool)
            val_mask[val_idx] = True

            fold_train_x = self._slice_inputs(x, ~val_mask)
            fold_train_labels = y[~val_mask]

            fold_test_x = self._slice_inputs(x, val_mask)
            fold_test_labels = y[val_mask]

            self.fit(fold_train_x, fold_train_labels)
            acc = self.score(fold_test_x, fold_test_labels)
            acc_by_fold.append(acc)

            print(f"Fold {i + 1}/{k} completed - accuracy: {acc:.4f}")

        return sum(acc_by_fold) / k

    def score(self, x, y, **kwargs):
        pass