import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from models.base_model import BaseModel


# Wrapper around scikit-learn LogisticRegression with optional leak-free standardization
class LinearModel(BaseModel):
    def __init__(self, seed=0, l1_ratio=0, C=1.0, standardize=True, max_iter=1000):
        solver = "saga" if l1_ratio > 0 else "lbfgs"
        classifier = LogisticRegression(
            random_state=seed,
            l1_ratio=l1_ratio,
            C=C,
            solver=solver,
            max_iter=max_iter,
        )

        if standardize:
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ])
        else:
            self.model = classifier

    @staticmethod
    def _to_numpy(arr):
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _prepare_x(self, x):
        """Convert single tensor/array or (spatial, metadata) tuple to a 2D numpy array."""
        if isinstance(x, tuple):
            parts = [self._to_numpy(part).reshape(len(part), -1) for part in x]
            return np.hstack(parts)
        return self._to_numpy(x)

    def fit(self, x, y):
        """
        Fit the linear model for binary classification.

        x: array/tensor of shape (n_samples, n_features) or tuple (x_main, x_meta).
        y: array/tensor of shape (n_samples,).
        """
        self.model.fit(self._prepare_x(x), self._to_numpy(y))

    def predict(self, x):
        """Predict binary class labels for samples in x."""
        return self.model.predict(self._prepare_x(x))

    def score(self, x, y):
        """Return classification accuracy on provided data and labels."""
        return self.model.score(self._prepare_x(x), self._to_numpy(y))