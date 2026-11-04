import numpy as np

class SyntheticDataGenerator():
    def __init__(self, num_samples=200, num_features=2):
        """
        Generates a linearly separable dataset.
        num_samples : Total number of data points.
        num_features: Number of features (dimensions) per point.
        """
        # Generate random normal data
        self.train_x = np.random.randn(num_samples, num_features)
        self.test_x = np.random.randn(num_samples, num_features)
        
        # Generate a random hyperplane defined by weights (w) and bias (b)
        w = np.random.randn(num_features)
        b = np.random.randn()
        
        # Determine the initial labels based on the hyperplane equation (w*X + b = 0)
        # np.sign returns -1 for negative values and 1 for positive values
        train_y_sign = np.sign(self.train_x.dot(w) + b)
        test_y_sign = np.sign(self.test_x.dot(w) + b)
        
        # Convert labels from {-1, 1} to standard binary labels {0, 1}
        self.train_y = np.where(train_y_sign == 1, 1, 0)
        self.test_y = np.where(test_y_sign == 1, 1, 0)
