from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from sklearn.model_selection import KFold
from math import floor
import numpy as np


# Pretty exact wrapper around the Logistic Regression class in sklearn
class LinearModel(BaseModel):
    def __init__(self, seed = 0, l1_ratio = 0, C = 1.0):
        if l1_ratio > 0:
            self.model = LogisticRegression(random_state=seed, l1_ratio=l1_ratio, C=C, solver = "saga")
        else:
            self.model = LogisticRegression(random_state=seed, l1_ratio=l1_ratio, C=C, solver = "lbfgs")
            
    def fit(self, x, y):
        """
        Fit the linear model for later predictions
        
        x: numpy array of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.

        y: numpy array of shape (n_samples,) 
        Target vector relative to X.
        """
        
        self.model.fit(x, y)
    
    def predict(self, x):
        """
        Predict class labels for samples in x.
        
        x: numpy array of shape (n_samples, n_features)
        """
        
        return self.model.predict(x)
    
    def k_fold_val(self, k, x, y):
        """
        Perform K-Fold Cross Validation returns the average accuracy over k folds
        """
        tot_samples = len(y)
        val_samples_per_split = floor(tot_samples / k)
        
        acc_by_fold = []
        for i in range(k):
            start = i*val_samples_per_split
            end = (i+1)*val_samples_per_split
            
            # Create mask of True values covering each row, flip the rows we are not including
            mask = np.ones(x.shape[0], dtype=bool)
            mask[start:end] = False

            # Train on all rows except those in val set
            fold_train_x = x[mask]
            fold_train_labels = y[mask]
            
            # Test on rows in the current fold
            fold_test_x = x[~mask]
            fold_test_labels = y[~mask]
            
            self.model.fit(fold_train_x, fold_train_labels)
            
            acc_by_fold.append(self.model.score(fold_test_x, fold_test_labels))
    
        return (sum(acc_by_fold) / k)
            
    def score(self, x, y):
        """
        Return accuracy on provided data and labels.
        
        x: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) 
        """

        return self.model.score(x, y)
    