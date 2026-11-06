from math import floor
import numpy as np

# Idea for the class is to serve largely as a wrapper around sklearn model classes 
# Can extend functionality and also provide the same interface even for torch models
class BaseModel:
    def __init__(self, **kwargs):
        pass
    
    def fit(self, x, y, **kwargs):
        pass
    
    def predict(self, x, **kwargs):
        pass
    
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
            
            self.fit(fold_train_x, fold_train_labels)
            
            acc_by_fold.append(self.score(fold_test_x, fold_test_labels))
            
            print(f"Fold {i+1}/{k} completed - accuracy: {acc_by_fold[-1]}")
    
        return (sum(acc_by_fold) / k)
    
    def score(self, x, y, **kwargs):
        pass
    
