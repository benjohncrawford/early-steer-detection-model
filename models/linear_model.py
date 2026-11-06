from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel
from sklearn.model_selection import KFold

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
                
    def score(self, x, y):
        """
        Return accuracy on provided data and labels.
        
        x: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,) 
        """

        return self.model.score(x, y)
    