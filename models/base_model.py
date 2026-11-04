

# Idea for the class is to serve largely as a wrapper around sklearn model classes 
# Can extend functionality and also provide the same interface even for torch models
class BaseModel:
    def __init__(self, **kwargs):
        pass
    
    def fit(self, x, y, **kwargs):
        pass
    
    def predict(self, x, **kwargs):
        pass
    
    def k_fold_val(self, k, x, y, **kwargs):
        pass
    
    def score(self, x, y, **kwargs):
        pass
    
