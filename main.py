import numpy as np
from models.linear_model import LinearModel
from dataset.generation.synthetic_data_generator import SyntheticDataGenerator

# Synthetic data generation - will replace with real data prep
data = SyntheticDataGenerator(num_samples = 1000, num_features = 128*128*16)

model = LinearModel(l1_ratio=0, C=np.inf)

model.fit(data.train_x, data.train_y)
print(model.score(data.test_x, data.test_y))


