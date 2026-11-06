import numpy as np
from models.linear_model import LinearModel
from models.deep_neural_network import DeepNeuralNetwork
from models.convolutional_neural_network import ConvolutionalNeuralNetwork
from dataset.diffusion_data_loader import BaseDataLoader, FlatDataLoader

train_data_folder = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\Train"
test_data_folder = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\Test"
label_json_filepath = "C:\\Users\\ben99\\Downloads\\sd3-numeracy-v2\\Results\\sd3-numeracy-v2-backpacks-reviews.json"

training_data = FlatDataLoader(train_data_folder, results_json_path=label_json_filepath, percent=90)
# testing_data = FlatDataLoader(test_data_folder, results_json_path=label_json_filepath, percent=90)

# training_data = BaseDataLoader(train_data_folder, results_json_path=label_json_filepath, percent=20)
# testing_data = BaseDataLoader(test_data_folder, results_json_path=label_json_filepath, percent=20)

training_data.load_data()
# testing_data.load_data()

# model = LinearModel()
model = DeepNeuralNetwork()
# model = ConvolutionalNeuralNetwork()

print(model.k_fold_val(10, training_data.data, training_data.labels))

# print(model.predict(testing_data.data))
# print(model.score(testing_data.data, testing_data.labels))


