from models.linear_model import LinearModel
from models.deep_neural_network import DeepNeuralNetwork
from models.convolutional_neural_network import ConvolutionalNeuralNetwork
from dataset.diffusion_data_loader import BaseDataLoader, FlatDataLoader

# Paths to cached PCA outputs produced by compute_pca_cache.py
train_data_folder = "D:\\sd35-numeracy-v3-share\\PCA_Cache\\Train"
test_data_folder = "D:\\sd35-numeracy-v3-share\\PCA_Cache\\Test"
label_json_filepath = "D:\\sd35-numeracy-v3-share\\reviews.json"

# Choose one of the two loader/model setups:
# 1) Flat loader for LinearModel or DeepNeuralNetwork
training_data = FlatDataLoader(
    train_data_folder,
    percent=20,
    results_json_path=label_json_filepath,
)
training_data.load_data()

testing_data = FlatDataLoader(
    test_data_folder,
    percent=20,
    results_json_path=label_json_filepath,
    class_names=training_data.class_names,
)
testing_data.load_data()

# 2) Spatial + metadata loader for ConvolutionalNeuralNetwork
# training_data = BaseDataLoader(
#     train_data_folder,
#     percent=20,
#     results_json_path=label_json_filepath,
# )
# training_data.load_data()
#
# testing_data = BaseDataLoader(
#     test_data_folder,
#     percent=20,
#     results_json_path=label_json_filepath,
#     class_names=training_data.class_names,
# )
# testing_data.load_data()

# model = LinearModel(C=0.01)
model = DeepNeuralNetwork()
# model = ConvolutionalNeuralNetwork()

print("CV Accuracy:", model.k_fold_val(10, training_data.data, training_data.labels))

# Refit on the full training split before final held-out test evaluation
# model.fit(training_data.data, training_data.labels)
# print("Test Accuracy:", model.score(testing_data.data, testing_data.labels))