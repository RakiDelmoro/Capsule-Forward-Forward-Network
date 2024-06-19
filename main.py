import torch
from mlp import multi_layer_perceptron, Mlp
from torch.utils.data import DataLoader, TensorDataset
from capsule_network import capsule_neural_network
from data_utils import load_data_to_memory, get_training_data, get_validation_data

def main():
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.01
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    image_for_train, expected_for_train, image_for_validate, expected_for_validate = load_data_to_memory('./training-data/mnist.pkl.gz')
    ff_training_data = get_training_data(image_for_train, expected_for_train, 10)
    images_with_combined_labels, labels = get_validation_data(image_for_validate, expected_for_validate)

    input_feature_size = HEIGHT * WIDTH
    back_prop_training_loader = DataLoader(TensorDataset(image_for_train, expected_for_train), batch_size=BATCH_SIZE)
    ff_training_loader = DataLoader(ff_training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    hidden_layers = [100] * 51
    ff_model_runner = capsule_neural_network(capsule_feature_size=hidden_layers, input_feature=input_feature_size, threshold=2.0, activation_function=torch.nn.functional.relu, lr=LEARNING_RATE, device="cuda", capsule_tall=1, capsule_wide=2, rotation_amount=1)
    back_prop_runner = multi_layer_perceptron([784, 2000, 2000])
    ff_model_runner(back_prop_training_loader, 1000)

main()
