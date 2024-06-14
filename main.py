import torch
from torch.utils.data import DataLoader
from capsule_network import capsule_neural_network
from data_utils import load_data_to_memory, get_training_data, get_validation_data

def main():
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.01
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    image_for_train, expected_for_train, image_for_validate, expected_for_validate = load_data_to_memory('./training-data/mnist.pkl.gz')
    training_data = get_training_data(image_for_train, expected_for_train, 10)
    images_with_combined_labels, labels = get_validation_data(image_for_validate, expected_for_validate)

    input_feature_size = HEIGHT * WIDTH
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    hidden_layers = [100] * 51
    model_runner = capsule_neural_network(capsule_feature_size=hidden_layers, input_feature=input_feature_size, threshold=2.0, activation_function=torch.nn.functional.relu, lr=LEARNING_RATE, device="cuda", capsule_tall=1, capsule_wide=1, rotation_amount=1)
    model_runner(train_loader, images_with_combined_labels, labels)

main()
