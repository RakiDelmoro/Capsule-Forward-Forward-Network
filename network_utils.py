import torch
import statistics
import numpy as np
from layer import ff_layer
from features import GREEN, RESET, RED

def initialize_layers_and_parameters(layers_feature_sizes, activation_function, device):
    layers = []
    layers_parameters = []

    for size in range(len(layers_feature_sizes)-1):
        input_feature = layers_feature_sizes[size] + 1
        output_feature = layers_feature_sizes[size+1]
        layer, w, b = ff_layer(input_feature, output_feature, activation_function, device)
        layers.append(layer)
        layers_parameters.extend([[w, b]])

    return layers, layers_parameters

def input_feature_view(x: torch.Tensor, capsule_tall):
    assert x.shape[-1] % capsule_tall == 0, f'{capsule_tall} should divisible by {x.shape[-1]}'
    input_view = x.shape[-1] // capsule_tall
    new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
    return x.view(new_input_shape)

def rotate_feature(capsule_outputs: list, rotation_amount: int, layer_idx: int):
    if layer_idx % 2 == 0:
        capsule_outputs.append(capsule_outputs[0][:, :rotation_amount])
        return torch.concat(capsule_outputs, dim=1)[:, rotation_amount:]
    else:
        capsule_outputs.insert(0, capsule_outputs[-rotation_amount][:, -rotation_amount:])
        return torch.concat(capsule_outputs, dim=1)[:, :-rotation_amount]

def print_correct_prediction(correct_prediction_list, amount_to_print):
    print(f"{GREEN}Correct prediction!{RESET}")
    for i in range(amount_to_print):
        each_item = correct_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_wrong_prediction(wrong_prediction_list, amount_to_print):
    print(f"{RED}Wrong prediction!{RESET}")
    for i in range(amount_to_print):
        each_item = wrong_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_percentile_of_correct_probabilities(probabilities_list):
    tenth_percentile = np.percentile(probabilities_list, 10)
    ninetieth_percentile = np.percentile(probabilities_list, 90)
    average = statistics.fmean(probabilities_list)

    print(f"Average: {average} Tenth percentile: {tenth_percentile} Ninetieth percentile: {ninetieth_percentile}")
