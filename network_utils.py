import torch
import statistics
import numpy as np
from layer import ff_layer
from features import GREEN, RESET, RED

def initialize_capsule_layers_and_parameters(layers_feature_sizes, input_data_feature, capsule_tall, activation_function, device):
    layers = []
    layers_parameters = []
    for layer_idx in range(len(layers_feature_sizes)):
        first_layer = layer_idx == 0
        if first_layer:
            first_layer_input_feature = (input_data_feature + input_data_feature % capsule_tall) // capsule_tall + 1
            first_layer_output_feature = layers_feature_sizes[layer_idx+1]
            layer, w, b = ff_layer(first_layer_input_feature, first_layer_output_feature, activation_function, device)
        else:
            last_layer = layer_idx == len(layers_feature_sizes) - 1
            if last_layer:
                last_layer_input_feature = layers_feature_sizes[layer_idx] + 1
                last_layer_output_feature = (input_data_feature + input_data_feature % capsule_tall) // capsule_tall 
                layer, w, b = ff_layer(last_layer_input_feature, last_layer_output_feature, activation_function, device)
            else:
                input_feature = layers_feature_sizes[layer_idx] + 1
                output_feature = layers_feature_sizes[layer_idx+1]
                layer, w, b = ff_layer(input_feature, output_feature, activation_function, device)

        layers.append(layer)
        layers_parameters.extend([[w, b]])

    return layers, layers_parameters

def encapsulate_input_feature(x: torch.Tensor, capsule_tall):
    assert x.shape[-1] % capsule_tall == 0, f'{capsule_tall} should divisible by {x.shape[-1]}'
    input_view = x.shape[-1] // capsule_tall
    new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
    return x.view(new_input_shape)

def rotate_feature(capsule_output: torch.Tensor, rotation_amount: int, layer_idx: int):
    if layer_idx % 2 == 0:
        return capsule_output.flatten(1).roll(rotation_amount)
    else:
        return capsule_output.flatten(1).roll(-rotation_amount)

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
