import torch
from network_utils import capsulate_input_feature, rotate_feature, print_wrong_prediction, print_percentile_of_correct_probabilities, print_correct_prediction

def predicting(batched_images_with_combine_labels, network, capsule_wide, capsule_tall):
    dataset = batched_images_with_combine_labels
    batched_goodness_per_label = []
    for c_i in range(capsule_wide):
        layers_goodness = []
        for i, layer in enumerate(network):
            previous_layer_outputs = []
            layer_goodness_per_label = [] 
            for all_image_per_label in dataset:
                capsulate_data = capsulate_input_feature(all_image_per_label, capsule_tall)
                layer_output = layer(capsulate_data)
                activation_each_capsule_tall = layer_output.pow(2).mean(-1)
                goodness_per_label = activation_each_capsule_tall.mean(-1)
                # TODO: average goodness per label and store to layer goodness per label
                previous_layer_outputs.append(layer_output)
            rotated_layer_outputs = []
            for per_label_output in previous_layer_outputs:
                input_for_next_layer = rotate_feature(capsule_output=per_label_output, rotation_amount=1, layer_idx=i)
                rotated_layer_outputs.append(input_for_next_layer)            
            dataset = rotated_layer_outputs
            # Layer goodness should have (number of layers)x item and each of the item has goodness per label
            # Sum layer goodness per label for all the layers then divide by number of layers do for all labels and that's the goodness of each capsule wide

    return batched_goodness_per_label

def validation_forward_pass(batched_image, batched_label, network, capsule_tall, capsule_wide):
    print('validating...')
    batched_model_prediction = predicting(batched_image, network, capsule_wide, capsule_tall)

    predictions_probabilities = []
    correct_predictions = []
    wrong_predictions = []
    model_predictions = []
    for each_item in range(batched_model_prediction.shape[0]):
        model_prediction = batched_model_prediction[each_item]
        expected = batched_label[each_item]
        digit_predicted = model_prediction.argmax()
        digit_probability = torch.nn.functional.softmax(model_prediction, dim=0).max().item()
        
        correct_or_wrong = digit_predicted.eq(expected).int().item()
        model_predictions.append(correct_or_wrong)

        predictions_probabilities.append(digit_probability)
        if digit_predicted.item() == expected.item():
            predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': expected.item()}
            correct_predictions.append(predicted_and_expected)
        else:
            predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': expected.item()}
            wrong_predictions.append(predicted_and_expected)

    print_correct_prediction(correct_predictions, 5)
    print_percentile_of_correct_probabilities(predictions_probabilities)
    print_wrong_prediction(wrong_predictions, 5)