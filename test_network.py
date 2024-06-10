import torch

def predicting(batched_images_with_combine_labels, network, capsule_wide, capsule_tall):
    batched_goodness_per_label = []
    for combined_label_in_image in batched_images_with_combine_labels:
        input_for_layer = combined_label_in_image
        layers_goodness = []
        for capsule_wide_index in range(capsule_wide):
            for layer in network:
                layer_output = layer(input_for_layer)
                layer_goodness = layer_output.pow(2).mean(1)
                input_for_layer = layer_output
                layers_goodness.append(layer_goodness)
        batched_layer_goodness = sum(layers_goodness)
        each_item_in_batch_per_label_goodness = batched_layer_goodness.view(batched_layer_goodness.shape[0], 1)
        batched_goodness_per_label.append(each_item_in_batch_per_label_goodness)
    batched_prediction_scores = torch.cat(batched_goodness_per_label, dim=1)
    return batched_prediction_scores

def validation_forward_pass(batched_image, batched_label, network):
    print('validating...')
    batched_model_prediction = predicting(batched_image, network)

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