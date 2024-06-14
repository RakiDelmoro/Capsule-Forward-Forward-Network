import torch
from network_utils import capsulate_input_feature, rotate_feature, print_wrong_prediction, print_percentile_of_correct_probabilities, print_correct_prediction

def predicting(batched_images_with_combine_labels, network, capsule_wide, capsule_tall):
    batch_goodness = []
    for all_image_per_label in batched_images_with_combine_labels:
        batch = all_image_per_label
        capsule_goodness_per_label = []
        for capsule_index in range(capsule_wide):
            layers_goodness = []
            for i, layer in enumerate(network):
                capsulate_data = capsulate_input_feature(batch, capsule_tall)
                capsule_idx_as_tensor = torch.full([capsulate_data.shape[0], capsule_tall, 1], capsule_index, device="cuda")
                layer_output = layer(torch.concat([capsulate_data, capsule_idx_as_tensor], dim=-1))
                capsule_tall_activation = layer_output.pow(2).mean(1).detach()
                goodness_each_item_in_batch = capsule_tall_activation.mean(-1)
                batch = rotate_feature(layer_output, 1, i)
                layers_goodness.append(goodness_each_item_in_batch)
            capsule_goodness_per_label.append(sum(layers_goodness))
        batch_goodness.append(sum(layers_goodness).view(batch.shape[0], 1))
    prediction_score = torch.concat(batch_goodness, dim=1)
    
    return prediction_score

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
    print_wrong_prediction(wrong_predictions, 5)
    print_percentile_of_correct_probabilities(predictions_probabilities)

    correct_prediction_count = model_predictions.count(1)
    wrong_prediction_count = model_predictions.count(0)
    correct_percentage = (correct_prediction_count / len(model_predictions)) * 100
    wrong_percentage = (wrong_prediction_count / len(model_predictions)) * 100
    print(f"Correct percentage: {round(correct_percentage, 1)} Wrong percentage: {round(wrong_percentage, 1)}")
