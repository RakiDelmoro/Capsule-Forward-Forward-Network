import torch
from network_utils import capsulate_input_feature, rotate_feature, print_wrong_prediction, print_percentile_of_correct_probabilities, print_correct_prediction

def predicting(batched_images_with_combine_labels, network, capsule_wide, capsule_tall):
    batched_goodness_per_label = []
    #for _ in range(capsule_wide):
        #for layer in network:
            # function to get dataset for layer
            # Loop each dataset in batch_images_with labels to the whole layers to get the goodness
    pass


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