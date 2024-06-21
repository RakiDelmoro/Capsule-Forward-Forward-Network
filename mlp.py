import torch
import statistics
from layer import linear_layer
from network_utils import print_correct_prediction, print_percentile_of_correct_probabilities, print_wrong_prediction, print_one_hot_outputs

def multi_layer_perceptron(feature_sizes: list):
    layers = []
    parameters = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i]
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature, "cuda", 'relu')
        layers.append(layer)
        parameters.extend([w, b])

    output_layer, output_w, output_b = linear_layer(in_features=feature_sizes[-1], out_features=10, device="cuda", activation_function='softmax')
    layers.append(output_layer)
    parameters.extend([output_w, output_b])

    def forward(input_batch):
        previous_layer_output = input_batch
        for layer in layers:
            previous_layer_output = layer(previous_layer_output)

        return previous_layer_output
    
    def training_for_batch(dataloader, loss_function, optimizer):
        losses_for_each_batch = []
        for batch_image, batch_expected in dataloader:
            output_batch = forward(batch_image)
            loss = loss_function(output_batch, batch_expected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_each_batch.append(loss.item())

        return statistics.fmean(losses_for_each_batch)
    
    def validation_for_batch(dataloader):
        predictions_probabilities = []
        correct_predictions = []
        wrong_predictions = []
        model_predictions = []
        for image, label in dataloader:
            model_output = forward(image)
            digit_predicted = model_output.argmax()
            model_output_probability = model_output.max().item()

            correct_or_wrong = digit_predicted.eq(label).int().item()
            model_predictions.append(correct_or_wrong)
            predictions_probabilities.append(model_output_probability)
            if digit_predicted.item() == label.item():
                predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': label.item()}
                correct_predictions.append(predicted_and_expected)
            else:
                predicted_and_expected = {'predicted': digit_predicted.item(), 'expected': label.item()}
                wrong_predictions.append(predicted_and_expected)

        print_correct_prediction(correct_predictions, 5)
        print_wrong_prediction(wrong_predictions, 5)
        print_percentile_of_correct_probabilities(predictions_probabilities)

        correct_prediction_count = model_predictions.count(1)
        wrong_prediction_count = model_predictions.count(0)
        correct_percentage = (correct_prediction_count / len(model_predictions)) * 100
        wrong_percentage = (wrong_prediction_count / len(model_predictions)) * 100
        print(f"Correct percentage: {round(correct_percentage, 1)} Wrong percentage: {round(wrong_percentage, 1)}")
        print_one_hot_outputs(model_output)

    def runner(training_loader, validation_loader, epochs):
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=parameters, lr=0.001)
        for _ in range(epochs):
            average_loss_for_whole_batch = training_for_batch(training_loader, loss_function, optimizer)
            print(average_loss_for_whole_batch)

            validation_for_batch(validation_loader)

    return runner
