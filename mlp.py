import torch
import statistics
from network_utils import ff_layer
from torch.nn.functional import relu, softmax

def multi_layer_perceptron(feature_sizes: list):
    layers = []
    parameters = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i]
        output_feature = feature_sizes[i+1]
        layer, w, b = ff_layer(input_feature, output_feature, relu, "cuda")
        layers.append(layer)
        parameters.extend([w, b])

    output_layer, output_w, output_b = ff_layer(in_features=feature_sizes[-1], out_features=10, activation_function=softmax, device="cuda", is_for_output=True)
    layers.append(output_layer)
    parameters.extend([output_w, output_b])

    def forward(input_batch):
        previous_layer_output = input_batch
        for layer in layers:
            previous_layer_output = layer(previous_layer_output)
        
        return previous_layer_output
    
    def train_for_batch(dataloader, loss_function, optimizer):
        losses_for_each_batch = []
        for batch_image, batch_expected in dataloader:
            output_batch = forward(batch_image)
            loss = loss_function(output_batch, batch_expected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_for_each_batch.append(loss.item())

        return statistics.fmean(losses_for_each_batch)
    
    def runner(training_loader, epochs):
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=parameters, lr=0.001)
        for _ in range(epochs):
            average_loss_for_whole_batch = train_for_batch(training_loader, loss_function, optimizer)
            print(average_loss_for_whole_batch)

    return runner