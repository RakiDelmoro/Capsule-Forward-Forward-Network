import torch
import statistics
from test_network import validation_forward_pass
from layer import ff_layer
from network_utils import initialize_layers_and_parameters, rotate_feature, capsulate_input_feature

def capsule_neural_network(capsule_feature_size: list, input_feature: int, threshold: int, activation_function, lr: float, device: str, capsule_tall: int, capsule_wide: int, rotation_amount: int):
    layers, layers_parameters = initialize_layers_and_parameters(capsule_feature_size, activation_function, device)

    first_layer_feature_size = (input_feature + input_feature % capsule_tall) // capsule_tall # + 1 
    first_layer, w, b = ff_layer(first_layer_feature_size, capsule_feature_size[0], activation_function, device)
    layers.insert(0, first_layer)
    layers_parameters.insert(0, [w, b])

    def train_capsule_column(dataloader):
        for layer_param, layer in enumerate(layers):
            optimizer = torch.optim.AdamW(layers_parameters[layer_param], lr=lr)
            bad_epoch = 0
            current_epoch = 0
            best_loss = None
            previous_loss = None
            while True:
                positive_features, negative_features = capsule_column_forward_pass(dataloader, layer)
                capsule_column_activation_for_positive_phase = positive_features.pow(2).mean(1)
                capsule_column_activation_for_negative_phase = negative_features.pow(2).mean(1)
                capsule_column_loss = torch.log(1 + torch.exp(torch.cat([
                    -capsule_column_activation_for_positive_phase+threshold,
                    capsule_column_activation_for_negative_phase-threshold
                ]))).mean()
                optimizer.zero_grad()
                capsule_column_loss.backward()
                optimizer.step()
                print(f'\r EPOCH: {current_epoch} loss: {capsule_column_loss}')

                if best_loss is None:
                    best_loss = capsule_column_loss
                elif capsule_column_loss < best_loss:
                    best_loss = capsule_column_loss
                    if previous_loss is not None:
                        if abs((previous_loss / capsule_column_loss) - 1) < 0.001:
                            bad_epoch += 1
                        else:
                            bad_epoch = 0
                else:
                    bad_epoch += 1

                if bad_epoch > 5:
                    print(f"Done training layer: {layer_param+1} of all column capsule")
                    dataloader = forward_once_for_next_layer(dataloader, layer)
                    break

                previous_loss = capsule_column_loss
                current_epoch += 1

    def capsule_column_forward_pass(dataloader, layer):
        positive_column_capsule_outputs = []
        negative_column_capsule_outputs = []
        for positive_data, negative_data in dataloader:
            capsulated_positive_data = capsulate_input_feature(positive_data, capsule_tall)
            capsulated_negative_data = capsulate_input_feature(negative_data, capsule_tall)
            for each in range(capsule_tall):
                positive_capsule_column = capsulated_positive_data[:, each, :]
                negative_capsule_column = capsulated_negative_data[:, each, :]
                positive_output_feature = layer(positive_capsule_column)
                negative_output_feature = layer(negative_capsule_column)
                positive_column_capsule_outputs.append(positive_output_feature)
                negative_column_capsule_outputs.append(negative_output_feature)

        return torch.concat(positive_column_capsule_outputs, dim=1), torch.concat(negative_column_capsule_outputs, dim=1)

    def forward_once_for_next_layer(dataloader, layer, layer_idx):
        previous_capsule_wide_output = []
        for positive_data, negative_data in dataloader:
            capsulated_positive_data = capsulate_input_feature(positive_data, capsule_tall)
            capsulated_negative_data = capsulate_input_feature(negative_data, capsule_tall)
            positive_column_capsule_outputs = []
            negative_column_capsule_outputs = []
            for each in range(capsule_tall):
                positive_capsule_column = capsulated_positive_data[:, each, :]
                negative_capsule_column = capsulated_negative_data[:, each, :]
                positive_output_feature = layer(positive_capsule_column).detach()
                negative_output_feature = layer(negative_capsule_column).detach()
                positive_column_capsule_outputs.append(positive_output_feature)
                negative_column_capsule_outputs.append(negative_output_feature)
            positive_features = rotate_feature(positive_column_capsule_outputs, rotation_amount, layer_idx)
            negative_features = rotate_feature(negative_column_capsule_outputs, rotation_amount, layer_idx)
            previous_capsule_wide_output.append((positive_features, negative_features))

        # TODO: take the input of each capsule tall and concatenate that with previous capsule wide output to input for the next capsule wide

        return previous_capsule_wide_output

    return train_capsule_column

x = torch.randn(1, 12, device="cuda")
y = torch.randn(1, 12, device="cuda")
data = [(x, y)]
m = capsule_neural_network(capsule_feature_size=[20, 20, 20], input_feature=12, threshold=2.0, activation_function=torch.nn.functional.relu, lr=0.01, device="cuda", capsule_tall=2, capsule_wide=1, rotation_amount=1)
print(m(data))