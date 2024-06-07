import torch
import statistics
from test_network import validation_forward_pass
from layer import ff_layer
from network_utils import initialize_layers_and_parameters, rotate_feature, input_feature_view

def capsule_neural_network(feature_sizes: list, input_feature: int, threshold: int, activation_function, lr: float, device: str, capsule_tall: int, capsule_wide: int, rotation_amount: int):
    layers, layers_parameters = initialize_layers_and_parameters(feature_sizes, activation_function, device)

    # first_layer_feature_size = (input_feature + input_feature % capsule_tall) // capsule_tall + feature_sizes[-1] + 1
    first_layer_feature_size = (input_feature + input_feature % capsule_tall) // capsule_tall + 1
    first_layer, w, b = ff_layer(first_layer_feature_size, feature_sizes[0], activation_function, device)
    layers.insert(0, first_layer)
    layers_parameters.insert(0, [w, b])

    def capsule_layer(input_data_features: torch.Tensor, layer, layer_idx: int, capsule_wide_idx: int):
        capsule_wide_idx_as_tensor = torch.full([input_data_features.shape[0], capsule_tall, 1], capsule_wide_idx, device="cuda")
        input_for_capsule = torch.concat([input_data_features, capsule_wide_idx_as_tensor], dim=-1)

        capsule_outputs = []
        for vertical_capsule_index in range(capsule_tall):
            input_view_features = input_for_capsule[:, vertical_capsule_index, :]
            capsule_output = layer(input_view_features)
            capsule_outputs.append(capsule_output)

        return rotate_feature(capsule_outputs, rotation_amount, layer_idx)
    
    def layer_forward_pass(dataloader, layer, layer_idx, capsule_wide_idx, optimizer):
        bad_epoch = 0
        current_epoch = 0
        best_loss = None
        previous_loss = None
        while True:
            lossess_for_each_batch = []
            for positive_data, negative_data in dataloader:
                positive_feature_viewed = input_feature_view(positive_data, capsule_tall)
                negative_feature_viewed = input_feature_view(negative_data, capsule_tall)
                positive_output_features = capsule_layer(positive_feature_viewed, layer, layer_idx, capsule_wide_idx)
                negative_output_features = capsule_layer(negative_feature_viewed, layer, layer_idx, capsule_wide_idx)
                positive_phase_activation = positive_output_features.pow(2).mean(dim=1)
                negative_phase_activation = negative_output_features.pow(2).mean(dim=1)
                layer_loss = torch.log(1 + torch.exp(torch.cat([
                    -positive_phase_activation+threshold,
                    negative_phase_activation-threshold
                ]))).mean()
                optimizer.zero_grad()
                layer_loss.backward()
                optimizer.step()
                lossess_for_each_batch.append(layer_loss.item())
            
            average_loss_for_whole_batch = statistics.fmean(lossess_for_each_batch)
            if best_loss is None:
                best_loss = average_loss_for_whole_batch
            elif average_loss_for_whole_batch < best_loss:
                best_loss = average_loss_for_whole_batch
                if previous_loss is not None:
                    if abs((previous_loss / average_loss_for_whole_batch) - 1) < 0.001:
                        bad_epoch += 1
                    else:
                        bad_epoch = 0
            else:
                bad_epoch += 1

            if bad_epoch > 5:
                print()
                print(f"Done training layer: {layer_idx+1}")
                break
            previous_loss = average_loss_for_whole_batch
            current_epoch += 1

                # Forward pass once for next layer
        def forward_for_next_layer(dataloader, layer):
            previous_output = []
            for positive_data, negative_data in dataloader:
                positive_output_features = layer(positive_data).detach()
                negative_output_features = layer(negative_data).detach()
                previous_output.append((positive_output_features, negative_output_features))
            return previous_output

        return forward_for_next_layer(dataloader, layer)

    def training_forward_pass(train_dataloader: torch.Tensor):
        for capsule_wide_idx in range(capsule_wide):
            for layer_idx, layer in enumerate(layers):
                optimizer = torch.optim.Adam(layers_parameters[layer_idx], lr)
                train_dataloader = layer_forward_pass(train_dataloader, layer, layer_idx, capsule_wide_idx, optimizer)

    def runner(train_loader, test_image, test_label):
        training_forward_pass(train_loader)
        validation_forward_pass(test_image, test_label, layers)

    return runner

# x = torch.randn(1, 12, device="cuda")
# y = torch.randn(1, 12, device="cuda")
# data = [(x, y)]
# m = capsule_neural_network(feature_sizes=[20, 20, 20], input_feature=12, threshold=2.0, activation_function=torch.nn.functional.relu, device="cuda", capsule_tall=2, capsule_wide=1, rotation_amount=1)
# print(m(data))