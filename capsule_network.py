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

    def capsule_layer(positive_data, negative_data, layer, layer_idx: int, capsule_wide_idx: int):
        capsule_wide_idx_as_tensor = torch.full([positive_data.shape[0], capsule_tall, 1], capsule_wide_idx, device="cuda")
        positive_phase = torch.concat([positive_data, capsule_wide_idx_as_tensor], dim=-1)
        negative_phase = torch.concat([negative_data, capsule_wide_idx_as_tensor], dim=-1)

        capsule_positive_outputs = []
        capsule_negative_outputs = []
        for vertical_capsule_index in range(capsule_tall):
            positive_input_view_features = positive_phase[:, vertical_capsule_index, :]
            negative_input_view_features = negative_phase[:, vertical_capsule_index, :]
            positive_phase_output = layer(positive_input_view_features)
            negative_phase_output = layer(negative_input_view_features)
            capsule_positive_outputs.append(positive_phase_output)
            capsule_negative_outputs.append(negative_phase_output)

        positive_feature_rotated = rotate_feature(capsule_positive_outputs, rotation_amount, layer_idx)
        negative_feature_rotated = rotate_feature(capsule_negative_outputs, rotation_amount, layer_idx)

        return positive_feature_rotated, negative_feature_rotated
    
    def forward_for_next_layer(dataloader, layer, capsule_wide_index):
        previous_output = []
        for positive_data, negative_data in dataloader:
            capsule_wide_idx_as_tensor = torch.full([positive_data.shape[0], capsule_tall, 1], capsule_wide_index, device="cuda")
            positive_phase = input_feature_view(positive_data, capsule_tall)
            negative_phase = input_feature_view(negative_data, capsule_tall)
            
            capsule_positive_outputs = []
            capsule_negative_outputs = []
            for vertical_capsule_index in range(capsule_tall):
                positive_phase_for_layer = torch.concat([positive_phase, capsule_wide_idx_as_tensor], dim=-1)
                negative_phase_for_layer = torch.concat([negative_phase, capsule_wide_idx_as_tensor], dim=-1)
                positive_viewed_features = positive_phase_for_layer[:, vertical_capsule_index, :]
                negative_viewed_features = negative_phase_for_layer[:, vertical_capsule_index, :]
                positive_output_features = layer(positive_viewed_features).detach()
                negative_output_features = layer(negative_viewed_features).detach()
                capsule_positive_outputs.append(positive_output_features)
                capsule_negative_outputs.append(negative_output_features)
            
            positive_phase_output = torch.concat(capsule_positive_outputs, dim=-1).detach()
            negative_phase_output = torch.concat(capsule_negative_outputs, dim=-1).detach()
            previous_output.append((positive_phase_output, negative_phase_output))
        
        return previous_output
    
    def layer_forward_pass(dataloader, layer, layer_idx, capsule_wide_idx, optimizer):
        print("Training layer...")
        bad_epoch = 0
        epoch = 0
        best_loss = None
        previous_loss = None
        while True:
            lossess_for_each_batch = []
            for positive_data, negative_data in dataloader:
                positive_feature_viewed = input_feature_view(positive_data, capsule_tall)
                negative_feature_viewed = input_feature_view(negative_data, capsule_tall)
                positive_features, negative_features = capsule_layer(positive_feature_viewed, negative_feature_viewed, layer, layer_idx, capsule_wide_idx)
                positive_phase_activation = positive_features.pow(2).mean(dim=1)
                negative_phase_activation = negative_features.pow(2).mean(dim=1)
                layer_loss = torch.log(1 + torch.exp(torch.cat([
                    -positive_phase_activation+threshold,
                    negative_phase_activation-threshold
                ]))).mean()
                optimizer.zero_grad()
                layer_loss.backward()
                optimizer.step()
                lossess_for_each_batch.append(layer_loss.item())  
            average_loss_for_whole_batch = statistics.fmean(lossess_for_each_batch)
            print(f'\r Epoch: {epoch} Layer: {layer_idx+1} average loss for each batch: {average_loss_for_whole_batch}', end='', flush=True)

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
                return forward_for_next_layer(dataloader, layer, capsule_wide_idx)
            previous_loss = average_loss_for_whole_batch
            epoch += 1

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