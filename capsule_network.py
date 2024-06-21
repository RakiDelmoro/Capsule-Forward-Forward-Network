import torch
import statistics
from test_network import validation_forward_pass
from network_utils import initialize_capsule_layers_and_parameters, rotate_feature, encapsulate_input_feature

def capsule_neural_network(capsule_feature_size: list, input_feature: int, threshold: int, activation_function, lr: float, device: str, capsule_tall: int, capsule_wide: int, rotation_amount: int):
    layers, layers_parameters = initialize_capsule_layers_and_parameters(capsule_feature_size, input_feature, capsule_tall, activation_function, device)

    def preprocess_dataloader(dataloader, capsule_index):
        processed_dataloader = []
        for positive_data, negative_data in dataloader:
            encapsulated_positive_data = encapsulate_input_feature(positive_data, capsule_tall)
            encapsulated_negative_data = encapsulate_input_feature(negative_data, capsule_tall)
            capsule_idx_as_tensor = torch.full([encapsulated_positive_data.shape[0], capsule_tall, 1], capsule_index, device="cuda")
            new_positive_data = torch.concat([encapsulated_positive_data, capsule_idx_as_tensor], dim=-1)
            new_negative_data = torch.concat([encapsulated_negative_data, capsule_idx_as_tensor], dim=-1)
            processed_dataloader.append((new_positive_data, new_negative_data))

        return processed_dataloader

    def forward_dataloader_in_layer(dataloader, layer, optimizer):
        loss_each_batch = []
        for positive_data, negative_data in dataloader:
            each_capsule_tall_output_for_positive_phase = layer(positive_data)
            each_capsule_tall_output_for_negative_phase = layer(negative_data)
            squared_activation_in_each_capsule_tall_for_positive_data = each_capsule_tall_output_for_positive_phase.pow(2).mean(1)
            squared_activation_in_each_capsule_tall_for_negative_data = each_capsule_tall_output_for_negative_phase.pow(2).mean(1)
            capsule_column_layer_goodness_for_positive_phase = squared_activation_in_each_capsule_tall_for_positive_data.mean(-1)
            capsule_column_layer_goodness_for_negative_phase = squared_activation_in_each_capsule_tall_for_negative_data.mean(-1)
            capsule_column_layer_loss = torch.log(1 + torch.exp(torch.cat([
                -capsule_column_layer_goodness_for_positive_phase+threshold,
                capsule_column_layer_goodness_for_negative_phase-threshold
            ]))).mean()
            optimizer.zero_grad()
            capsule_column_layer_loss.backward()
            optimizer.step()
            loss_each_batch.append(capsule_column_layer_loss.item())
        average_loss_for_whole_batch = statistics.fmean(loss_each_batch)
        print(f'\r {round(average_loss_for_whole_batch, 5)}', end='', flush=True)
        return average_loss_for_whole_batch

    def forward_layer_in_each_capsule(previous_layer_output, dataloader, layer, optimizer, layer_idx, capsule_wide_index):
        bad_epoch = 0
        current_epoch = 0
        best_loss = None
        previous_layer_loss = None
        capsule_layer_outputs = []
        for idx in range(capsule_wide_index):
            first_capsule_first_layer = idx == 0 and layer_idx == 0
            if first_capsule_first_layer:
                input_for_layer = preprocess_dataloader(dataloader, idx)
            else:
                input_for_layer = preprocess_dataloader(previous_layer_output, idx)
            while True:
                capsule_column_loss = forward_dataloader_in_layer(input_for_layer, layer, optimizer)
                if best_loss is None:
                    best_loss = capsule_column_loss
                elif capsule_column_loss < best_loss:
                    best_loss = capsule_column_loss
                    if previous_layer_loss is not None:
                        if abs((previous_layer_loss / capsule_column_loss) - 1) < 0.001:
                            bad_epoch += 1
                        else:
                            bad_epoch = 0
                else:
                    bad_epoch += 1
                if bad_epoch > 5:
                    print(f"Done training layer: {layer_idx+1} EPOCH: {current_epoch}")
                    break
                previous_layer_loss = capsule_column_loss
                current_epoch += 1
            capsule_layer_outputs.append(forward_once_for_next_layer_in_capsule(input_for_layer, layer, layer_idx))
        return sum(capsule_layer_outputs)

    def forward_once_for_next_layer_in_capsule(dataloader, layer, layer_idx):
        previous_capsule_column_output = []
        for positive_data, negative_data in dataloader:
            positive_column_capsule_outputs = layer(positive_data).detach()
            negative_column_capsule_outputs = layer(negative_data).detach()
            positive_features = rotate_feature(positive_column_capsule_outputs, rotation_amount, layer_idx)
            negative_features = rotate_feature(negative_column_capsule_outputs, rotation_amount, layer_idx)
            previous_capsule_column_output.append((positive_features, negative_features))

        return previous_capsule_column_output

    def train_capsule_column(previous_capsule_output, dataloader, capsule_wide_index):
        layer_output = previous_capsule_output
        for layer_index, layer in enumerate(layers):
            optimizer = torch.optim.AdamW(layers_parameters[layer_index], lr=lr)
            layer_output = forward_layer_in_each_capsule(layer_output, dataloader, layer, optimizer, layer_index, capsule_wide_index)

        return layer_output

    def capsule_forward_pass(dataloader, test_image, test_label):
        previous_capsule_output = dataloader
        for capsule_wide_index in range(1, capsule_wide+1):
            previous_capsule_output = train_capsule_column(previous_capsule_output, dataloader, capsule_wide_index)

        validation_forward_pass(test_image, test_label, layers, capsule_tall, capsule_wide)

    return capsule_forward_pass
