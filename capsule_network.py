import torch
import statistics
from test_network import validation_forward_pass
from network_utils import initialize_capsule_layers_and_parameters, rotate_feature, capsulate_input_feature

def capsule_neural_network(capsule_feature_size: list, input_feature: int, threshold: int, activation_function, lr: float, device: str, capsule_tall: int, capsule_wide: int, rotation_amount: int):
    layers, layers_parameters = initialize_capsule_layers_and_parameters(capsule_feature_size, input_feature, capsule_tall, activation_function, device)

    def forward_dataloader_in_layer(dataloader, layer, optimizer):
        loss_each_batch = []
        for positive_data, negative_data in dataloader:
            # capsuled_data shape -> [Batch, Capsule tall, Input feature // Capsule tall]
            capsulate_positive_data = capsulate_input_feature(positive_data, capsule_tall)
            capsulate_negative_data = capsulate_input_feature(negative_data, capsule_tall)
            # capsule_outputs shape -> [Batch, Capsule tall, Layer output feature]
            each_capsule_tall_output_for_positive_phase = layer(capsulate_positive_data)
            each_capsule_tall_output_for_negative_phase = layer(capsulate_negative_data)
            # activation_each_capsule shape -> [Batch, Capsule tall]
            activation_in_each_capsule_tall_for_positive_data = each_capsule_tall_output_for_positive_phase.pow(2).mean(-1)
            activation_in_each_capsule_tall_for_negative_data = each_capsule_tall_output_for_negative_phase.pow(2).mean(-1)
            # capsule_goodness shape -> [Batch]
            capsule_column_layer_goodness_for_positive_phase = activation_in_each_capsule_tall_for_positive_data.mean(-1)
            capsule_column_layer_goodness_for_negative_phase = activation_in_each_capsule_tall_for_negative_data.mean(-1)
            
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

    def forward_layer_in_each_capsule(dataloader, layer, optimizer, layer_idx):
        bad_epoch = 0
        current_epoch = 0
        best_loss = None
        previous_layer_loss = None
        while True:
            capsule_column_loss = forward_dataloader_in_layer(dataloader, layer, optimizer)
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
                return forward_once_for_next_layer_in_capsule(dataloader, layer, layer_idx)
            previous_layer_loss = capsule_column_loss
            current_epoch += 1
    
    def forward_once_for_next_layer_in_capsule(dataloader, layer, layer_idx):
        previous_capsule_column_output = []
        for positive_data, negative_data in dataloader:
            capsulated_positive_data = capsulate_input_feature(positive_data, capsule_tall)
            capsulated_negative_data = capsulate_input_feature(negative_data, capsule_tall)
            positive_column_capsule_outputs = layer(capsulated_positive_data).detach()
            negative_column_capsule_outputs = layer(capsulated_negative_data).detach()
            positive_features = rotate_feature(positive_column_capsule_outputs, rotation_amount, layer_idx)
            negative_features = rotate_feature(negative_column_capsule_outputs, rotation_amount, layer_idx)
            previous_capsule_column_output.append((positive_features, negative_features))
    
        return previous_capsule_column_output
    
    def train_capsule_column(dataloader):
        for layer_index, layer in enumerate(layers):
            optimizer = torch.optim.AdamW(layers_parameters[layer_index], lr=lr)
            dataloader = forward_layer_in_each_capsule(dataloader, layer, optimizer, layer_index)

        return dataloader
    
    def capsule_forward_pass(dataloader, test_image, test_label):
        # for _ in range(capsule_wide):
        #     dataloader = train_capsule_column(dataloader)

        validation_forward_pass(test_image, test_label, layers, capsule_tall, capsule_wide)

    return capsule_forward_pass

# x = torch.randn(1, 12, device="cuda")
# y = torch.randn(1, 12, device="cuda")
# data = [(x, y)]
# m = capsule_neural_network(capsule_feature_size=[20, 20, 20], input_feature=12, threshold=2.0, activation_function=torch.nn.functional.relu, lr=0.01, device="cuda", capsule_tall=2, capsule_wide=1, rotation_amount=1)
# print(m(data))