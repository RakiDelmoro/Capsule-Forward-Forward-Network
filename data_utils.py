import gzip
import torch
import random
import pickle

def combine_label_in_image(image, label, num_classes=10):
    clone_image = image.copy()
    clone_image[:num_classes] = 0.0
    clone_image[label] = image.max()
    return clone_image

def get_wrong_label(label, num_classes=10):
    indices_to_use = list(range(num_classes))
    indices_to_use.remove(label)
    return random.choice(indices_to_use)

def get_training_data(all_images, all_labels, num_classes=10):
    training_data = []
    for index in range(all_images.shape[0]):
        image = all_images[index]
        label = all_labels[index]
        wrong_label = get_wrong_label(label, num_classes)
        positive_data = combine_label_in_image(image, label, num_classes)
        negative_data = combine_label_in_image(image, wrong_label)
        training_data.append((torch.tensor(positive_data, device="cuda"), torch.tensor(negative_data, device="cuda")))

    return training_data

def get_validation_data(all_image, all_label, num_classes=10):
    combine_image_and_labels = []
    for label in range(num_classes):
        label_combine_in_images = []
        for each in range(all_image.shape[0]):
            image = all_image[each]
            imabe_and_label_combine = combine_label_in_image(image, label)
            label_combine_in_images.append((torch.tensor(imabe_and_label_combine, device="cuda")))

        stacked_image_based_on_label = torch.stack(label_combine_in_images)
        combine_image_and_labels.append(stacked_image_based_on_label)

    labels_as_tensor = torch.tensor(all_label, device="cuda")
    return combine_image_and_labels, labels_as_tensor

def load_data_to_memory(filename: str):
    with (gzip.open(filename, 'rb')) as file:
        ((training_image_array, training_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding="latin-1")

    return training_image_array, training_label_array, validation_image_array, validation_label_array
