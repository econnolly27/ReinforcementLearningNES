# Helper functions for creating datasets and splitting them
# Author: Gerardo Aragon-Camarasa

import torch
from torch.utils.data import DataLoader


def create_datasets(dataset, shuffle_dataset, batch_size):
    # Creating data indices for training and validation splits:
    train_size = len(dataset)
    print("Dataset size: {}".format(train_size))
    out_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=4)
    return out_loader

def create_datasets_split(dataset, shuffle_dataset, validation_split, batch_training, batch_validation):
    # Creating data indices for training and validation splits:
    train_size = int(validation_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, validation_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Train size: {}".format(len(train_ds)))
    print("Validation size: {}".format(len(validation_ds)))

    train_loader = DataLoader(train_ds, batch_size=batch_training, shuffle=shuffle_dataset, num_workers=4)
    validation_loader = DataLoader(validation_ds, batch_size=batch_validation, shuffle=shuffle_dataset, num_workers=4)

    return train_loader, validation_loader