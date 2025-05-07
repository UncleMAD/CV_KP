import random

import torch
from torch.utils.data import Subset
from torch.utils.data import random_split
from torchvision import datasets


def load_data(path, size, transform):
    full_dataset = datasets.ImageFolder(path, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # остаток

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    def reduce_dataset(dataset, fraction=0.2, seed=42):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), int(len(dataset) * fraction))
        return Subset(dataset, indices)

    train_dataset = reduce_dataset(train_dataset, fraction=size)
    val_dataset = reduce_dataset(val_dataset, fraction=size)
    test_dataset = reduce_dataset(test_dataset, fraction=size)
    classes = full_dataset.class_to_idx

    return train_dataset, val_dataset, test_dataset, classes
