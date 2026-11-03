import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def load_data(batch_size=64, *, augment=False):
    """Load CIFAR-10 as in notebook.ipynb or data_augmentation.ipynb.

    Return train, validation, and test loaders.
    """

    dataset = load_dataset("uoft-cs/cifar10")
    train_val = dataset["train"].train_test_split(test_size=0.2, seed=111)
    raw_train = train_val["train"]

    base_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    pixel_sum = torch.zeros(3, 32, 32)
    count = 0

    for batch in raw_train.iter(batch_size=256):
        imgs = torch.stack([base_transforms(img) for img in batch["img"]])
        pixel_sum += imgs.sum(dim=0)
        count += imgs.shape[0]

    pixel_mean = pixel_sum / count

    eval_transforms = v2.Compose(
        [
            *base_transforms.transforms,
            v2.Lambda(lambda img: img - pixel_mean),
        ]
    )

    if augment:
        train_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomHorizontalFlip(),
                v2.RandomCrop(32, padding=4),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(lambda img: img - pixel_mean),
            ]
        )
    else:
        train_transforms = eval_transforms

    def apply_train_transforms(examples):
        examples["img"] = [train_transforms(img) for img in examples["img"]]
        return examples

    def apply_eval_transforms(examples):
        examples["img"] = [eval_transforms(img) for img in examples["img"]]
        return examples

    train = raw_train.with_transform(apply_train_transforms)
    val = train_val["test"].with_transform(apply_eval_transforms)
    test = dataset["test"].with_transform(apply_eval_transforms)

    return (
        DataLoader(train, batch_size=batch_size),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
