"""
Dataset and DataLoader utilities for ImageNet Validation, CIFAR-100 (debug), and Synthetic datasets.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import torchvision
import torchvision.transforms as T


@dataclass
class DatasetConfig:
    name: str = "imagenet_val"  # "imagenet_val", "cifar100", "synthetic"
    data_dir: Optional[str] = "./data"
    split: str = "val"
    num_samples: Optional[int] = 250
    batch_size: int = 25
    num_workers: int = 0
    img_size: int = 224
    seed: int = 42
    mode: str = "dev"  # "debug", "dev", "full"


class SyntheticDataset(Dataset):
    """Deterministic synthetic dataset for offline unit tests and zero-network verification."""
    def __init__(self, num_samples: int = 100, img_size: int = 224, num_classes: int = 1000, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Deterministic generation
        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num_samples, 3, img_size, img_size, generator=generator)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], int(self.labels[idx])


def get_transforms(img_size: int = 224) -> T.Compose:
    """Standard ImageNet evaluation transforms."""
    scale_size = int(256 / 224 * img_size)
    return T.Compose([
        T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_cached_or_download_imagenet_val(data_dir: str = "./data", num_samples: int = 250) -> Dataset:
    """
    Loads real ImageNet-1k validation samples from local cache or downloads directly.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"imagenet_val_{num_samples}.pt")
    generic_cache = os.path.join(data_dir, "imagenet_val_250.pt")

    # 1. Check direct cached tensor files
    if os.path.isfile(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return TensorDataset(data["images"], data["labels"])
    elif os.path.isfile(generic_cache):
        data = torch.load(generic_cache, map_location="cpu", weights_only=True)
        images = data["images"][:num_samples]
        labels = data["labels"][:num_samples]
        return TensorDataset(images, labels)

    # 2. Check local ImageFolder structure
    val_dir = os.path.join(data_dir, "imagenet", "val")
    if os.path.isdir(val_dir):
        return torchvision.datasets.ImageFolder(root=val_dir, transform=get_transforms(224))
    
    # 3. Stream from Hugging Face dataset if no local cache exists
    print(f"[DataLoader] Downloading {num_samples} ImageNet validation samples from repository...")
    from datasets import load_dataset
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val", streaming=True)
    transforms = get_transforms(224)

    images, labels = [], []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(transforms(img))
        labels.append(item["label"])

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save to cache
    torch.save({"images": images_tensor, "labels": labels_tensor}, cache_path)
    return TensorDataset(images_tensor, labels_tensor)


def get_dataloader(config: DatasetConfig) -> Tuple[DataLoader, int]:
    """
    Creates a DataLoader according to the DatasetConfig.
    Returns (DataLoader, total_selected_samples).
    """
    transforms = get_transforms(config.img_size)

    if config.name.lower() == "synthetic":
        num_samples = config.num_samples if config.num_samples is not None else 100
        dataset = SyntheticDataset(
            num_samples=num_samples,
            img_size=config.img_size,
            seed=config.seed,
        )
    elif config.name.lower() == "cifar100":
        os.makedirs(config.data_dir or "./data", exist_ok=True)
        is_train = (config.split.lower() == "train")
        dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir or "./data",
            train=is_train,
            download=True,
            transform=transforms,
        )
    elif config.name.lower() in ("imagenet_val", "imagenet"):
        num_samples = config.num_samples if config.num_samples is not None else 250
        dataset = load_cached_or_download_imagenet_val(
            data_dir=config.data_dir or "./data",
            num_samples=num_samples,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {config.name}")

    total_len = len(dataset)
    num_samples = config.num_samples

    # Determine subset indices if num_samples is restricted
    if num_samples is not None and num_samples < total_len:
        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(total_len, generator=generator)[:num_samples].tolist()
        dataset = Subset(dataset, indices)
        final_count = len(dataset)
    else:
        final_count = total_len

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(torch.cuda.is_available()),
    )
    return loader, final_count
