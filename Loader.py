import sys
import getpass
from pathlib import Path

try:
    import torch
except ImportError:
    sys.path.insert(0, f'/goinfre/{getpass.getuser()}/torch_env')
    import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Dataset
VAL_SPLIT  = 0.20
BATCH_SIZE = 32
SEED       = 42

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])



class _TransformSubset(Dataset):
    """Wraps a Subset and applies an independent transform."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_datasets(root: Path):
    """Split root (ImageFolder layout) into 80/20 train/val DataLoaders."""
    # Load without any transform so each split can apply its own
    full_dataset = datasets.ImageFolder(root=str(root), transform=None)
    class_names  = full_dataset.classes
    print(f"\nClasses found ({len(class_names)}): {class_names}")

    n_val   = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    print(f"Split: {n_train} train / {n_val} val")

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_set = _TransformSubset(train_subset, train_transform)
    val_set   = _TransformSubset(val_subset,   val_transform)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, class_names
