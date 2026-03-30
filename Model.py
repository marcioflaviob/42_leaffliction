import torch.nn as nn


class LeafCNN(nn.Module):
    """
    Small CNN trained from scratch for leaf disease classification.

    Architecture:
        3 convolutional blocks (conv → BN → ReLU → MaxPool)
        followed by a 2-layer classifier head with dropout.

    Input:  (B, 3, 224, 224)  — normalised to [-1, 1]
    Output: (B, num_classes)  — raw logits
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
