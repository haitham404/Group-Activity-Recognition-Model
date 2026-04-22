import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class B1Classifier(nn.Module):
    """
    Baseline 1 model:
    - Uses pretrained ResNet50 as backbone
    - Freezes early layers
    - Replaces final FC layer for custom number of classes
    """

    def __init__(self, num_classes):
        super(B1Classifier, self).__init__()

        # Load pretrained ResNet50 (ImageNet weights)
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # =========================
        # Freeze early feature extractor layers
        # (to keep low-level features like edges, textures)
        # =========================
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False

        # =========================
        # Replace classification head
        # from 1000 ImageNet classes → your dataset classes
        # =========================
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass:
        input → ResNet50 → logits
        """
        return self.backbone(x)