"""
AgriSense AI - Module 3: Leaf Disease Detection & Segmentation
- MobileNetV2 (Transfer Learning for edge deployment)
- UNet (Segmentation & localization of disease lesions)
"""

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LeafClassifierMobileNet(nn.Module):
        """
        MobileNetV2 adapted for PlantVillage leaf disease classification.
        Targeted for ExecuTorch mobile deployment (~3.5MB).
        """
        def __init__(self, num_classes=10, pretrained=True, dropout=0.2):
            super(LeafClassifierMobileNet, self).__init__()
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)

            # Replace classifier head
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2.0),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)


    class LeafDiseaseUNet(nn.Module):
        """
        Lightweight UNet for segmenting diseased regions on plant leaves.
        """
        def __init__(self, in_channels=3, out_channels=1):
            super(LeafDiseaseUNet, self).__init__()

            def double_conv(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )

            self.inc = double_conv(in_channels, 16)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(16, 32))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(32, 64))

            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv_up1 = double_conv(64, 32)

            self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.conv_up2 = double_conv(32, 16)

            self.outc = nn.Conv2d(16, out_channels, 1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)

            x = self.up1(x3)
            x = torch.cat([x, x2], dim=1)
            x = self.conv_up1(x)

            x = self.up2(x)
            x = torch.cat([x, x1], dim=1)
            x = self.conv_up2(x)

            return torch.sigmoid(self.outc(x))
else:
    class LeafClassifierMobileNet:
        pass
    class LeafDiseaseUNet:
        pass

