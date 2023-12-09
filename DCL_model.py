import torch
import torchvision
from torch import nn


class DCL(nn.Module):
    def __init__(self, num_classes, cls_2, cls_2xmul):
        super(DCL, self).__init__()
        self.num_classes = num_classes
        self.cls_2 = cls_2
        self.cls_2xmul = cls_2xmul

        backbone = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)  # small:576  large:960
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.Convmask = nn.Conv2d(576, 1, 1, stride=1, padding=0, bias=True)
        self.avgpool2 = nn.AvgPool2d(2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(576, self.num_classes, bias=False)

        if self.cls_2:
            self.classifier_swap = nn.Linear(576, 2, bias=False)
        if self.cls_2xmul:
            self.classifier_swap = nn.Linear(576, 2 * self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)

        mask = self.Convmask(x)
        mask = self.avgpool2(mask)
        mask = torch.tanh(mask)
        mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        out = [
            self.classifier(x),
            self.classifier_swap(x),
            mask
        ]
        return out
