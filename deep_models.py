"""Model architectures and image preprocessing for the Galaxy10 experiments."""

from dataclasses import dataclass

import timm
from torch import Tensor, nn


NUM_CLASSES = 10
_PRETRAINED_WEIGHTS = {
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}
MODEL_NAMES = ("scratch_cnn", *_PRETRAINED_WEIGHTS)


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    weights_id: str | None
    interpolation: str
    crop_pct: float


class ScratchCNN(nn.Module):
    """Five convolutional blocks followed by global pooling and a classifier."""

    def __init__(self) -> None:
        super().__init__()
        channels = (3, 32, 64, 96, 128, 192)
        self.features = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                for in_channels, out_channels in zip(channels, channels[1:])
            )
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], NUM_CLASSES),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))


def build_model(
    name: str,
    *,
    pretrained: bool,
    scratch_mean: tuple[float, float, float],
    scratch_std: tuple[float, float, float],
) -> ModelBundle:
    """Build a ten-class model and look up preprocessing for its weight variant."""
    if name == "scratch_cnn":
        return ModelBundle(ScratchCNN(), scratch_mean, scratch_std, None, "bilinear", 1.0)

    if name not in _PRETRAINED_WEIGHTS:
        raise ValueError(f"Unknown Galaxy10 model {name!r}. Expected one of {MODEL_NAMES}.")

    weights_id = _PRETRAINED_WEIGHTS[name]
    model = timm.create_model(weights_id, pretrained=pretrained, num_classes=NUM_CLASSES)
    data_config = timm.data.resolve_model_data_config(model)
    return ModelBundle(
        model=model,
        mean=data_config["mean"],
        std=data_config["std"],
        weights_id=weights_id if pretrained else None,
        interpolation=data_config["interpolation"],
        crop_pct=data_config["crop_pct"],
    )
