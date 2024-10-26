from functools import partial
from typing import Any, Optional

from torch import nn

from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel


__all__ = ["fcn_resnet50"]


class FCN(_SimpleSegmentationModel):
    pass


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
    "_docs": """
        These weights were trained on a subset of COCO, using only the 20 categories that are present in the Pascal VOC
        dataset.
    """,
}


class FCN_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 35322218,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet50",
            "_metrics": {
                "COCO-val2017-VOC-labels": {
                    "miou": 60.5,
                    "pixel_acc": 91.4,
                }
            },
            "_ops": 152.717,
            "_file_size": 135.009,
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _fcn_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)


@register_model()
@handle_legacy_interface(
    weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def fcn_resnet50(
    *,
    weights: Optional[FCN_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> FCN:

    weights = FCN_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
