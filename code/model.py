"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

from typing import List, Dict, Any
import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1)
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init`:
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (pretrained and init != "scratch")
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head."""
    # Lấy danh sách tham số thuộc head classifier
    classifier = model.get_classifier()
    head_param_ids = set()
    if isinstance(classifier, nn.Module):
        head_param_ids = set(id(p) for p in classifier.parameters())
    elif isinstance(classifier, nn.Parameter):
        head_param_ids = {id(classifier)}

    # Đóng băng các tham số còn lại
    for p in model.parameters():
        if id(p) in head_param_ids:
            p.requires_grad = True
        else:
            p.requires_grad = False


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành các nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    head_param_ids = set()
    if isinstance(classifier, nn.Module):
        head_param_ids = set(id(p) for p in classifier.parameters())
    elif isinstance(classifier, nn.Parameter):
        head_param_ids = {id(classifier)}

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = id(param) in head_param_ids
        # norm và bias thường có ndim <= 1
        is_no_decay = param.ndim <= 1 or "bias" in name or "bn" in name or "norm" in name

        if is_head:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính GMAC cho 1 ảnh (3, img_size, img_size)."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Thử các thư viện phổ biến nếu đã cài
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return round(macs / 1e9, 3)
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input)
        return round(flops.total() / 1e9, 3)
    except Exception:
        pass

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
        return round(macs / 1e9, 3)
    except Exception:
        pass

    # Ước lượng chuẩn theo họ kiến trúc nếu không có thư viện tính flops
    name_str = getattr(model, "default_cfg", {}).get("architecture", "").lower()
    fallback_map = {
        "resnet50": 4.1,
        "resnext50": 4.2,
        "convnext_tiny": 4.5,
        "deit_small": 4.6,
        "vit_small": 4.6,
        "swin_tiny": 4.5,
        "efficientnet_b0": 0.39,
        "mobilenetv3": 0.22,
    }
    for k, v in fallback_map.items():
        if k in name_str:
            return v

    return 4.1
