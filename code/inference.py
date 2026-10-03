"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy
from typing import List, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

# Import hằng số từ eval.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as ev


def softmax_np(z: np.ndarray) -> np.ndarray:
    """Hàm Softmax an toàn cho mảng numpy."""
    z_max = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - z_max)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.maximum(s, 1e-12)


def predict_logits(model: nn.Module, loader, device: torch.device, view: Callable | None = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_fnames = []
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels, filenames in loader:
            if view is not None:
                images = view(images)

            images = images.to(device, non_blocking=True)
            logits = model(images)

            all_fnames.extend(filenames)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.append(logits.cpu().numpy())

    y_true = np.array(all_labels, dtype=int)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, ev.NUM_CLASSES))
    return all_fnames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Không biến đổi ảnh."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop: 4 góc + 1 trung tâm."""
    _, _, h, w = x.shape
    if h < crop or w < crop:
        return [x]

    top_left = x[:, :, :crop, :crop]
    top_right = x[:, :, :crop, w - crop:]
    bottom_left = x[:, :, h - crop:, :crop]
    bottom_right = x[:, :, h - crop:, w - crop:]

    start_h = (h - crop) // 2
    start_w = (w - crop) // 2
    center = x[:, :, start_h:start_h + crop, start_w:start_w + crop]

    return [top_left, top_right, bottom_left, bottom_right, center]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    res = []
    for s in sizes:
        scaled = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        res.append(scaled)
    return res


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một ma trận xác suất (N, 9).

    - space="prob": trung bình softmax của từng view
    - space="logit": trung bình logit rồi softmax
    """
    if len(logits_per_view) == 0:
        raise ValueError("Danh sách logits_per_view không được rỗng")

    if space == "prob":
        probs_list = [softmax_np(lg) for lg in logits_per_view]
        avg_prob = np.mean(probs_list, axis=0)
        return avg_prob / np.sum(avg_prob, axis=1, keepdims=True)
    elif space == "logit":
        avg_logit = np.mean(logits_per_view, axis=0)
        return softmax_np(avg_logit)
    else:
        raise ValueError(f"Không hỗ trợ không gian gộp: '{space}' (chỉ chọn 'prob' hoặc 'logit')")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình."""
    if len(list_of_probs) == 0:
        raise ValueError("list_of_probs không được rỗng")
    avg = np.mean(list_of_probs, axis=0)
    return avg / np.sum(avg, axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL (Negative Log-Likelihood) trên tập VAL."""
    val_labels = np.asarray(val_labels, dtype=int)
    n = len(val_labels)

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        scaled = val_logits / T
        probs = softmax_np(scaled)
        correct_probs = np.clip(probs[np.arange(n), val_labels], 1e-12, 1.0)
        return float(-np.mean(np.log(correct_probs)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T)."""
    return softmax_np(logits / max(1e-5, T))


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (Conv2d + BatchNorm2d)."""
    fused_model = copy.deepcopy(model)
    fused_model.eval()

    try:
        torch.nn.utils.fuse_conv_bn_eval(fused_model)
    except Exception:
        # Tự động duyệt và gộp thủ công các block Sequential
        for name, child in fused_model.named_children():
            if isinstance(child, nn.Sequential):
                for i in range(len(child) - 1):
                    if isinstance(child[i], nn.Conv2d) and isinstance(child[i+1], nn.BatchNorm2d):
                        conv = child[i]
                        bn = child[i+1]
                        w_conv = conv.weight.clone().view(conv.out_channels, -1)
                        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
                        fused_w = torch.mm(w_bn, w_conv).view(conv.weight.size())

                        if conv.bias is not None:
                            b_conv = conv.bias
                        else:
                            b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
                        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
                        fused_b = torch.matmul(w_bn, b_conv) + b_bn

                        child[i].weight.data.copy_(fused_w)
                        if child[i].bias is None:
                            child[i].bias = nn.Parameter(fused_b)
                        else:
                            child[i].bias.data.copy_(fused_b)
                        child[i+1] = nn.Identity()
            else:
                fuse_conv_bn(child)

    return fused_model


def test_inference_sanity() -> bool:
    """Unit test kiểm tra các hàm suy luận."""
    # 1. Softmax
    z = np.random.randn(5, 9)
    p = softmax_np(z)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(5), atol=1e-6)

    # 2. Temperature scaling
    y = np.random.randint(0, 9, size=100)
    T = fit_temperature(np.random.randn(100, 9), y)
    assert 0.05 <= T <= 10.0, f"T nằm ngoài biên hợp lệ: {T}"

    # 3. Fuse Conv-BN
    conv = nn.Conv2d(3, 16, 3, padding=1)
    bn = nn.BatchNorm2d(16)
    seq = nn.Sequential(conv, bn)
    seq.eval()
    dummy = torch.randn(2, 3, 32, 32)
    out_orig = seq(dummy)
    fused_seq = fuse_conv_bn(seq)
    out_fused = fused_seq(dummy)
    diff = (out_orig - out_fused).abs().max().item()
    assert diff < 1e-4, f"Sai số sau gộp Conv-BN quá lớn: {diff}"

    return True


if __name__ == "__main__":
    test_inference_sanity()
    print("Sanity checks trong inference.py: PASSED")
