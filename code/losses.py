"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

from typing import Union, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57)."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        target_view = target.unsqueeze(1)
        log_pt = log_p.gather(1, target_view).squeeze(1)
        pt = p.gather(1, target_view).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, target)
            loss = loss * alpha_t

        return loss.mean()


def class_weights(counts: Union[dict, list, np.ndarray, torch.Tensor], beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    if isinstance(counts, dict):
        keys = sorted(counts.keys())
        counts_arr = np.array([counts[k] for k in keys], dtype=np.float64)
    elif isinstance(counts, torch.Tensor):
        counts_arr = counts.detach().cpu().numpy().astype(np.float64)
    else:
        counts_arr = np.array(counts, dtype=np.float64)

    num_classes = len(counts_arr)

    if beta <= 0.0:
        inv = 1.0 / np.maximum(counts_arr, 1.0)
        weights = inv / inv.mean()
    else:
        # Class-balanced loss sử dụng effective number of samples
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix)."""
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[index]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, h, w = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        actual_lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return x_mixed, (y_a, y_b, actual_lam)

    return x, (y, y, 1.0)


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def test_losses_sanity() -> bool:
    """Unit test kiểm tra tính đúng đắn của các hàm loss (RUBRIC mục H)."""
    # 1. Focal Loss với gamma=0 phải bằng Cross Entropy Loss
    torch.manual_seed(42)
    logits = torch.randn(10, 9)
    target = torch.randint(0, 9, (10,))
    ce_val = F.cross_entropy(logits, target).item()
    focal_val = FocalLoss(gamma=0.0)(logits, target).item()
    assert abs(ce_val - focal_val) < 1e-6, f"Focal gamma=0 ({focal_val}) không khớp CE ({ce_val})"

    # 2. Label Smoothing với eps=0 phải bằng CE
    ls_val = LabelSmoothingCE(smoothing=0.0)(logits, target).item()
    assert abs(ce_val - ls_val) < 1e-6, f"LS eps=0 ({ls_val}) không khớp CE ({ce_val})"

    # 3. Trọng số lớp
    counts = [1000] * 8 + [9000]
    w = class_weights(counts, beta=0.0)
    assert len(w) == 9 and abs(w.mean().item() - 1.0) < 1e-5

    return True


if __name__ == "__main__":
    test_losses_sanity()
    print("Sanity checks trong losses.py: PASSED")
