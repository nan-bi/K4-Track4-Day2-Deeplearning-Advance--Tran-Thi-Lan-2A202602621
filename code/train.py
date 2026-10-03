"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H).
Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dataset
import model as model_utils
import losses as loss_utils
import inference as inf_utils

# Import eval từ repo gốc
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(net: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (Slide 52)."""
    groups = model_utils.param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (Slide 55)."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, net: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(net)
        for p in self.shadow.parameters():
            p.requires_grad = False
        self.shadow.eval()

    def update(self, net: nn.Module) -> None:
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), net.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
            for s_buf, m_buf in zip(self.shadow.buffers(), net.buffers()):
                s_buf.copy_(m_buf)

    def copy_to(self, net: nn.Module) -> None:
        net.load_state_dict(self.shadow.state_dict())


def train_one_epoch(net: nn.Module, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    net.train()
    if cfg.init == "frozen":
        # Giữ phần backbone và BatchNorm ở chế độ eval
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    total_loss = 0.0
    num_batches = len(loader)

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg.mix in ["mixup", "cutmix"]:
            images, targets = loss_utils.mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
        else:
            targets = None

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.amp):
            logits = net(images)
            if targets is not None:
                loss = loss_utils.mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, labels)

        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update(net)

        total_loss += loss.item()

    avg_loss = total_loss / max(1, num_batches)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": avg_loss, "lr": current_lr}


def evaluate(net: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên loader ở chế độ eval, KHÔNG tính gradient."""
    net.eval()
    all_filenames = []
    all_labels = []
    all_logits = []
    total_loss = 0.0

    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = net(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            all_filenames.extend(filenames)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.append(logits.cpu().numpy())

    y_true = np.array(all_labels, dtype=int)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, ev.NUM_CLASSES))
    avg_loss = total_loss / max(1, len(loader)) if criterion is not None else 0.0

    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color=color, fontsize=11)
    line1 = ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linestyle="--", linewidth=1.8)
    line2 = ax1.plot(epochs, val_loss, label="Val Loss", color="tab:cyan", linewidth=1.8)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val Macro-F1", color=color, fontsize=11)
    line3 = ax2.plot(epochs, val_f1, label="Val Macro-F1", color="tab:red", linewidth=2.2)
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết (GUIDE.md mục 0-5)."""
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    # Lưu cấu hình json
    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Nạp và kiểm tra dữ liệu theo S1-S6
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers,
        seed=cfg.seed
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers,
        seed=cfg.seed
    )

    # 2. Khởi tạo mô hình
    net = model_utils.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    # 3. Khởi tạo loss criterion
    criterion_kwargs = {}
    if cfg.loss == "ls":
        criterion_kwargs["smoothing"] = cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1
    elif cfg.loss == "focal":
        criterion_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        class_counts = train_df["Label"].value_counts().to_dict()
        weights = loss_utils.class_weights(class_counts, beta=cfg.class_weight_beta or 0.0)
        criterion_kwargs["weight"] = weights.to(device)

    criterion = loss_utils.build_criterion(kind=cfg.loss, **criterion_kwargs)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(net, decay=cfg.ema_decay) if cfg.ema_decay else None

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_val_probs = None
    best_val_logits = None
    best_val_filenames = None
    best_val_y_true = None

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.time()
        train_res = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)

        # Đánh giá trên tập val
        eval_model = ema.shadow if ema is not None else net
        val_fnames, val_y_true, val_logits, val_loss = evaluate(eval_model, val_loader, eval_criterion, device)

        # Tính xác suất softmax và macro-F1
        val_probs = inf_utils.softmax_np(val_logits)
        metrics = ev.compute_metrics(val_y_true, val_probs)
        val_macro_f1 = metrics["macro_f1"]
        val_acc = metrics["accuracy"]

        ep_duration = time.time() - ep_start
        rec = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_acc": val_acc,
            "lr": train_res["lr"],
            "duration": ep_duration
        }
        history.append(rec)

        # Chọn checkpoint theo Macro-F1 val (nếu hòa lấy epoch sớm hơn)
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            best_val_probs = val_probs
            best_val_logits = val_logits
            best_val_filenames = val_fnames
            best_val_y_true = val_y_true
            torch.save(eval_model.state_dict(), r_dir / "best_model.pth")

    total_duration = time.time() - start_time
    avg_epoch_time = total_duration / max(1, cfg.epochs)

    # Lưu dự đoán val của checkpoint tốt nhất
    if best_val_filenames is not None:
        ev.save_predictions(
            pred_path(cfg, "val"),
            best_val_filenames,
            best_val_y_true,
            best_val_probs
        )

    # Ghi history.csv
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(r_dir / "history.csv", index=False)

    # Vẽ và lưu biểu đồ training
    curve_file = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_file, title=f"Experiment {cfg.exp_id} ({cfg.backbone})")

    # Nếu bật save_test_predictions (CHỈ ở Bước 4 chung kết)
    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, val_tf,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers,
            seed=cfg.seed
        )
        best_net = model_utils.build_model(name=cfg.backbone, pretrained=False, num_classes=ev.NUM_CLASSES).to(device)
        best_net.load_state_dict(torch.load(r_dir / "best_model.pth", map_location=device))
        best_net.eval()

        test_fnames, test_y_true, test_logits, _ = evaluate(best_net, test_loader, None, device)
        uncal_test_probs = inf_utils.softmax_np(test_logits)

        # 1. Lưu dự đoán chưa hiệu chuẩn (uncal) cho eval.py grade I4a
        uncal_pred_file = Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv"
        ev.save_predictions(
            uncal_pred_file,
            test_fnames,
            test_y_true,
            uncal_test_probs
        )

        # 2. Khớp Temperature Scaling trên VAL và áp dụng sang TEST
        if best_val_logits is not None:
            T_opt = inf_utils.fit_temperature(best_val_logits, best_val_y_true)
        else:
            T_opt = 1.0
        cal_test_probs = inf_utils.apply_temperature(test_logits, T_opt)

        # 3. Lưu dự đoán test chính thức
        ev.save_predictions(
            pred_path(cfg, "test"),
            test_fnames,
            test_y_true,
            cal_test_probs
        )
        test_metrics = ev.compute_metrics(test_y_true, cal_test_probs)

    num_params = model_utils.count_params(net)
    gmacs = model_utils.count_gmacs(net, img_size=cfg.img_size)

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "avg_epoch_time": avg_epoch_time,
        "num_params_m": num_params,
        "gmacs": gmacs,
        "test_macro_f1": test_metrics["macro_f1"] if test_metrics else None,
        "test_accuracy": test_metrics["accuracy"] if test_metrics else None
    }
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Chuyển đổi các cặp key=value thành dictionary ép đúng kiểu Config."""
    field_types = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in field_types:
            raise KeyError(f"Config không có thuộc tính '{key}'")

        if val.lower() == "none":
            overrides[key] = None
        elif val.lower() == "true":
            overrides[key] = True
        elif val.lower() == "false":
            overrides[key] = False
        else:
            try:
                if "." in val or "e" in val.lower():
                    overrides[key] = float(val)
                else:
                    overrides[key] = int(val)
            except ValueError:
                overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện một thí nghiệm DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Cấu hình ghi đè: KEY=VAL ...")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"--- Bắt đầu huấn luyện {cfg.exp_id} ({cfg.backbone}, seed={cfg.seed}) ---")
    result = run(cfg)
    print(f"Hoàn thành: Val Macro-F1 = {result['best_val_f1']:.4f} tại epoch {result['best_epoch']}")


if __name__ == "__main__":
    main()
