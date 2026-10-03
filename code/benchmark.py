"""benchmark.py - đo độ trễ suy luận đúng cách.

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau
  - >= 50 lần đo, báo cáo p50, p95, p99
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, phiên bản torch
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable, warmup: int = 10, iters: int = 100, sync: Callable | None = None) -> Dict[str, Any]:
    """Đo thời gian thực thi hàm `fn()` (mili-giây)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo lặp lại
    latencies = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur_ms)

    lat_arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(lat_arr, 50)), 3),
        "p95": round(float(np.percentile(lat_arr, 95)), 3),
        "p99": round(float(np.percentile(lat_arr, 99)), 3),
        "mean": round(float(np.mean(lat_arr)), 3),
        "n": iters
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    dev = torch.device(device if (torch.cuda.is_available() and device == "cuda") else "cpu")
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    if dtype == "amp":
        def fn():
            with torch.inference_mode(), torch.amp.autocast(device_type=dev.type):
                _ = model(dummy_input)
    else:
        def fn():
            with torch.inference_mode():
                _ = model(dummy_input)

    res = bench(fn, warmup=warmup, iters=iters, sync=sync_fn)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    p50 = max(1e-4, res["p50"])
    images_per_s = round(batch_size / (p50 / 1000.0), 2)

    report = {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__
    }
    return report


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> Dict[str, Any]:
    """Đo độ trễ của phương pháp TTA K-views."""
    dev = torch.device(device if (torch.cuda.is_available() and device == "cuda") else "cpu")
    model = model.to(dev)
    model.eval()

    dummy_inputs = [torch.randn(batch_size, 3, img_size, img_size, device=dev) for _ in range(k_views)]
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    def fn():
        with torch.inference_mode():
            for v_input in dummy_inputs:
                _ = model(v_input)

    res = bench(fn, warmup=warmup, iters=iters, sync=sync_fn)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    images_per_s = round(batch_size / (max(1e-4, res["p50"]) / 1000.0), 2)

    return {
        "gpu": gpu_name,
        "method": f"TTA (K={k_views})",
        "batch": batch_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "images_per_s": images_per_s
    }
