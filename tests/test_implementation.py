"""test_implementation.py - Kiểm tra toàn diện chất lượng triển khai mã nguồn (100% Passing).
"""
import unittest
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import dataset
import model as model_utils
import losses as loss_utils
import inference as inf_utils
import benchmark
import train
import eval as ev


class TestImplementation(unittest.TestCase):

    def test_model_builder_and_param_groups(self):
        net = model_utils.build_model("resnet50", pretrained=False, num_classes=9)
        self.assertIsInstance(net, nn.Module)
        
        # Test param groups
        groups = model_utils.param_groups(net, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        self.assertGreater(len(groups), 0)
        
        # Verify head LR
        head_lr_found = any(g["lr"] == 1e-3 for g in groups)
        self.assertTrue(head_lr_found)
        
        # Test param counting
        p = model_utils.count_params(net)
        self.assertGreater(p, 20.0)

    def test_losses_and_focal_equivalence(self):
        torch.manual_seed(42)
        logits = torch.randn(10, 9)
        target = torch.randint(0, 9, (10,))
        
        ce_loss = nn.CrossEntropyLoss()(logits, target).item()
        focal_loss = loss_utils.FocalLoss(gamma=0.0)(logits, target).item()
        self.assertAlmostEqual(ce_loss, focal_loss, places=5)
        
        ls_loss = loss_utils.LabelSmoothingCE(smoothing=0.0)(logits, target).item()
        self.assertAlmostEqual(ce_loss, ls_loss, places=5)

    def test_cutmix_batch(self):
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        x_mix, (y_a, y_b, lam) = loss_utils.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_mix.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)

    def test_temperature_scaling(self):
        np.random.seed(42)
        logits = np.random.randn(50, 9)
        y = np.random.randint(0, 9, size=50)
        T = inf_utils.fit_temperature(logits, y)
        self.assertTrue(0.05 <= T <= 10.0)
        
        probs = inf_utils.apply_temperature(logits, T)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(50), atol=1e-6)

    def test_fuse_conv_bn(self):
        conv = nn.Conv2d(3, 8, 3, padding=1)
        bn = nn.BatchNorm2d(8)
        seq = nn.Sequential(conv, bn)
        seq.eval()
        
        dummy = torch.randn(2, 3, 16, 16)
        out1 = seq(dummy)
        fused = inf_utils.fuse_conv_bn(seq)
        out2 = fused(dummy)
        
        diff = (out1 - out2).abs().max().item()
        self.assertLess(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
