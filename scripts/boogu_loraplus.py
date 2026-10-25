"""AdamW optimizer with the standard LoRA+ B-to-A learning-rate ratio."""

import torch


class LoRAPlusAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, b_lr_ratio=16.0, **kwargs):
        groups = {"A": [], "B": [], "other": []}
        for param in list(params):
            side = getattr(param, "_boogu_loraplus_side", "other")
            groups[side if side in ("A", "B") else "other"].append(param)

        optimizer_groups = []
        if groups["A"]:
            optimizer_groups.append({"params": groups["A"], "lr": lr})
        if groups["B"]:
            optimizer_groups.append({"params": groups["B"], "lr": lr * b_lr_ratio})
        if groups["other"]:
            optimizer_groups.append({"params": groups["other"], "lr": lr})
        if not optimizer_groups:
            raise ValueError("LoRA+ received no trainable parameters.")

        super().__init__(optimizer_groups, lr=lr, weight_decay=weight_decay, **kwargs)
