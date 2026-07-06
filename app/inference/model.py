"""Encoder + classification head used at inference time.

Self-contained copy of ``training/classifier.py`` so the deployed ``app/``
folder rebuilds the exact architecture used during training before loading the
bundled ``state_dict`` checkpoints.
"""

from __future__ import annotations

import re

import torch.nn as nn
from transformers import AutoModel

_LAYER_INDEX = re.compile(r"\.layer\.(\d+)\.")


def _set_encoder_trainable_layers(
    encoder: nn.Module, freeze_until: int | None
) -> None:
    """Freeze whole encoder; if ``freeze_until`` is set, unfreeze layers > it."""
    for _, p in encoder.named_parameters():
        p.requires_grad = False
    if freeze_until is None:
        return
    for name, p in encoder.named_parameters():
        m = _LAYER_INDEX.search(name)
        if m is not None and int(m.group(1)) > freeze_until:
            p.requires_grad = True


class TransformerClassifier(nn.Module):
    """Linear( H->512 ) -> ReLU -> Dropout(0.1) -> Linear(512->C) -> LogSoftmax."""

    def __init__(
        self,
        hf_name: str,
        freeze_until: int | None,
        *,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_name)
        _set_encoder_trainable_layers(self.encoder, freeze_until)
        hidden = int(self.encoder.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)
