"""
Two-Stage Hybrid Spatial + Sequence Model Architecture for Multi-Label Knee MRI Detection.
Stage 1: CAFormer-S36 2D spatial encoder with auxiliary plane classification head (planeaux01).
Stage 2: Bidirectional Series GRU + Soft Log-Sum-Exp (SoftLSE) volumetric pooling across slices.
"""

import torch
import torch.nn as nn
import timm


class SoftLSEPool(nn.Module):
    """Smooth attention-like pooling across slice embeddings using Log-Sum-Exp."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: (batch_size, num_slices, num_classes)
        scaled = x / self.temperature
        if mask is not None:
            scaled = scaled.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        return self.temperature * torch.logsumexp(scaled, dim=1)


class KneeHybridNet(nn.Module):
    """Hybrid ConvNet-Transformer (CAFormer-S36) + Bidirectional Series GRU."""

    def __init__(
        self,
        backbone_name: str = "caformer_s36.sail_in22k_ft_in1k_384",
        pretrained: bool = False,
        num_classes: int = 12,
        gru_hidden: int = 256,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.encoder.num_features

        # Auxiliary head for anatomical plane supervision (Axial, Sagittal, Coronal)
        self.plane_head = nn.Linear(feat_dim, 3)

        # Bidirectional sequence modeling across ordered MRI slices
        self.series_gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)
        self.pool = SoftLSEPool(temperature=1.0)

    def forward(self, slices: torch.Tensor, mask: torch.Tensor = None):
        # slices shape: (B, S, C, H, W)
        b, s, c, h, w = slices.shape
        flat_slices = slices.view(b * s, c, h, w)

        slice_feats = self.encoder(flat_slices)
        plane_logits = self.plane_head(slice_feats).view(b, s, 3)

        seq_feats = slice_feats.view(b, s, -1)
        gru_out, _ = self.series_gru(seq_feats)
        slice_logits = self.classifier(gru_out)

        study_logits = self.pool(slice_logits, mask=mask)
        return study_logits, plane_logits
