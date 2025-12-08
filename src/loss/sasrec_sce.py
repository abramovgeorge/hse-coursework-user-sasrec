import torch
from torch import nn


class SASRecSCELoss(nn.Module):
    """
    Scalable cross-entropy loss for SASRec. https://arxiv.org/abs/2409.18721
    """

    def __init__(self, n_buckets, bucket_size_x, bucket_size_y, mix_x, **data_kwargs):
        """
        Args:
            n_buckets (int): number of buckets
            bucket_size_x (int): bucket size for inputs
            bucket_size_y (int): bucket size for targets
            mix_x (bool): if True, mix hidden states with random matrix
        """
        super().__init__()
        self._n_buckets = n_buckets
        self._bucket_size_x = bucket_size_x
        self._bucket_size_y = bucket_size_y
        self._mix_x = mix_x

    def forward(self, seq: torch.tensor, sce_loss_fn, **batch):
        """
        Autoregressive SCE loss function calculation logic.

        Args:
            seq (torch.tensor): sequence of items in a session. Shape: (B, L)
            sce_loss_fn (function): _sce_forward wrapper
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        labels = seq[:, 1:]
        loss = sce_loss_fn(
            target=labels,
            n_buckets=self._n_buckets,
            bucket_size_x=self._bucket_size_x,
            bucket_size_y=self._bucket_size_y,
            mix_x=self._mix_x,
        )
        return {"loss": loss}
