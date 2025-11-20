import torch
from torch import nn


class SASRecCELoss(nn.Module):
    """
    Full cross-entropy loss for SASRec
    """

    def __init__(self, pad_token, **data_kwargs):
        """
        Args:
            pad_token (int): pad token to ignore in loss calculation.
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            ignore_index=pad_token,
        )

    def forward(self, logits: torch.tensor, seq: torch.tensor, **batch):
        """
        Autoregressive CE loss function calculation logic.

        Args:
            logits (torch.tensor): model output predictions. Shape: (B, L, N_items)
            seq (torch.tensor): sequence of items in a session. Shape: (B, L)
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        shift_logits = logits[:, :-1, :]
        labels = seq[:, 1:]
        N_items = shift_logits.shape[-1]
        shift_logits = shift_logits.reshape(-1, N_items)
        labels = labels.reshape(-1).to(torch.long)
        return {"loss": self.loss(shift_logits, labels)}
