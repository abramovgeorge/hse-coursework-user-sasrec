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


class SASRecBCELoss(nn.Module):
    """
    Binary cross-entropy loss with negative sampling for SASRec.
    """

    def __init__(self, num_neg, n_items, **data_kwargs):
        """
        Args:
            num_neg (int): number of negative items to sample
            n_items (int): number of items in the dataset
        """
        super().__init__()
        self._num_neg = num_neg
        self._n_items = n_items
        self._criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self, seq: torch.tensor, attention_mask: torch.tensor, loss_fn, **batch
    ):
        """
        BCE loss function calculation logic.

        Args:
            seq (torch.tensor): sequence of items in a session. Shape: (B, L)
            attention_mask (torch.tensor): attention mask indicating non-pad tokens. Shape (B, L)
            loss_fn (function): _bce_forward wrapper
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        labels = seq[:, 1:]
        attention_mask = attention_mask[:, :-1]
        B, L = labels.shape
        neg_items = torch.randint(
            low=0, high=self._n_items, size=(B, L, self._num_neg), device=seq.device
        )
        bce_output = loss_fn(pos_items=labels, neg_items=neg_items)

        pos_logits = bce_output["pos_logits"]  # (B, L - 1)
        neg_logits = bce_output["neg_logits"]  # (B, L - 1, N)
        pos_targets = torch.ones_like(pos_logits)
        neg_targets = torch.zeros_like(neg_logits)

        pos_loss = self._criterion(pos_logits, pos_targets)
        neg_loss = self._criterion(neg_logits, neg_targets)

        pos_loss[~attention_mask.bool()] = 0
        neg_loss[~attention_mask.unsqueeze(-1).expand_as(neg_loss).bool()] = 0

        loss = pos_loss.sum() + neg_loss.sum()
        loss /= attention_mask.sum()

        return {"loss": loss}


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

    def forward(self, seq: torch.tensor, loss_fn, **batch):
        """
        SCE loss function calculation logic.

        Args:
            seq (torch.tensor): sequence of items in a session. Shape: (B, L)
            loss_fn (function): _sce_forward wrapper
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        labels = seq[:, 1:]
        loss = loss_fn(
            target=labels,
            n_buckets=self._n_buckets,
            bucket_size_x=self._bucket_size_x,
            bucket_size_y=self._bucket_size_y,
            mix_x=self._mix_x,
        )
        return {"loss": loss}
