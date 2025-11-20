import torch

from src.metrics.base_metric import BaseMetric


class HitRateMetric(BaseMetric):
    def __init__(self, k, *args, **kwargs):
        """
        Hit rate at k samples metric.

        Args:
            k (int): number of sampled items for each user
        """
        super().__init__(*args, **kwargs)
        self._k = k

    def __call__(self, logits: torch.Tensor, item: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions. Shape (B, L, N_items)
            item (Tensor): ground-truth labels. Shape (B)
        Returns:
            metric (float): calculated metric.
        """
        last_logits = logits[:, -1, :]
        _, top_items = torch.topk(last_logits, k=self._k, dim=1)
        hit = top_items == item.reshape(-1, 1)
        return torch.any(hit, dim=1).to(torch.float32).mean()
