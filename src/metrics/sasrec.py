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


class NDCGMetric(BaseMetric):
    def __init__(self, k, *args, **kwargs):
        """
        NDCG at k samples metric.

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
        logs = torch.log2(torch.arange(2, self._k + 2, device=logits.device))
        # since only one item is relevant, IDCG is 1
        ndcg = torch.sum(hit / logs, dim=1)
        return ndcg.mean()


class CoverageMetric(BaseMetric):
    def __init__(self, k, n_items, *args, **kwargs):
        """
        Coverage at k samples metric.
        This metric is special and its final value is handled
            inside this class instead of MetricTracker.

        Args:
            k (int): number of sampled items for each user
            n_items (int): number of items in the dataset
        """
        super().__init__(*args, **kwargs)
        self._k = k
        self._n_items = n_items
        self._seen_items = set()

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
        self._seen_items.update(top_items.flatten().cpu().tolist())
        return len(self._seen_items) / self._n_items

    def reset(self):
        """
        Reset seen items
        """
        self._seen_items = set()
