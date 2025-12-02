import torch

from src.metrics.base_metric import BaseMetric


class HitRateMetric(BaseMetric):
    def __init__(
        self, k, n_items, item_counts, num_bins=None, bin_idx=None, *args, **kwargs
    ):
        """
        Hit rate at k samples metric.

        Args:
            k (int): number of sampled items for each user
            n_items (int): number of items in the dataset
            item_counts (dict): popularity of items in the test dataset.
                Used for counting metric per the popularity bins
            num_bins (int | None): if not None,
                number of total popularity bins with equal number of items
            bin_idx (int | None): it not None, index of the bin for this metric
        """
        if num_bins is not None and bin_idx is not None:
            name = kwargs.get("name", type(self).__name__)
            name = f"{name}_bin_{bin_idx}"
            kwargs["name"] = name
        super().__init__(*args, **kwargs)
        self._k = k
        self._n_items = n_items
        self._bin_mask = None
        if num_bins is not None:
            assert bin_idx is not None and bin_idx < num_bins, "Incorrect pop bin index"
            counts = torch.zeros(self._n_items, dtype=torch.long)
            ids = torch.tensor(list(item_counts.keys()), dtype=torch.long)
            vals = torch.tensor(list(item_counts.values()), dtype=torch.long)
            counts[ids] = vals
            sorted_ids = torch.argsort(counts)
            sorted_counts = counts[sorted_ids]
            cumsum = torch.cumsum(sorted_counts, dim=0)
            total = cumsum[-1]
            bin_ids = (cumsum * num_bins / total).long().clamp(0, num_bins - 1)
            self._bin_mask = torch.zeros_like(counts, dtype=torch.long)
            self._bin_mask[sorted_ids] = bin_ids
            self._bin_mask = self._bin_mask == bin_idx

    def __call__(
        self,
        logits: torch.Tensor,
        item: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions. Shape (B, L, N_items)
            item (Tensor): ground-truth labels. Shape (B)
            attention_mask (Tensor): attention mask indicating non-pad tokens. Shape (B, L)
        Returns:
            metric (float | None): calculated metric.
                If no items from batch appear in bins, returns None
        """
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_indices, last_indices, :]
        _, top_items = torch.topk(last_logits, k=self._k, dim=1)
        hit = top_items == item.reshape(-1, 1)
        if self._bin_mask is not None:
            if self._bin_mask.device != item.device:
                self._bin_mask = self._bin_mask.to(item.device)
            hit = hit[self._bin_mask[item], :]
            if hit.shape[0] == 0:
                return None
        return torch.any(hit, dim=1).to(torch.float32).mean()


class NDCGMetric(HitRateMetric):
    def __init__(self, *args, **kwargs):
        """
        NDCG at k samples metric.

        Args:
            k (int): number of sampled items for each user
            n_items (int): number of items in the dataset
            item_counts (dict): popularity of items in the test dataset.
                Used for counting metric per the popularity bins
            num_bins (int | None): if not None,
                number of total popularity bins with equal number of items
            bin_idx (int | None): it not None, index of the bin for this metric
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        logits: torch.Tensor,
        item: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions. Shape (B, L, N_items)
            item (Tensor): ground-truth labels. Shape (B)
            attention_mask (Tensor): attention mask indicating non-pad tokens. Shape (B, L)
        Returns:
            metric (float | None): calculated metric.
                If no items from batch appear in bins, returns None
        """
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_indices, last_indices, :]
        _, top_items = torch.topk(last_logits, k=self._k, dim=1)
        hit = top_items == item.reshape(-1, 1)
        if self._bin_mask is not None:
            if self._bin_mask.device != item.device:
                self._bin_mask = self._bin_mask.to(item.device)
            hit = hit[self._bin_mask[item], :]
            if hit.shape[0] == 0:
                return None
        logs = torch.log2(torch.arange(2, self._k + 2, device=logits.device))
        # since only one item is relevant, IDCG is 1
        ndcg = torch.sum(hit / logs, dim=1)
        return ndcg.mean()


class CoverageMetric(HitRateMetric):
    def __init__(self, *args, **kwargs):
        """
        Coverage at k samples metric.
        This metric is special and its final value is handled
            inside this class instead of MetricTracker.

        Args:
            k (int): number of sampled items for each user
            n_items (int): number of items in the dataset
            item_counts (dict): popularity of items in the test dataset.
                Used for counting metric per the popularity bins
            num_bins (int | None): if not None,
                number of total popularity bins with equal number of items
            bin_idx (int | None): it not None, index of the bin for this metric
        """
        super().__init__(*args, **kwargs)
        self._seen_items = set()
        if self._bin_mask is not None:
            self._n_items = torch.sum(self._bin_mask)

    def __call__(
        self,
        logits: torch.Tensor,
        item: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions. Shape (B, L, N_items)
            item (Tensor): ground-truth labels. Shape (B)
            attention_mask (Tensor): attention mask indicating non-pad tokens. Shape (B, L)
        Returns:
            metric (float): calculated metric.
        """
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_indices, last_indices, :]
        _, top_items = torch.topk(last_logits, k=self._k, dim=1)
        if self._bin_mask is not None:
            if self._bin_mask.device != item.device:
                self._bin_mask = self._bin_mask.to(item.device)
            top_items = top_items[self._bin_mask[item], :]
        self._seen_items.update(top_items.flatten().cpu().tolist())
        return len(self._seen_items) / self._n_items

    def reset(self):
        """
        Reset seen items
        """
        self._seen_items = set()
