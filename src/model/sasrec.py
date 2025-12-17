from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model from https://arxiv.org/pdf/1808.09781
    Implementation is taken from https://github.com/AIRI-Institute/Scalable-SASRec
    """

    def __init__(
        self,
        hidden_dim,
        num_blocks,
        max_len,
        pad_token,
        num_heads,
        dropout_rate,
        n_items,
        loss_class,
        **data_kwargs,
    ):
        """
        Args:
            hidden_dim (int): hidden dimension of the transformer.
            num_blocks (int): number of transformer blocks.
            max_len (int): maximum length of a input sequence.
            pad_token (int): padding token.
            num_heads (int): number of head for multi-head attention.
            dropout_rate (float): dropout rate for dropout layers.
            n_items (int): number of items in the dataset.
            loss_class (str): loss function class.
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_blocks = num_blocks
        self._max_len = max_len
        self._pad_token = pad_token
        self._num_heads = num_heads
        self._p = dropout_rate
        self._n_items = n_items
        loss_class_map = {
            "src.loss.SASRecCELoss": "ce",
            "src.loss.SASRecBCELoss": "bce",
            "src.loss.SASRecSCELoss": "sce",
        }
        self._loss_type = loss_class_map[loss_class]

        self.item_emb = nn.Embedding(
            self._n_items + 1, self._hidden_dim, padding_idx=self._pad_token
        )
        self.pos_emb = nn.Embedding(self._max_len, self._hidden_dim)
        self.emb_dropout = nn.Dropout(self._p)

        self.attention_layernorms = nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self._hidden_dim, eps=1e-8)

        for _ in range(self._num_blocks):
            new_attn_layernorm = nn.LayerNorm(self._hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = nn.MultiheadAttention(
                self._hidden_dim, self._num_heads, self._p
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self._hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self._hidden_dim, self._p)
            self.forward_layers.append(new_fwd_layer)

        self._initialize()

    def _initialize(self):
        for _, param in self.named_parameters():
            try:
                nn.init.xavier_uniform_(param.data)
            except:  # noqa: E722
                pass  # just ignore those failed init layers

    def _get_hidden_states(self, seq, **batch):
        """
        Args:
            seq (torch.tensor): tensor of shape (B, L) containing input sequences
        Returns:
            hidden_states (torch.tensor): last hidden states tensor.
        """
        device = seq.device
        seqs = self.item_emb(seq)
        seqs *= self.item_emb.embedding_dim**0.5
        positions = np.tile(np.arange(seq.shape[1]), [seq.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = seq == self._pad_token
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        hidden_states = self.last_layernorm(seqs)  # (B, L, D)
        return hidden_states

    def _forward_bce(self, hidden_states, pos_items, neg_items, **batch):
        """
        This function calculates binary cross-entropy loss with negative sampling

        Args:
            hidden_states (torch.tensor): tensor containing precomputed
                hidden_states. Shape: (B, L, D)
            pos_items (torch.tensor): tensor containing next items
                (i.e., targets) for next-item prediction. Shape: (B, L - 1)
            neg_items (torch.tensor): tensor containing N wrong items
                for each position for next-item prediction. Shape: (B, L - 1, N)
        Returns:
            output (dict): output dict containing logits for pos_items and neg_items.
        """

        hidden_states = hidden_states[:, :-1, :]  # (B, L - 1, D)

        pos_embs = self.item_emb(pos_items)  # (B, L - 1, D)
        neg_emds = self.item_emb(neg_items)  # (B, L - 1, N, D)

        pos_logits = torch.einsum(
            "bld, bld -> bl", hidden_states, pos_embs
        )  # (B, L - 1)
        neg_logits = torch.einsum(
            "bld, blnd -> bln", hidden_states, neg_emds
        )  # (B, L - 1, N)

        return {"pos_logits": pos_logits, "neg_logits": neg_logits}

    def _forward_sce(
        self,
        seq,
        hidden_states,
        target,
        n_buckets,
        bucket_size_x,
        bucket_size_y,
        mix_x,
        **batch,
    ):
        """
        This function calculates Scalable cross-entropy loss
            from https://arxiv.org/abs/2409.18721

        Args:
            seq (torch.tensor): tensor containing input sequences. Shape: (B, L)
            hidden_states (torch.tensor): tensor containing precomputed
                hidden_states. Shape: (B, L, D)
            target (torch.tensor): tensor containing next items
                (i.e., targets) for next-item prediction. Shape: (B, L - 1)
            n_buckets (int): number of buckets
            bucket_size_x (int): bucket size for inputs
            bucket_size_y (int): bucket size for targets
            mix_x (bool): if True, mix hidden states with random matrix
        Returns:
            output (dict): output dict containing SCE loss.
        """

        seq = seq[:, :-1]
        hidden_states = hidden_states[:, :-1, :]

        hd = hidden_states.shape[-1]

        x = hidden_states.reshape(-1, hd)
        y = target.reshape(-1)
        w = self.item_emb.weight

        correct_class_logits_ = (x * torch.index_select(w, dim=0, index=y)).sum(
            dim=1
        )  # (bs,)

        with torch.no_grad():
            if mix_x:
                omega = (
                    1
                    / np.sqrt(np.sqrt(hd))
                    * torch.randn(x.shape[0], n_buckets, device=x.device)
                )
                buckets = omega.T @ x
                del omega
            else:
                buckets = (
                    1
                    / np.sqrt(np.sqrt(hd))
                    * torch.randn(n_buckets, hd, device=x.device)
                )  # (n_b, hd)

        with torch.no_grad():
            x_bucket = buckets @ x.T  # (n_b, hd) x (hd, b) -> (n_b, b)
            x_bucket[:, seq.reshape(-1) == self.pad_token] = float("-inf")
            _, top_x_bucket = torch.topk(
                x_bucket, dim=1, k=bucket_size_x
            )  # (n_b, bs_x)
            del x_bucket

            y_bucket = buckets @ w.T  # (n_b, hd) x (hd, n_cl) -> (n_b, n_cl)

            y_bucket[:, self.pad_token] = float("-inf")
            _, top_y_bucket = torch.topk(
                y_bucket, dim=1, k=bucket_size_y
            )  # (n_b, bs_y)
            del y_bucket

        x_bucket = torch.gather(x, 0, top_x_bucket.view(-1, 1).expand(-1, hd)).view(
            n_buckets, bucket_size_x, hd
        )  # (n_b, bs_x, hd)
        y_bucket = torch.gather(w, 0, top_y_bucket.view(-1, 1).expand(-1, hd)).view(
            n_buckets, bucket_size_y, hd
        )  # (n_b, bs_y, hd)

        wrong_class_logits = x_bucket @ y_bucket.transpose(-1, -2)  # (n_b, bs_x, bs_y)
        mask = (
            torch.index_select(y, dim=0, index=top_x_bucket.view(-1)).view(
                n_buckets, bucket_size_x
            )[:, :, None]
            == top_y_bucket[:, None, :]
        )  # (n_b, bs_x, bs_y)
        wrong_class_logits = wrong_class_logits.masked_fill(
            mask, float("-inf")
        )  # (n_b, bs_x, bs_y)
        correct_class_logits = torch.index_select(
            correct_class_logits_, dim=0, index=top_x_bucket.view(-1)
        ).view(n_buckets, bucket_size_x)[
            :, :, None
        ]  # (n_b, bs_x, 1)
        logits = torch.cat(
            (wrong_class_logits, correct_class_logits), dim=2
        )  # (n_b, bs_x, bs_y + 1)

        loss_ = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            (logits.shape[-1] - 1)
            * torch.ones(
                logits.shape[0] * logits.shape[1],
                dtype=torch.int64,
                device=logits.device,
            ),
            reduction="none",
        )  # (n_b * bs_x,)
        loss = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        loss.scatter_reduce_(
            0, top_x_bucket.view(-1), loss_, reduce="amax", include_self=False
        )
        loss = loss[(loss != 0) & (y != self.pad_token)]
        loss = torch.mean(loss)

        return {"sce_loss": loss}

    def _get_last_logits(self, hidden_states, attention_mask, **batch):
        """
        Get last logits for metric calculation
        Args:
            hidden_states (torch.tensor): tensor containing precomputed
                hidden_states. Shape: (B, L, D)
            attention_mask (torch.tensor): attention mask indicating non-pad tokens. Shape (B, L)
        Returns:
            output (dict): output dict containing last logit tensor of shape (B, N_items)
        """
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(
            hidden_states.shape[0], device=hidden_states.device
        )
        last_states = hidden_states[batch_indices, last_indices, :]  # (B, D)
        last_logits = last_states @ self.item_emb.weight.T  # (B, N_items)
        return last_logits

    def forward(self, seq, **batch):
        """
        Args:
            seq (torch.tensor): tensor of shape (B, L) containing input sequences
        Returns:
            output (dict): output dict containing appropriate forward output for CE type.
        """

        hidden_states = self._get_hidden_states(seq)

        output = dict()
        output["last_logits"] = self._get_last_logits(hidden_states, **batch)

        if self._loss_type == "sce":
            output["loss_fn"] = partial(
                self._forward_sce, hidden_states=hidden_states, seq=seq
            )
            return output
        elif self._loss_type == "bce":
            output["loss_fn"] = partial(self._forward_bce, hidden_states=hidden_states)
            return output

        logits = hidden_states @ self.item_emb.weight.T  # (B, L, n_item)
        output["logits"] = logits

        return output

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
