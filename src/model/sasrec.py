import numpy as np
import torch
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
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_blocks = num_blocks
        self._max_len = max_len
        self._pad_token = pad_token
        self._num_heads = num_heads
        self._p = dropout_rate
        self._n_items = n_items

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

    def forward(self, seq, **batch):
        """
        Args:
            seq (torch.tensor): tensor of shape (B, L) containing input sequences
        Returns:
            output (dict): output dict containing hidden states after last layer.
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

        logits = hidden_states @ self.item_emb.weight.T  # (B, L, n_item)

        return {"logits": logits}

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
