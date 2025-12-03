import numpy as np
import torch
from opt_einsum import contract
from torch import nn

from src.model.sasrec import PointWiseFeedForward


class UserSASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model with explicit user modeling
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
        n_users,
        user_handling,
        user_dim=None,
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
            n_users (int): number of users in the dataset.
            user_handling (str): specifies user embedding handling.
            user_dim (int | None): if not None, hidden dimension for the user embeddings,
                different from `hidden_dim`. It is used in Tucker decomposition.
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_blocks = num_blocks
        self._max_len = max_len
        self._pad_token = pad_token
        self._num_heads = num_heads
        self._p = dropout_rate
        self._n_items = n_items
        self._n_users = n_users
        self._user_handling = user_handling
        self._user_dim = hidden_dim if user_dim is None else user_dim

        self.item_emb = nn.Embedding(
            self._n_items + 1, self._hidden_dim, padding_idx=self._pad_token
        )
        self.pos_emb = nn.Embedding(self._max_len, self._hidden_dim)
        self.user_emb = nn.Embedding(self._n_users, self._user_dim)
        self.emb_dropout = nn.Dropout(self._p)

        if user_handling == "mult":
            self.user_linear = nn.Linear(self._hidden_dim, self._hidden_dim)

        if user_handling == "tucker":
            self.core = nn.Parameter(
                torch.rand(self._user_dim, self._hidden_dim, self._hidden_dim)
            )  # reinitialized in self._initialize

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

    def forward(self, seq, user, **batch):
        """
        Args:
            seq (torch.tensor): tensor containing input sequences. Shape: (B, L)
            user (torch.tensor): tensor containing users
                for corresponding sequences. Shape: (B)
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

        user_emb = self.user_emb(user)

        if self._user_handling == "add":
            hidden_states += user_emb.unsqueeze(1)
            logits = hidden_states @ self.item_emb.weight.T  # (B, L, n_item)
        elif self._user_handling == "mult":
            user_emb = user_emb.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            user_emb = self.emb_dropout(user_emb)
            user_emb = self.user_linear(user_emb)
            hidden_states *= user_emb
            logits = hidden_states @ self.item_emb.weight.T  # (B, L, n_item)
        elif self._user_handling == "tucker":
            user_emb = self.emb_dropout(user_emb)
            # we use opt_einsum to avoid forming B x L x N_items x D_u x D x D tensor
            logits = contract(
                "bi, blj, nk, ijk -> bln",
                user_emb,  # (B, D_u)
                hidden_states,  # (B, L, D)
                self.item_emb.weight,  # (N_items, D)
                self.core,  # (D_u, D, D)
                backend="torch",
            )
        else:
            raise NotImplementedError

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
