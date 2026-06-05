# -*- coding: utf-8 -*-
r"""
DIF-SR attention layers.

This module ports the Decoupled Information Fusion (DIF-SR) attention contract
into the local RecBole 1.2.1 codebase.  Position and side-information signals
are used only in the attention-score path.  The value path remains item-only.
"""

import copy
import math

import torch
from torch import nn

from recbole.model.layers import FeedForward, VanillaAttention


class DIFMultiHeadAttention(nn.Module):
    """Multi-head score-level DIF attention for multiple side features."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        attribute_hidden_size,
        feat_num,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        fusion_type="gate",
        max_len=None,
    ):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "hidden_size ({}) must be divisible by n_heads ({})".format(
                    hidden_size, n_heads
                )
            )
        if len(attribute_hidden_size) != feat_num:
            raise ValueError(
                "attribute_hidden_size length ({}) must equal feat_num ({})".format(
                    len(attribute_hidden_size), feat_num
                )
            )
        for size in attribute_hidden_size:
            if size % n_heads != 0:
                raise ValueError(
                    "attribute_hidden_size ({}) must be divisible by n_heads ({})".format(
                        size, n_heads
                    )
                )
        if fusion_type not in ["sum", "concat", "gate"]:
            raise ValueError("DIF-SR supports fusion_type in ['sum', 'concat', 'gate']")
        if fusion_type in ["concat", "gate"] and max_len is None:
            raise ValueError("max_len is required for DIF-SR concat/gate fusion")

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.attribute_attention_head_size = [
            int(size / n_heads) for size in attribute_hidden_size
        ]
        self.attribute_all_head_size = [
            self.num_attention_heads * size
            for size in self.attribute_attention_head_size
        ]
        self.feat_num = feat_num
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Position is score-only; it is never added to input/value states.
        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.query_layers = nn.ModuleList(
            [
                nn.Linear(attribute_hidden_size[i], self.attribute_all_head_size[i])
                for i in range(self.feat_num)
            ]
        )
        self.key_layers = nn.ModuleList(
            [
                nn.Linear(attribute_hidden_size[i], self.attribute_all_head_size[i])
                for i in range(self.feat_num)
            ]
        )

        if self.fusion_type == "concat":
            self.fusion_layer = nn.Linear(self.max_len * (2 + self.feat_num), self.max_len)
        elif self.fusion_type == "gate":
            self.fusion_layer = VanillaAttention(self.max_len, self.max_len)
        else:
            self.fusion_layer = None

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x, head_size=None):
        head_size = self.attention_head_size if head_size is None else head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _attribute_to_sequence(self, attribute_tensor):
        if attribute_tensor.dim() == 4:
            if attribute_tensor.size(-2) == 1:
                return attribute_tensor.squeeze(-2)
            return attribute_tensor.mean(dim=-2)
        return attribute_tensor

    def _score_components(self, input_tensor, attribute_table, position_embedding):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_scores = []
        for i, (attribute_query, attribute_key) in enumerate(
            zip(self.query_layers, self.key_layers)
        ):
            attribute_tensor = self._attribute_to_sequence(attribute_table[i])
            attribute_query_layer = self.transpose_for_scores(
                attribute_query(attribute_tensor), self.attribute_attention_head_size[i]
            )
            attribute_key_layer = self.transpose_for_scores(
                attribute_key(attribute_tensor), self.attribute_attention_head_size[i]
            )
            scores = torch.matmul(
                attribute_query_layer, attribute_key_layer.transpose(-1, -2)
            )
            attribute_scores.append(scores.unsqueeze(-2))

        attribute_score_table = torch.cat(attribute_scores, dim=-2)
        return item_scores, pos_scores, attribute_score_table, item_value_layer

    def _fuse_scores(self, item_scores, pos_scores, attribute_score_table):
        if self.fusion_type == "sum":
            return attribute_score_table.sum(dim=-2) + item_scores + pos_scores
        if self.fusion_type == "concat":
            table_shape = attribute_score_table.shape
            feat_num, attention_size = table_shape[-2], table_shape[-1]
            attention_scores = attribute_score_table.view(
                table_shape[:-2] + (feat_num * attention_size,)
            )
            attention_scores = torch.cat([attention_scores, item_scores, pos_scores], dim=-1)
            return self.fusion_layer(attention_scores)

        # Gate fusion is score-level fusion across branch dimension.
        attention_scores = torch.cat(
            [
                attribute_score_table,
                item_scores.unsqueeze(-2),
                pos_scores.unsqueeze(-2),
            ],
            dim=-2,
        )
        attention_scores, _ = self.fusion_layer(attention_scores)
        return attention_scores

    def forward(self, input_tensor, attribute_table, position_embedding, attention_mask):
        if len(attribute_table) != self.feat_num:
            raise ValueError(
                "attribute_table length ({}) must equal feat_num ({})".format(
                    len(attribute_table), self.feat_num
                )
            )
        item_scores, pos_scores, attribute_score_table, item_value_layer = (
            self._score_components(input_tensor, attribute_table, position_embedding)
        )
        attention_scores = self._fuse_scores(item_scores, pos_scores, attribute_score_table)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, attention_probs


class DIFTransformerLayer(nn.Module):
    """One DIF-SR transformer layer."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        attribute_hidden_size,
        feat_num,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        fusion_type,
        max_len,
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads,
            hidden_size,
            attribute_hidden_size,
            feat_num,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            fusion_type,
            max_len,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attribute_embed, position_embedding, attention_mask):
        attention_output, attention_probs = self.multi_head_attention(
            hidden_states, attribute_embed, position_embedding, attention_mask
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, attention_probs


class DIFTransformerEncoder(nn.Module):
    """Stack of DIF-SR transformer layers."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=None,
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        fusion_type="gate",
        max_len=None,
    ):
        super(DIFTransformerEncoder, self).__init__()
        attribute_hidden_size = attribute_hidden_size or [64]
        layer = DIFTransformerLayer(
            n_heads,
            hidden_size,
            attribute_hidden_size,
            feat_num,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            fusion_type,
            max_len,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        attribute_hidden_states,
        position_embedding,
        attention_mask,
        output_all_encoded_layers=True,
        return_attention_probs=False,
    ):
        all_encoder_layers = []
        all_attention_probs = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(
                hidden_states,
                attribute_hidden_states,
                position_embedding,
                attention_mask,
            )
            if return_attention_probs:
                all_attention_probs.append(attention_probs)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attention_probs
