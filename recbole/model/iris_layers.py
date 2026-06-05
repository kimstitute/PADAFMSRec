# -*- coding: utf-8 -*-
r"""
IRIS attention layers.

This module ports the IRIS attention contract into the current RecBole codebase
without importing the whole upstream IRIS RecBole fork.  IRIS keeps positional
signals out of the input value representation and uses them only in attention
score fusion together with the ID and one side-information branch.
"""

import copy
import math

import torch
from torch import nn

from recbole.model.layers import FeedForward


class IRISMultiHeadAttention(nn.Module):
    """IRIS multi-head attention for one independent side-information branch."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        attribute_hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        fusion_type="sum",
        max_len=None,
    ):
        super(IRISMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "hidden_size ({}) must be divisible by n_heads ({})".format(
                    hidden_size, n_heads
                )
            )
        if attribute_hidden_size % n_heads != 0:
            raise ValueError(
                "attribute_hidden_size ({}) must be divisible by n_heads ({})".format(
                    attribute_hidden_size, n_heads
                )
            )
        if fusion_type not in ["sum", "concat"]:
            raise ValueError("IRIS v1 supports fusion_type in ['sum', 'concat'] only")
        if fusion_type == "concat" and max_len is None:
            raise ValueError("max_len is required when fusion_type='concat'")

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.attribute_attention_head_size = int(attribute_hidden_size / n_heads)
        self.attribute_all_head_size = (
            self.num_attention_heads * self.attribute_attention_head_size
        )
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Position is intentionally score-only, matching original IRIS.
        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.attribute_query = nn.Linear(
            attribute_hidden_size, self.attribute_all_head_size
        )
        self.attribute_key = nn.Linear(
            attribute_hidden_size, self.attribute_all_head_size
        )

        if self.fusion_type == "concat":
            self.fusion_layer = nn.Linear(self.max_len * 3, self.max_len)
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

    def forward(self, input_tensor, attribute_tensor, position_embedding, attention_mask):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        attribute_query_layer = self.transpose_for_scores(
            self.attribute_query(attribute_tensor), self.attribute_attention_head_size
        )
        attribute_key_layer = self.transpose_for_scores(
            self.attribute_key(attribute_tensor), self.attribute_attention_head_size
        )

        item_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        attribute_scores = torch.matmul(
            attribute_query_layer, attribute_key_layer.transpose(-1, -2)
        )

        if self.fusion_type == "sum":
            attention_scores = item_scores + pos_scores + attribute_scores
        else:
            attention_scores = torch.cat(
                [attribute_scores, item_scores, pos_scores], dim=-1
            )
            attention_scores = self.fusion_layer(attention_scores)

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


class IRISTransformerLayer(nn.Module):
    """One IRIS transformer layer for one feature branch."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        attribute_hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        fusion_type,
        max_len,
    ):
        super(IRISTransformerLayer, self).__init__()
        self.multi_head_attention = IRISMultiHeadAttention(
            n_heads,
            hidden_size,
            attribute_hidden_size,
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


class IRISTransformerEncoder(nn.Module):
    """Stack of IRIS transformer layers for one feature branch."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        attribute_hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=None,
    ):
        super(IRISTransformerEncoder, self).__init__()
        layer = IRISTransformerLayer(
            n_heads,
            hidden_size,
            attribute_hidden_size,
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
