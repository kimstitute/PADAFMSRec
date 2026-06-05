# -*- coding: utf-8 -*-
r"""
PADAFRec attention layers.

These layers implement the v1 PADAF contract from the local research plan:

    P_f = S_id + beta_f * S_f
    S_final = sum_f alpha_f * P_f
    H = softmax(S_final + causal_mask) * V_id

Side information is used only through Q/K attention scores.  The value path is
always the ID/collaborative representation.
"""

import math

import torch
from torch import nn

from recbole.model.layers import FeedForward


class PADAFMultiHeadAttention(nn.Module):
    """ID-anchored pairwise adaptive decoupled multi-head attention.

    Args:
        n_heads (int): number of attention heads.
        hidden_size (int): ID and projected feature hidden size.
        feat_num (int): number of side features.
        hidden_dropout_prob (float): output dropout probability.
        attn_dropout_prob (float): attention probability dropout.
        layer_norm_eps (float): layer norm epsilon.
        beta_init (float): initial value for feature-wise beta scalars.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        feat_num,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        beta_init=0.01,
        beta=None,
    ):
        super(PADAFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.feat_num = feat_num

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.feature_query = nn.ModuleList(
            [nn.Linear(hidden_size, self.all_head_size) for _ in range(feat_num)]
        )
        self.feature_key = nn.ModuleList(
            [nn.Linear(hidden_size, self.all_head_size) for _ in range(feat_num)]
        )

        self.alpha_layer = nn.Linear(hidden_size, feat_num) if feat_num > 0 else None
        if beta is None:
            beta = nn.Parameter(torch.full((feat_num,), float(beta_init)))
        self.beta = beta

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _masked_mean_pool(self, hidden_states, item_seq_mask):
        if item_seq_mask is None:
            return hidden_states.mean(dim=1)

        mask = item_seq_mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _context_from_scores(self, attention_scores, attention_mask, value_layer, residual):
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + residual)
        return hidden_states, attention_probs

    def forward(
        self,
        input_tensor,
        feature_tensors,
        attention_mask,
        item_seq_mask=None,
        return_pair_context=False,
        return_attention_stats=False,
    ):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        item_attention_scores = torch.matmul(
            item_query_layer, item_key_layer.transpose(-1, -2)
        )

        pair_scores = []
        for i, (feature_tensor, feature_query, feature_key) in enumerate(
            zip(feature_tensors, self.feature_query, self.feature_key)
        ):
            feature_query_layer = self.transpose_for_scores(feature_query(feature_tensor))
            feature_key_layer = self.transpose_for_scores(feature_key(feature_tensor))
            feature_scores = torch.matmul(
                feature_query_layer, feature_key_layer.transpose(-1, -2)
            )
            pair_scores.append(item_attention_scores + self.beta[i] * feature_scores)

        if pair_scores:
            # [B, heads, L, L, F]
            pair_score_stack = torch.stack(pair_scores, dim=-1)
            pooled_context = self._masked_mean_pool(input_tensor, item_seq_mask)
            alpha = torch.softmax(self.alpha_layer(pooled_context), dim=-1)  # [B, F]
            attention_scores = (
                pair_score_stack * alpha[:, None, None, None, :]
            ).sum(dim=-1)
        else:
            pair_score_stack = None
            alpha = input_tensor.new_zeros((input_tensor.size(0), 0))
            attention_scores = item_attention_scores

        hidden_states, attention_probs = self._context_from_scores(
            attention_scores, attention_mask, item_value_layer, input_tensor
        )

        pair_contexts = None
        if return_pair_context and pair_scores:
            pair_contexts = []
            for score in pair_scores:
                pair_context, _ = self._context_from_scores(
                    score, attention_mask, item_value_layer, input_tensor
                )
                pair_contexts.append(pair_context)

        stats = None
        if return_attention_stats:
            stats = {
                "alpha": alpha,
                "beta": self.beta,
                "attention_probs": attention_probs,
                "pair_scores": pair_score_stack,
            }

        return hidden_states, pair_contexts, stats


class PADAFTransformerLayer(nn.Module):
    """One PADAF transformer layer."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        feat_num,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        beta_init=0.01,
        beta=None,
    ):
        super(PADAFTransformerLayer, self).__init__()
        self.multi_head_attention = PADAFMultiHeadAttention(
            n_heads,
            hidden_size,
            feat_num,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            beta_init=beta_init,
            beta=beta,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        feature_hidden_states,
        attention_mask,
        item_seq_mask=None,
        return_pair_context=False,
        return_attention_stats=False,
    ):
        attention_output, pair_contexts, stats = self.multi_head_attention(
            hidden_states,
            feature_hidden_states,
            attention_mask,
            item_seq_mask=item_seq_mask,
            return_pair_context=return_pair_context,
            return_attention_stats=return_attention_stats,
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, pair_contexts, stats


class PADAFTransformerEncoder(nn.Module):
    """Stack of PADAF transformer layers."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        beta_init=0.01,
    ):
        super(PADAFTransformerEncoder, self).__init__()
        shared_beta = nn.Parameter(torch.full((feat_num,), float(beta_init)))
        self.layer = nn.ModuleList(
            [
                PADAFTransformerLayer(
                    n_heads,
                    hidden_size,
                    feat_num,
                    inner_size,
                    hidden_dropout_prob,
                    attn_dropout_prob,
                    hidden_act,
                    layer_norm_eps,
                    beta_init=beta_init,
                    beta=shared_beta,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        hidden_states,
        feature_hidden_states,
        attention_mask,
        item_seq_mask=None,
        output_all_encoded_layers=True,
        return_pair_context=False,
        return_attention_stats=False,
    ):
        all_encoder_layers = []
        last_pair_contexts = None
        all_stats = []
        for layer_module in self.layer:
            hidden_states, pair_contexts, stats = layer_module(
                hidden_states,
                feature_hidden_states,
                attention_mask,
                item_seq_mask=item_seq_mask,
                return_pair_context=return_pair_context,
                return_attention_stats=return_attention_stats,
            )
            last_pair_contexts = pair_contexts
            if stats is not None:
                all_stats.append(stats)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers, last_pair_contexts, all_stats
