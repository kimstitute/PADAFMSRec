# -*- coding: utf-8 -*-
r"""
IRIS
################################################

RecBole-compatible port of Independent Representation of Side Information for
Sequential Recommendation.  This port keeps original IRIS's position contract:
position embeddings are used only in attention score fusion and are not added to
input item embeddings.

The `IRIS-full(ported variant)` configuration can consume the PADAF dense cache
contract: row 0 is padding, row index equals RecBole-remapped item id, and dense
feature rows may come from BERT/ResNet `.npy` matrices.
"""

import os

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.iris_layers import IRISTransformerEncoder
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.model.sequential_recommender.padafrec import (
    PrecomputedItemFeatureEmbedding,
)


class IRISDenseFeatureEmbedding(PrecomputedItemFeatureEmbedding):
    """Precomputed dense item feature lookup projected to IRIS attribute size."""

    def __init__(self, feature_path, attribute_hidden_size, device, expected_rows=None):
        super(IRISDenseFeatureEmbedding, self).__init__(
            feature_path, attribute_hidden_size, device
        )
        if expected_rows is not None and self.feature_matrix.size(0) != expected_rows:
            raise ValueError(
                "Dense feature matrix row count must equal dataset.item_num "
                "because row index is the RecBole remapped item id. got {} rows, "
                "expected {}.".format(self.feature_matrix.size(0), expected_rows)
            )


class IRIS(SequentialRecommender):
    """IRIS sequential recommender with optional PADAF dense cache branches."""

    def __init__(self, config, dataset):
        super(IRIS, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.attribute_hidden_size = list(config["attribute_hidden_size"] or [])
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]
        self.fusion_type = config["fusion_type"]
        self.combine_type = config["combine_type"]

        self.selected_features = list(config["selected_features"] or [])
        self.dense_features = list(config["dense_features"] or [])
        self.structured_features = list(config["structured_features"] or [])
        if not self.structured_features:
            self.structured_features = [
                feature
                for feature in self.selected_features
                if feature not in self.dense_features
            ]
        self.feature_names = self.structured_features + self.dense_features
        self.num_feature_field = len(self.feature_names)

        self._validate_config(config)
        self.attribute_size = self.attribute_hidden_size[0]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer_list = nn.ModuleList(
            [
                FeatureSeqEmbLayer(
                    dataset,
                    self.attribute_size,
                    [feature],
                    self.pooling_mode,
                    self.device,
                )
                for feature in self.structured_features
            ]
        )

        self.dense_feature_layers = nn.ModuleDict()
        for feature in self.dense_features:
            path = self._get_dense_feature_path(config, feature)
            self.dense_feature_layers[feature] = IRISDenseFeatureEmbedding(
                path,
                self.attribute_size,
                self.device,
                expected_rows=dataset.item_num,
            )

        self.trm_encoder = IRISTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            attribute_hidden_size=self.attribute_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.fusion_type,
            max_len=self.max_seq_length,
        )

        if self.combine_type == "concat":
            self.combine_layer = nn.Linear(
                self.hidden_size * self.num_feature_field, self.hidden_size
            )
        else:
            self.combine_layer = None

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer_list"]

    def _validate_config(self, config):
        if not self.selected_features:
            raise ValueError("IRIS requires at least one selected feature.")
        if self.selected_features != self.feature_names:
            raise ValueError(
                "selected_features must equal structured_features + dense_features "
                "to keep IRIS branch order explicit. got selected_features={}, "
                "structured_features={}, dense_features={}".format(
                    self.selected_features,
                    self.structured_features,
                    self.dense_features,
                )
            )
        if len(self.attribute_hidden_size) != self.num_feature_field:
            raise ValueError(
                "attribute_hidden_size length ({}) must equal number of selected "
                "features ({}).".format(
                    len(self.attribute_hidden_size), self.num_feature_field
                )
            )
        if len(set(self.attribute_hidden_size)) != 1:
            raise ValueError(
                "IRIS v1 requires all attribute_hidden_size values to be identical "
                "because the shared encoder uses one attribute projection size."
            )
        if self.attribute_hidden_size[0] % self.n_heads != 0:
            raise ValueError(
                "attribute_hidden_size ({}) must be divisible by n_heads ({}).".format(
                    self.attribute_hidden_size[0], self.n_heads
                )
            )
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                "hidden_size ({}) must be divisible by n_heads ({}).".format(
                    self.hidden_size, self.n_heads
                )
            )
        if self.fusion_type not in ["sum", "concat"]:
            raise ValueError("IRIS v1 supports fusion_type in ['sum', 'concat'] only")
        if self.combine_type not in ["sum", "mean", "concat"]:
            raise ValueError(
                "IRIS v1 supports combine_type in ['sum', 'mean', 'concat'] only"
            )
        dense_feature_paths = config["dense_feature_paths"] or {}
        for feature in self.dense_features:
            explicit_key = "{}_feature_path".format(feature)
            if feature not in dense_feature_paths and (
                explicit_key not in config or config[explicit_key] is None
            ):
                raise ValueError(
                    "Missing dense feature path for '{}'. Set dense_feature_paths['{}'] "
                    "or {}.".format(feature, feature, explicit_key)
                )

    def _get_dense_feature_path(self, config, feature):
        dense_feature_paths = config["dense_feature_paths"] or {}
        if feature in dense_feature_paths:
            return dense_feature_paths[feature]
        explicit_key = "{}_feature_path".format(feature)
        if explicit_key in config and config[explicit_key] is not None:
            return config[explicit_key]
        raise ValueError(
            "Missing dense feature path for '{}'. Set dense_feature_paths['{}'] "
            "or {}.".format(feature, feature, explicit_key)
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _feature_output_to_sequence(self, sparse_embedding, dense_embedding):
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        if not feature_table:
            raise ValueError("Configured feature produced no embedding output.")
        feature_emb = torch.cat(feature_table, dim=-2)
        if feature_emb.size(-2) == 1:
            return feature_emb.squeeze(-2)
        return feature_emb.mean(dim=-2)

    def _get_feature_tensors(self, item_seq):
        feature_tensors = []
        for feature_embed_layer in self.feature_embed_layer_list:
            sparse_embedding, dense_embedding = feature_embed_layer(None, item_seq)
            feature_tensors.append(
                self._feature_output_to_sequence(
                    sparse_embedding["item"], dense_embedding["item"]
                )
            )
        for feature in self.dense_features:
            feature_tensors.append(self.dense_feature_layers[feature](item_seq))
        return feature_tensors

    def _combine_branch_outputs(self, branch_outputs):
        stacked = torch.stack(branch_outputs, dim=0)
        if self.combine_type == "sum":
            return stacked.sum(dim=0)
        if self.combine_type == "mean":
            return stacked.mean(dim=0)
        concat = torch.cat(branch_outputs, dim=-1)
        return self.combine_layer(concat)

    def forward(self, item_seq, item_seq_len, return_branch_outputs=False):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        # IRIS contract: do not add position_embedding to the input value path.
        input_emb = self.LayerNorm(item_emb)
        input_emb = self.dropout(input_emb)

        feature_tensors = self._get_feature_tensors(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)

        branch_outputs = []
        for feature_tensor in feature_tensors:
            trm_output, _ = self.trm_encoder(
                input_emb,
                feature_tensor,
                position_embedding,
                extended_attention_mask,
                output_all_encoded_layers=True,
            )
            output = trm_output[-1]
            branch_outputs.append(self.gather_indexes(output, item_seq_len - 1))

        seq_output = self._combine_branch_outputs(branch_outputs)
        if return_branch_outputs:
            return {"seq_output": seq_output, "branch_outputs": branch_outputs}
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_items_emb.transpose(0, 1))
