# -*- coding: utf-8 -*-
r"""
SASRecFDense
################################################

PADAF-compatible dense-cache extension of SASRecF.  This model keeps the
SASRecF early-fusion contract: item id, structured side features, and
precomputed dense item features are concatenated before a linear projection and
then passed into the standard SASRec Transformer encoder.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer, TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.model.sequential_recommender.padafrec import PrecomputedItemFeatureEmbedding


class SASRecFDense(SequentialRecommender):
    """SASRecF-style early fusion over structured and precomputed dense features."""

    def __init__(self, config, dataset):
        super(SASRecFDense, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]

        self.selected_features = list(config["selected_features"] or [])
        self.dense_features = list(config["dense_features"] or [])
        self.structured_features = list(config["structured_features"] or [])
        if not self.structured_features:
            self.structured_features = [
                feature for feature in self.selected_features if feature not in self.dense_features
            ]
        self.feature_names = self.structured_features + self.dense_features
        self.num_feature_field = len(self.feature_names)

        self._validate_config(config)

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer_list = nn.ModuleList(
            [
                FeatureSeqEmbLayer(
                    dataset,
                    self.hidden_size,
                    [feature],
                    self.pooling_mode,
                    self.device,
                )
                for feature in self.structured_features
            ]
        )

        self.dense_feature_layers = nn.ModuleDict()
        for feature in self.dense_features:
            dense_layer = PrecomputedItemFeatureEmbedding(
                self._get_dense_feature_path(config, feature), self.hidden_size, self.device
            )
            if dense_layer.feature_matrix.size(0) != dataset.item_num:
                raise ValueError(
                    "Dense feature matrix row count for '{}' must equal dataset.item_num. "
                    "got {} rows, expected {}.".format(
                        feature, dense_layer.feature_matrix.size(0), dataset.item_num
                    )
                )
            self.dense_feature_layers[feature] = dense_layer

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.concat_layer = nn.Linear(
            self.hidden_size * (1 + self.num_feature_field), self.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer_list", "dense_feature_layers"]

    def _validate_config(self, config):
        if not self.selected_features:
            raise ValueError("SASRecFDense requires at least one selected feature.")
        if self.selected_features != self.feature_names:
            raise ValueError(
                "selected_features must equal structured_features + dense_features "
                "to keep SASRecFDense early-fusion order explicit. got "
                "selected_features={}, structured_features={}, dense_features={}".format(
                    self.selected_features,
                    self.structured_features,
                    self.dense_features,
                )
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("SASRecFDense feature names must be unique.")
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

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        feature_tensors = self._get_feature_tensors(item_seq)
        feature_emb = torch.cat(feature_tensors, dim=-1)
        input_concat = torch.cat((item_emb, feature_emb), dim=-1)

        input_emb = self.concat_layer(input_concat)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        return self.gather_indexes(output, item_seq_len - 1)

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
