# -*- coding: utf-8 -*-
r"""
PADAFRec
################################################

ID-Anchored Pairwise Adaptive Decoupled Attention Fusion for multimodal
sequential recommendation.

The v1 implementation keeps SASRec-style input positional embeddings and uses
side information only to guide attention scores.  The value path remains the
ID/collaborative representation.
"""

import os

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.model.padaf_layers import PADAFTransformerEncoder


class PrecomputedItemFeatureEmbedding(nn.Module):
    """Lookup frozen item feature rows by RecBole remapped item id.

    Row 0 is reserved for padding and is forced to zero.  Non-padding rows that
    are all zero are treated as missing and replaced by a learned missing vector
    before projection.
    """

    def __init__(self, feature_path, hidden_size, device):
        super(PrecomputedItemFeatureEmbedding, self).__init__()
        feature_matrix = self._load_feature_matrix(feature_path)
        if feature_matrix.dim() != 2:
            raise ValueError(
                "Precomputed item feature matrix must be 2-D, got shape {}".format(
                    tuple(feature_matrix.shape)
                )
            )

        feature_matrix = feature_matrix.float()
        feature_matrix[0].zero_()
        self.register_buffer("feature_matrix", feature_matrix.to(device))
        self.missing_embedding = nn.Parameter(torch.zeros(feature_matrix.size(1)))
        self.projection = nn.Linear(feature_matrix.size(1), hidden_size)

    @staticmethod
    def _load_feature_matrix(feature_path):
        if feature_path is None:
            raise ValueError("feature_path must not be None")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(feature_path)

        if feature_path.endswith(".npy"):
            return torch.from_numpy(np.load(feature_path))

        loaded = torch.load(feature_path, map_location="cpu")
        if isinstance(loaded, np.ndarray):
            return torch.from_numpy(loaded)
        if isinstance(loaded, torch.Tensor):
            return loaded
        if isinstance(loaded, dict):
            for key in ("features", "feature_matrix", "embeddings"):
                if key in loaded:
                    value = loaded[key]
                    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        raise ValueError(
            "Unsupported precomputed feature file. Expected tensor/ndarray or dict "
            "with one of: features, feature_matrix, embeddings."
        )

    def forward(self, item_seq):
        feature = self.feature_matrix[item_seq]
        non_padding = item_seq.ne(0)
        missing = feature.abs().sum(dim=-1).eq(0) & non_padding
        if missing.any():
            feature = feature.clone()
            feature[missing] = self.missing_embedding.to(feature.dtype)
        return self.projection(feature)


class PADAFRec(SequentialRecommender):
    """RecBole-compatible PADAFRec sequential recommender."""

    def __init__(self, config, dataset):
        super(PADAFRec, self).__init__(config, dataset)

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
                feature
                for feature in self.selected_features
                if feature not in self.dense_features
            ]

        self.feature_names = self.structured_features + self.dense_features
        if self.selected_features and self.selected_features != self.feature_names:
            raise ValueError(
                "selected_features must equal structured_features + dense_features "
                "to keep PADAF alpha/beta feature order explicit. got "
                "selected_features={}, structured_features={}, dense_features={}".format(
                    self.selected_features,
                    self.structured_features,
                    self.dense_features,
                )
            )
        self.num_feature_field = len(self.feature_names)
        self.beta_init = config["beta_init"] if config["beta_init"] is not None else 0.01

        self.category_aux_field = config["category_aux_field"] or "category"
        self.brand_aux_field = config["brand_aux_field"] or "brand"
        self.use_category_aux = bool(config["use_category_aux"])
        self.use_brand_aux = bool(config["use_brand_aux"])
        self.lambda_cat = config["lambda_cat"] if config["lambda_cat"] is not None else 0.1
        self.lambda_brand = (
            config["lambda_brand"] if config["lambda_brand"] is not None else 0.05
        )

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
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
            path = self._get_dense_feature_path(config, feature)
            self.dense_feature_layers[feature] = PrecomputedItemFeatureEmbedding(
                path, self.hidden_size, self.device
            )

        self.trm_encoder = PADAFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            feat_num=self.num_feature_field,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            beta_init=self.beta_init,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.attribute_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.category_predictor = self._build_aux_head(dataset, self.category_aux_field)
        self.brand_predictor = self._build_aux_head(dataset, self.brand_aux_field)

        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer_list"]

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

    def _build_aux_head(self, dataset, field):
        if field not in dataset.field2token_id:
            return None
        return nn.Linear(self.hidden_size, len(dataset.field2token_id[field]))

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
            sparse_embedding = sparse_embedding["item"]
            dense_embedding = dense_embedding["item"]
            feature_tensors.append(
                self._feature_output_to_sequence(sparse_embedding, dense_embedding)
            )

        for feature in self.dense_features:
            feature_tensors.append(self.dense_feature_layers[feature](item_seq))

        return feature_tensors

    def forward(
        self,
        item_seq,
        item_seq_len,
        return_pair_context=False,
        return_attention_stats=False,
    ):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        feature_tensors = self._get_feature_tensors(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)
        item_seq_mask = item_seq.gt(0)

        trm_output, pair_contexts, attention_stats = self.trm_encoder(
            input_emb,
            feature_tensors,
            extended_attention_mask,
            item_seq_mask=item_seq_mask,
            output_all_encoded_layers=True,
            return_pair_context=return_pair_context,
            return_attention_stats=return_attention_stats,
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)

        if not (return_pair_context or return_attention_stats):
            return seq_output

        result = {"seq_output": seq_output}
        if return_pair_context:
            result["pair_contexts"] = pair_contexts
        if return_attention_stats:
            result["attention_stats"] = attention_stats
        return result

    def _main_loss(self, seq_output, interaction):
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

    def _auxiliary_loss(self, interaction, pair_contexts, item_seq_len):
        if pair_contexts is None:
            return 0

        aux_loss = 0
        feature_to_context = {
            feature: pair_contexts[i] for i, feature in enumerate(self.feature_names)
        }

        if (
            self.use_category_aux
            and self.category_predictor is not None
            and self.category_aux_field in interaction
            and self.category_aux_field in feature_to_context
        ):
            cat_context = self.gather_indexes(
                feature_to_context[self.category_aux_field], item_seq_len - 1
            )
            cat_logits = self.category_predictor(cat_context)
            cat_labels = interaction[self.category_aux_field].long().view(-1)
            aux_loss = aux_loss + self.lambda_cat * self.attribute_loss_fct(
                cat_logits, cat_labels
            )

        if (
            self.use_brand_aux
            and self.brand_predictor is not None
            and self.brand_aux_field in interaction
            and self.brand_aux_field in feature_to_context
        ):
            brand_context = self.gather_indexes(
                feature_to_context[self.brand_aux_field], item_seq_len - 1
            )
            brand_logits = self.brand_predictor(brand_context)
            brand_labels = interaction[self.brand_aux_field].long().view(-1)
            aux_loss = aux_loss + self.lambda_brand * self.attribute_loss_fct(
                brand_logits, brand_labels
            )

        return aux_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        forward_output = self.forward(
            item_seq,
            item_seq_len,
            return_pair_context=self.use_category_aux or self.use_brand_aux,
        )
        if isinstance(forward_output, dict):
            seq_output = forward_output["seq_output"]
            pair_contexts = forward_output.get("pair_contexts")
        else:
            seq_output = forward_output
            pair_contexts = None

        loss = self._main_loss(seq_output, interaction)
        loss = loss + self._auxiliary_loss(interaction, pair_contexts, item_seq_len)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
