from recbole.model.sequential_recommender import PADAFRec
from recbole.model.sequential_recommender.padafrec import PrecomputedItemFeatureEmbedding
from recbole.quick_start import objective_function
from recbole.utils import get_model
import numpy as np
import torch


def test_padafrec_model_registration():
    assert get_model("PADAFRec") is PADAFRec


def test_padafrec_quick_objective_with_structured_feature():
    config_dict = {
        "model": "PADAFRec",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year"],
        "structured_features": ["release_year"],
        "dense_features": [],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "train_neg_sample_args": None,
        "epochs": 1,
        "eval_step": 1,
        "stopping_step": 1,
        "use_category_aux": False,
        "use_brand_aux": False,
        "show_progress": False,
    }
    result = objective_function(
        config_dict=config_dict,
        config_file_list=["tests/model/test_model.yaml"],
        saved=False,
    )
    assert isinstance(result, dict)


def test_padafrec_rejects_ambiguous_selected_feature_order():
    config_dict = {
        "model": "PADAFRec",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["dense_text", "release_year"],
        "structured_features": ["release_year"],
        "dense_features": ["dense_text"],
        "dense_feature_paths": {"dense_text": "missing.npy"},
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "train_neg_sample_args": None,
        "epochs": 1,
        "use_category_aux": False,
        "use_brand_aux": False,
        "show_progress": False,
    }

    try:
        objective_function(
            config_dict=config_dict,
            config_file_list=["tests/model/test_model.yaml"],
            saved=False,
        )
    except ValueError as exc:
        assert "selected_features must equal structured_features + dense_features" in str(exc)
    else:
        raise AssertionError("ambiguous PADAF feature order should be rejected")


def test_precomputed_item_feature_embedding_uses_padding_and_missing_policy(tmp_path):
    feature_path = tmp_path / "features.npy"
    matrix = np.array(
        [
            [5.0, 5.0, 5.0],  # padding row should be forced to zero
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],  # real missing item row
        ],
        dtype=np.float32,
    )
    np.save(feature_path, matrix)

    layer = PrecomputedItemFeatureEmbedding(str(feature_path), hidden_size=4, device="cpu")
    with torch.no_grad():
        layer.missing_embedding.fill_(2.0)

    output = layer(torch.tensor([[0, 1, 2]]))

    assert output.shape == (1, 3, 4)
    assert torch.max(torch.abs(layer.feature_matrix[0])).item() == 0.0
    projected_missing = layer.projection(layer.missing_embedding)
    assert torch.max(torch.abs(output[0, 2] - projected_missing)).item() <= 1e-6
