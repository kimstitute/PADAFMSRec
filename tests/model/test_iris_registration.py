import numpy as np
import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.model.sequential_recommender import IRIS
from recbole.model.sequential_recommender.iris import IRISDenseFeatureEmbedding
from recbole.quick_start import objective_function
from recbole.utils import get_model


def test_iris_model_registration():
    assert get_model("IRIS") is IRIS


def test_iris_quick_objective_with_structured_feature():
    config_dict = {
        "model": "IRIS",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year"],
        "structured_features": ["release_year"],
        "dense_features": [],
        "attribute_hidden_size": [16],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "fusion_type": "sum",
        "combine_type": "sum",
        "train_neg_sample_args": None,
        "epochs": 1,
        "eval_step": 1,
        "stopping_step": 1,
        "show_progress": False,
    }
    result = objective_function(
        config_dict=config_dict,
        config_file_list=["tests/model/test_model.yaml"],
        saved=False,
    )
    assert isinstance(result, dict)


def _make_dataset(config_dict):
    config = Config(config_dict=config_dict, config_file_list=["tests/model/test_model.yaml"])
    return config, create_dataset(config)




def _iris_config_for_validation(**overrides):
    config_dict = {
        "model": "IRIS",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year"],
        "structured_features": ["release_year"],
        "dense_features": [],
        "attribute_hidden_size": [16],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "fusion_type": "sum",
        "combine_type": "sum",
        "train_neg_sample_args": None,
        "epochs": 1,
        "show_progress": False,
    }
    config_dict.update(overrides)
    return config_dict


def test_iris_rejects_attribute_hidden_size_length_mismatch():
    config_dict = _iris_config_for_validation(attribute_hidden_size=[])
    try:
        config, dataset = _make_dataset(config_dict)
        IRIS(config, dataset)
    except ValueError as exc:
        assert "attribute_hidden_size length" in str(exc)
    else:
        raise AssertionError("attribute_hidden_size length mismatch should fail")


def test_iris_rejects_unsupported_combine_type():
    config_dict = _iris_config_for_validation(combine_type="median")
    try:
        config, dataset = _make_dataset(config_dict)
        IRIS(config, dataset)
    except ValueError as exc:
        assert "combine_type" in str(exc)
    else:
        raise AssertionError("unsupported combine_type should fail")


def test_iris_rejects_missing_dense_feature_path():
    config_dict = _iris_config_for_validation(
        selected_features=["dense_text"],
        structured_features=[],
        dense_features=["dense_text"],
        dense_feature_paths={},
        attribute_hidden_size=[16],
    )
    try:
        config, dataset = _make_dataset(config_dict)
        IRIS(config, dataset)
    except ValueError as exc:
        assert "Missing dense feature path" in str(exc)
    else:
        raise AssertionError("missing dense feature path should fail")


def test_iris_concat_combine_layer_registered_and_full_sort_shape(tmp_path):
    dense_path = tmp_path / "dense.npy"
    config_dict = _iris_config_for_validation(
        selected_features=["dense_text"],
        structured_features=[],
        dense_features=["dense_text"],
        dense_feature_paths={"dense_text": str(dense_path)},
        attribute_hidden_size=[16],
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        combine_type="concat",
    )
    config, dataset = _make_dataset(config_dict)
    np.save(
        dense_path,
        np.random.RandomState(7).randn(dataset.item_num, 3).astype(np.float32),
    )
    model = IRIS(config, dataset)
    assert "combine_layer.weight" in dict(model.named_parameters())
    before = id(model.combine_layer)

    item_seq = torch.randint(1, dataset.item_num, (2, 5))
    item_seq_len = torch.tensor([5, 5])
    output = model.forward(item_seq, item_seq_len)
    scores = model.full_sort_predict(
        {model.ITEM_SEQ: item_seq, model.ITEM_SEQ_LEN: item_seq_len}
    )

    assert id(model.combine_layer) == before
    assert output.shape == (2, 16)
    assert scores.shape == (2, dataset.item_num)




def test_iris_concat_combine_supports_multiple_dense_branches(tmp_path):
    text_path = tmp_path / "text.npy"
    image_path = tmp_path / "image.npy"
    config_dict = _iris_config_for_validation(
        selected_features=["dense_text", "dense_image"],
        structured_features=[],
        dense_features=["dense_text", "dense_image"],
        dense_feature_paths={
            "dense_text": str(text_path),
            "dense_image": str(image_path),
        },
        attribute_hidden_size=[16, 16],
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        combine_type="concat",
    )
    config, dataset = _make_dataset(config_dict)
    np.save(
        text_path,
        np.random.RandomState(11).randn(dataset.item_num, 3).astype(np.float32),
    )
    np.save(
        image_path,
        np.random.RandomState(13).randn(dataset.item_num, 5).astype(np.float32),
    )
    model = IRIS(config, dataset)
    assert model.combine_layer.in_features == 32

    item_seq = torch.randint(1, dataset.item_num, (2, 5))
    item_seq_len = torch.tensor([5, 5])
    outputs = model.forward(item_seq, item_seq_len, return_branch_outputs=True)

    assert len(outputs["branch_outputs"]) == 2
    assert outputs["seq_output"].shape == (2, 16)


def test_iris_rejects_ambiguous_selected_feature_order():
    config_dict = {
        "model": "IRIS",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["dense_text", "release_year"],
        "structured_features": ["release_year"],
        "dense_features": ["dense_text"],
        "dense_feature_paths": {"dense_text": "missing.npy"},
        "attribute_hidden_size": [16, 16],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "fusion_type": "sum",
        "combine_type": "sum",
        "train_neg_sample_args": None,
        "epochs": 1,
        "show_progress": False,
    }
    try:
        config, dataset = _make_dataset(config_dict)
        IRIS(config, dataset)
    except ValueError as exc:
        assert "selected_features must equal structured_features + dense_features" in str(exc)
    else:
        raise AssertionError("ambiguous IRIS feature order should be rejected")


def test_iris_rejects_unequal_attribute_hidden_sizes(tmp_path):
    dense_path = tmp_path / "dense.npy"
    # Row count is irrelevant here because attribute validation must fail first.
    np.save(dense_path, np.zeros((101, 3), dtype=np.float32))
    config_dict = {
        "model": "IRIS",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year", "dense_text"],
        "structured_features": ["release_year"],
        "dense_features": ["dense_text"],
        "dense_feature_paths": {"dense_text": str(dense_path)},
        "attribute_hidden_size": [16, 8],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "fusion_type": "sum",
        "combine_type": "sum",
        "train_neg_sample_args": None,
        "epochs": 1,
        "show_progress": False,
    }
    try:
        config, dataset = _make_dataset(config_dict)
        IRIS(config, dataset)
    except ValueError as exc:
        assert "attribute_hidden_size" in str(exc)
    else:
        raise AssertionError("IRIS v1 should reject unequal attribute hidden sizes")


def test_iris_dense_embedding_validates_row_count_and_missing_policy(tmp_path):
    feature_path = tmp_path / "dense.npy"
    matrix = np.array(
        [
            [9.0, 9.0, 9.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(feature_path, matrix)

    layer = IRISDenseFeatureEmbedding(
        str(feature_path), attribute_hidden_size=4, device="cpu", expected_rows=3
    )
    with torch.no_grad():
        layer.missing_embedding.fill_(2.0)
    output = layer(torch.tensor([[0, 1, 2]]))

    assert output.shape == (1, 3, 4)
    assert torch.max(torch.abs(layer.feature_matrix[0])).item() == 0.0
    expected_missing = layer.projection(layer.missing_embedding)
    assert torch.max(torch.abs(output[0, 2] - expected_missing)).item() <= 1e-6

    try:
        IRISDenseFeatureEmbedding(
            str(feature_path), attribute_hidden_size=4, device="cpu", expected_rows=4
        )
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("dense row mismatch should fail fast")



def test_iris_full_variant_objective_with_dense_cache(tmp_path):
    dense_path = tmp_path / "dense.npy"
    # test fixture has 301 item feature rows after remapping in the quick objective path.
    np.save(dense_path, np.random.RandomState(1).randn(301, 3).astype(np.float32))
    config_dict = {
        "model": "IRIS",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year", "dense_text"],
        "structured_features": ["release_year"],
        "dense_features": ["dense_text"],
        "dense_feature_paths": {"dense_text": str(dense_path)},
        "attribute_hidden_size": [16, 16],
        "hidden_size": 16,
        "inner_size": 32,
        "n_layers": 1,
        "n_heads": 2,
        "hidden_dropout_prob": 0.0,
        "attn_dropout_prob": 0.0,
        "fusion_type": "sum",
        "combine_type": "mean",
        "train_neg_sample_args": None,
        "epochs": 1,
        "eval_step": 1,
        "stopping_step": 1,
        "show_progress": False,
    }
    result = objective_function(
        config_dict=config_dict,
        config_file_list=["tests/model/test_model.yaml"],
        saved=False,
    )
    assert isinstance(result, dict)
