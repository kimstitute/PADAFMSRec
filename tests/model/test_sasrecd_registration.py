import numpy as np
from recbole.model.sequential_recommender import SASRecD
from recbole.data.interaction import Interaction
from recbole.quick_start import objective_function
from recbole.utils import get_model
import torch


def test_sasrecd_model_registration():
    assert get_model("SASRecD") is SASRecD


def _write_toy_dataset(tmp_path):
    root = tmp_path / "toy_difsr"
    root.mkdir()
    (root / "toy_difsr.inter").write_text(
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
        "u1\ti1\t1\t1\n"
        "u1\ti2\t1\t2\n"
        "u1\ti3\t1\t3\n"
        "u2\ti1\t1\t1\n"
        "u2\ti3\t1\t2\n"
        "u2\ti4\t1\t3\n"
    )
    (root / "toy_difsr.item").write_text(
        "item_id:token\tcategory:token\tbrand:token\n"
        "i1\tc1\tb1\n"
        "i2\tc1\tb2\n"
        "i3\tc2\tb1\n"
        "i4\tc2\tb2\n"
    )
    return root.parent


def _write_toy_dense_caches(tmp_path):
    text_path = tmp_path / "text_features.npy"
    image_path = tmp_path / "image_features.npy"
    text_features = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    image_features = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    text_features[0] = 0
    image_features[0] = 0
    np.save(text_path, text_features)
    np.save(image_path, image_features)
    return {"text": str(text_path), "image": str(image_path)}


def _sasrecd_config(**overrides):
    config_dict = {
        "model": "SASRecD",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "release_year"],
        },
        "selected_features": ["release_year"],
        "attribute_hidden_size": [16],
        "lamdas": [10],
        "attribute_predictor": "linear",
        "fusion_type": "gate",
        "pooling_mode": "sum",
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
        "show_progress": False,
    }
    config_dict.update(overrides)
    return config_dict


def test_sasrecd_quick_objective_canonical_structured_no_aux():
    result = objective_function(
        config_dict=_sasrecd_config(attribute_predictor="not", lamdas=[]),
        config_file_list=["tests/model/test_model.yaml"],
        saved=False,
    )
    assert isinstance(result, dict)


def test_sasrecd_quick_objective_dataset_native_structured_no_aux(tmp_path):
    data_path = _write_toy_dataset(tmp_path)
    result = objective_function(
        config_dict=_sasrecd_config(
            dataset="toy_difsr",
            data_path=str(data_path),
            selected_features=["category", "brand"],
            attribute_hidden_size=[16, 8],
            load_col={
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item": ["item_id", "category", "brand"],
            },
            attribute_predictor="not",
            lamdas=[],
            topk=[1, 2],
        ),
        saved=False,
    )
    assert isinstance(result, dict)


def test_sasrecd_accepts_heterogeneous_attribute_hidden_sizes_when_divisible(tmp_path):
    data_path = _write_toy_dataset(tmp_path)
    result = objective_function(
        config_dict=_sasrecd_config(
            dataset="toy_difsr",
            data_path=str(data_path),
            selected_features=["category", "brand"],
            attribute_hidden_size=[16, 8],
            load_col={
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item": ["item_id", "category", "brand"],
            },
            attribute_predictor="not",
            lamdas=[],
            topk=[1, 2],
        ),
        saved=False,
    )
    assert isinstance(result, dict)


def test_sasrecd_structured_aux_on_objective(tmp_path):
    data_path = _write_toy_dataset(tmp_path)
    result = objective_function(
        config_dict=_sasrecd_config(
            dataset="toy_difsr",
            data_path=str(data_path),
            selected_features=["category", "brand"],
            attribute_hidden_size=[16, 8],
            load_col={
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item": ["item_id", "category", "brand"],
            },
            attribute_predictor="linear",
            lamdas=[10, 10],
            topk=[1, 2],
        ),
        saved=False,
    )
    assert isinstance(result, dict)


def test_sasrecd_full_modality_dense_cache_objective(tmp_path):
    data_path = _write_toy_dataset(tmp_path)
    dense_feature_paths = _write_toy_dense_caches(tmp_path)
    result = objective_function(
        config_dict=_sasrecd_config(
            dataset="toy_difsr",
            data_path=str(data_path),
            selected_features=["category", "brand", "text", "image"],
            structured_features=["category", "brand"],
            dense_features=["text", "image"],
            dense_feature_paths=dense_feature_paths,
            auxiliary_features=["category", "brand"],
            attribute_hidden_size=[16, 8, 8, 8],
            load_col={
                "inter": ["user_id", "item_id", "rating", "timestamp"],
                "item": ["item_id", "category", "brand"],
            },
            attribute_predictor="linear",
            lamdas=[10, 10],
            topk=[1, 2],
        ),
        saved=False,
    )
    assert isinstance(result, dict)


def test_sasrecd_config_validation_failures_before_feature_layer_init():
    cases = [
        (_sasrecd_config(selected_features=[]), "selected feature"),
        (_sasrecd_config(attribute_hidden_size=[]), "attribute_hidden_size length"),
        (_sasrecd_config(attribute_hidden_size=[15]), "divisible"),
        (_sasrecd_config(lamdas=[]), "lamdas length"),
        (_sasrecd_config(fusion_type="median"), "fusion_type"),
        (_sasrecd_config(attribute_predictor="mlp"), "attribute_predictor"),
    ]
    for config_dict, message in cases:
        model = object.__new__(SASRecD)
        model.n_heads = config_dict["n_heads"]
        model.hidden_size = config_dict["hidden_size"]
        model.attribute_hidden_size = list(config_dict.get("attribute_hidden_size") or [])
        model.fusion_type = config_dict["fusion_type"]
        model.selected_features = list(config_dict.get("selected_features") or [])
        model.dense_features = list(config_dict.get("dense_features") or [])
        model.structured_features = list(config_dict.get("structured_features") or [])
        if not model.structured_features:
            model.structured_features = [
                feature for feature in model.selected_features if feature not in model.dense_features
            ]
        model.feature_names = model.structured_features + model.dense_features
        model.num_feature_field = len(model.feature_names)
        model.attribute_predictor = config_dict.get("attribute_predictor") or "not"
        model.auxiliary_features = list(config_dict.get("auxiliary_features") or [])
        if not model.auxiliary_features and model.attribute_predictor == "linear":
            model.auxiliary_features = list(model.structured_features)
        model.lamdas = list(config_dict.get("lamdas") or [])
        try:
            model._validate_config()
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected validation failure containing {message}")


def test_sasrecd_aux_label_missing_fails_fast():
    model = object.__new__(SASRecD)
    model.attribute_predictor = "linear"
    model.selected_features = ["release_year"]
    model.auxiliary_features = ["release_year"]
    interaction = Interaction({"item_id": torch.tensor([1])})
    seq_output = torch.zeros(1, 16)
    try:
        model._attribute_auxiliary_loss(interaction, seq_output)
    except ValueError as exc:
        assert "Missing auxiliary label field" in str(exc)
    else:
        raise AssertionError("missing aux label should fail fast")


def test_sasrecd_rejects_dense_row_count_mismatch(tmp_path):
    data_path = _write_toy_dataset(tmp_path)
    bad_text = tmp_path / "bad_text.npy"
    bad_image = tmp_path / "bad_image.npy"
    np.save(bad_text, np.zeros((4, 6), dtype=np.float32))
    np.save(bad_image, np.zeros((5, 8), dtype=np.float32))
    try:
        objective_function(
            config_dict=_sasrecd_config(
                dataset="toy_difsr",
                data_path=str(data_path),
                selected_features=["category", "brand", "text", "image"],
                structured_features=["category", "brand"],
                dense_features=["text", "image"],
                dense_feature_paths={"text": str(bad_text), "image": str(bad_image)},
                auxiliary_features=["category", "brand"],
                attribute_hidden_size=[16, 8, 8, 8],
                load_col={
                    "inter": ["user_id", "item_id", "rating", "timestamp"],
                    "item": ["item_id", "category", "brand"],
                },
                attribute_predictor="linear",
                lamdas=[10, 10],
                topk=[1, 2],
            ),
            saved=False,
        )
    except ValueError as exc:
        assert "Dense feature matrix row count" in str(exc)
    else:
        raise AssertionError("dense row count mismatch should fail")
