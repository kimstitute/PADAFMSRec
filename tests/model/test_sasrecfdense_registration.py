import numpy as np

from recbole.quick_start import objective_function
from recbole.utils import get_model
from recbole.model.sequential_recommender import SASRecFDense


def _write_toy_dataset(tmp_path):
    root = tmp_path / "toy_sasrecfdense"
    root.mkdir()
    (root / "toy_sasrecfdense.inter").write_text(
        "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
        "u1\ti1\t1\t1\n"
        "u1\ti2\t1\t2\n"
        "u1\ti3\t1\t3\n"
        "u2\ti1\t1\t1\n"
        "u2\ti3\t1\t2\n"
        "u2\ti4\t1\t3\n"
    )
    (root / "toy_sasrecfdense.item").write_text(
        "item_id:token\tcategory:token\tbrand:token\n"
        "i1\tc1\tb1\n"
        "i2\tc1\tb2\n"
        "i3\tc2\tb1\n"
        "i4\tc2\tb2\n"
    )
    return root.parent


def _write_dense_caches(tmp_path):
    text_path = tmp_path / "text_features.npy"
    image_path = tmp_path / "image_features.npy"
    text_features = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    image_features = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    text_features[0] = 0
    image_features[0] = 0
    np.save(text_path, text_features)
    np.save(image_path, image_features)
    return {"text": str(text_path), "image": str(image_path)}


def _sasrecfdense_config(tmp_path, **overrides):
    config_dict = {
        "model": "SASRecFDense",
        "dataset": "toy_sasrecfdense",
        "data_path": str(_write_toy_dataset(tmp_path)),
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
            "item": ["item_id", "category", "brand"],
        },
        "selected_features": ["category", "brand", "text", "image"],
        "structured_features": ["category", "brand"],
        "dense_features": ["text", "image"],
        "dense_feature_paths": _write_dense_caches(tmp_path),
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
        "topk": [1, 2],
        "show_progress": False,
    }
    config_dict.update(overrides)
    return config_dict


def test_sasrecfdense_model_registration():
    assert get_model("SASRecFDense") is SASRecFDense


def test_sasrecfdense_full_modality_objective(tmp_path):
    result = objective_function(config_dict=_sasrecfdense_config(tmp_path), saved=False)
    assert isinstance(result, dict)


def test_sasrecfdense_rejects_ambiguous_feature_order(tmp_path):
    config_dict = _sasrecfdense_config(
        tmp_path,
        selected_features=["text", "category", "brand", "image"],
    )
    try:
        objective_function(config_dict=config_dict, saved=False)
    except ValueError as exc:
        assert "selected_features must equal structured_features + dense_features" in str(exc)
    else:
        raise AssertionError("ambiguous SASRecFDense feature order should fail")


def test_sasrecfdense_rejects_missing_dense_feature_path(tmp_path):
    config_dict = _sasrecfdense_config(tmp_path, dense_feature_paths={})
    try:
        objective_function(config_dict=config_dict, saved=False)
    except ValueError as exc:
        assert "Missing dense feature path" in str(exc)
    else:
        raise AssertionError("missing dense feature path should fail")


def test_sasrecfdense_rejects_dense_row_count_mismatch(tmp_path):
    bad_text = tmp_path / "bad_text.npy"
    bad_image = tmp_path / "bad_image.npy"
    np.save(bad_text, np.zeros((4, 6), dtype=np.float32))
    np.save(bad_image, np.zeros((5, 8), dtype=np.float32))
    config_dict = _sasrecfdense_config(
        tmp_path,
        dense_feature_paths={"text": str(bad_text), "image": str(bad_image)},
    )
    try:
        objective_function(config_dict=config_dict, saved=False)
    except ValueError as exc:
        assert "Dense feature matrix row count" in str(exc)
    else:
        raise AssertionError("dense row count mismatch should fail")
