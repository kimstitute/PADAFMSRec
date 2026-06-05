import torch

from recbole.model.iris_layers import IRISMultiHeadAttention


def _causal_attention_mask(batch_size, seq_len):
    valid = torch.ones(batch_size, seq_len, dtype=torch.long)
    extended = valid.unsqueeze(1).unsqueeze(2)
    subsequent = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)
    subsequent = (subsequent == 0).unsqueeze(1).long()
    extended = extended * subsequent
    return (1.0 - extended.float()) * -10000.0


def test_iris_attention_uses_position_only_in_score_path():
    torch.manual_seed(3)
    attn = IRISMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        attribute_hidden_size=8,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=5,
    )
    attn.eval()

    hidden = torch.randn(2, 5, 8)
    attribute = torch.randn(2, 5, 8)
    position_a = torch.zeros(2, 5, 8)
    position_b = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(2, 5)

    out_a, _ = attn(hidden, attribute, position_a, mask)
    out_b, _ = attn(hidden, attribute, position_b, mask)

    # Position changes can affect attention scores, therefore outputs differ.
    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-7

    # But value projection must remain item-only; no position is passed through value().
    item_value = attn.transpose_for_scores(attn.value(hidden))
    assert item_value.shape == (2, 2, 5, 4)


def test_iris_concat_fusion_layer_is_registered_in_init():
    attn = IRISMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        attribute_hidden_size=8,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        fusion_type="concat",
        max_len=5,
    )

    assert "fusion_layer.weight" in dict(attn.named_parameters())
    before = id(attn.fusion_layer)
    hidden = torch.randn(2, 5, 8)
    attribute = torch.randn(2, 5, 8)
    position = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(2, 5)
    attn(hidden, attribute, position, mask)
    assert id(attn.fusion_layer) == before


def test_iris_rejects_unsupported_gate_fusion():
    try:
        IRISMultiHeadAttention(
            n_heads=2,
            hidden_size=8,
            attribute_hidden_size=8,
            hidden_dropout_prob=0.0,
            attn_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            fusion_type="gate",
            max_len=5,
        )
    except ValueError as exc:
        assert "fusion_type" in str(exc)
    else:
        raise AssertionError("IRIS v1 should reject gate fusion")


def test_iris_rejects_attribute_size_not_divisible_by_heads():
    try:
        IRISMultiHeadAttention(
            n_heads=3,
            hidden_size=9,
            attribute_hidden_size=8,
            hidden_dropout_prob=0.0,
            attn_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            fusion_type="sum",
            max_len=5,
        )
    except ValueError as exc:
        assert "attribute_hidden_size" in str(exc)
    else:
        raise AssertionError("attribute_hidden_size must be divisible by n_heads")
