import torch

from recbole.model.dif_layers import DIFMultiHeadAttention
from recbole.model.layers import VanillaAttention


def _causal_attention_mask(batch_size, seq_len):
    valid = torch.ones(batch_size, seq_len, dtype=torch.long)
    extended = valid.unsqueeze(1).unsqueeze(2)
    subsequent = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)
    subsequent = (subsequent == 0).unsqueeze(1).long()
    extended = extended * subsequent
    return (1.0 - extended.float()) * -10000.0


def test_dif_attention_uses_position_only_in_score_path():
    torch.manual_seed(5)
    attn = DIFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        attribute_hidden_size=[6],
        feat_num=1,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=5,
    )
    attn.eval()

    hidden = torch.randn(2, 5, 8)
    attribute = [torch.randn(2, 5, 6)]
    position_a = torch.zeros(2, 5, 8)
    position_b = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(2, 5)

    out_a, _ = attn(hidden, attribute, position_a, mask)
    out_b, _ = attn(hidden, attribute, position_b, mask)

    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-7
    item_value = attn.transpose_for_scores(attn.value(hidden))
    assert item_value.shape == (2, 2, 5, 4)


def test_dif_gate_is_score_level_fusion_and_registered_in_init():
    attn = DIFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        attribute_hidden_size=[6, 4],
        feat_num=2,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        fusion_type="gate",
        max_len=5,
    )
    assert isinstance(attn.fusion_layer, VanillaAttention)
    before = id(attn.fusion_layer)

    hidden = torch.randn(2, 5, 8)
    attributes = [torch.randn(2, 5, 6), torch.randn(2, 5, 4)]
    position = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(2, 5)

    item_scores, pos_scores, attribute_scores, _ = attn._score_components(
        hidden, attributes, position
    )
    stacked = torch.cat(
        [attribute_scores, item_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2
    )
    assert stacked.shape == (2, 2, 5, 4, 5)

    output, probs = attn(hidden, attributes, position, mask)
    assert id(attn.fusion_layer) == before
    assert output.shape == (2, 5, 8)
    assert probs.shape == (2, 2, 5, 5)


def test_dif_concat_fusion_layer_is_registered_in_init():
    attn = DIFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        attribute_hidden_size=[6, 4],
        feat_num=2,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        fusion_type="concat",
        max_len=5,
    )
    assert "fusion_layer.weight" in dict(attn.named_parameters())
    before = id(attn.fusion_layer)
    hidden = torch.randn(2, 5, 8)
    attributes = [torch.randn(2, 5, 6), torch.randn(2, 5, 4)]
    position = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(2, 5)
    output, _ = attn(hidden, attributes, position, mask)
    assert id(attn.fusion_layer) == before
    assert output.shape == (2, 5, 8)


def test_dif_rejects_unsupported_fusion_and_bad_attribute_size():
    try:
        DIFMultiHeadAttention(
            n_heads=2,
            hidden_size=8,
            attribute_hidden_size=[6],
            feat_num=1,
            hidden_dropout_prob=0.0,
            attn_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            fusion_type="median",
            max_len=5,
        )
    except ValueError as exc:
        assert "fusion_type" in str(exc)
    else:
        raise AssertionError("unsupported DIF fusion should fail")

    try:
        DIFMultiHeadAttention(
            n_heads=3,
            hidden_size=9,
            attribute_hidden_size=[8],
            feat_num=1,
            hidden_dropout_prob=0.0,
            attn_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            fusion_type="sum",
            max_len=5,
        )
    except ValueError as exc:
        assert "attribute_hidden_size" in str(exc)
    else:
        raise AssertionError("attribute_hidden_size must divide n_heads")
