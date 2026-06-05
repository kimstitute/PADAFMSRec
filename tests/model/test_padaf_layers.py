import torch

from recbole.model.layers import MultiHeadAttention
from recbole.model.padaf_layers import PADAFMultiHeadAttention, PADAFTransformerEncoder


def _causal_attention_mask(batch_size, seq_len):
    valid = torch.ones(batch_size, seq_len, dtype=torch.long)
    extended = valid.unsqueeze(1).unsqueeze(2)
    subsequent = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)
    subsequent = (subsequent == 0).unsqueeze(1).long()
    extended = extended * subsequent
    return (1.0 - extended.float()) * -10000.0


def test_padaf_beta_initialization_and_alpha_simplex():
    torch.manual_seed(7)
    attn = PADAFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        feat_num=4,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        beta_init=0.01,
    )
    attn.eval()

    hidden = torch.randn(2, 5, 8)
    features = [torch.randn(2, 5, 8) for _ in range(4)]
    mask = _causal_attention_mask(batch_size=2, seq_len=5)
    item_seq_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )

    _, _, stats = attn(
        hidden,
        features,
        mask,
        item_seq_mask=item_seq_mask,
        return_attention_stats=True,
    )

    assert stats["alpha"].shape == (2, 4)
    assert torch.max(torch.abs(stats["alpha"].sum(dim=-1) - 1)).item() <= 1e-6
    assert stats["beta"].shape == (4,)
    assert torch.max(torch.abs(stats["beta"] - 0.01)).item() <= 1e-6


def test_padaf_pair_score_contract():
    torch.manual_seed(11)
    attn = PADAFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        feat_num=2,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        beta_init=0.25,
    )
    attn.eval()

    hidden = torch.randn(2, 5, 8)
    features = [torch.randn(2, 5, 8) for _ in range(2)]
    mask = _causal_attention_mask(batch_size=2, seq_len=5)

    _, _, stats = attn(
        hidden,
        features,
        mask,
        item_seq_mask=torch.ones(2, 5, dtype=torch.bool),
        return_attention_stats=True,
    )

    item_q = attn.transpose_for_scores(attn.query(hidden))
    item_k = attn.transpose_for_scores(attn.key(hidden))
    item_scores = torch.matmul(item_q, item_k.transpose(-1, -2))

    for idx, feature in enumerate(features):
        feature_q = attn.transpose_for_scores(attn.feature_query[idx](feature))
        feature_k = attn.transpose_for_scores(attn.feature_key[idx](feature))
        feature_scores = torch.matmul(feature_q, feature_k.transpose(-1, -2))
        expected = item_scores + attn.beta[idx] * feature_scores
        actual = stats["pair_scores"][..., idx]
        assert torch.max(torch.abs(actual - expected)).item() <= 1e-6


def test_padaf_causal_mask_blocks_future_attention():
    torch.manual_seed(13)
    attn = PADAFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        feat_num=1,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    attn.eval()

    hidden = torch.randn(2, 5, 8)
    features = [torch.randn(2, 5, 8)]
    mask = _causal_attention_mask(batch_size=2, seq_len=5)
    _, _, stats = attn(
        hidden,
        features,
        mask,
        item_seq_mask=torch.ones(2, 5, dtype=torch.bool),
        return_attention_stats=True,
    )

    probs = stats["attention_probs"]
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert probs[..., future].max().item() <= 1e-7


def test_padaf_does_not_use_feature_values_when_scores_are_fixed():
    torch.manual_seed(17)
    attn = PADAFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        feat_num=1,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    attn.eval()

    # If feature Q/K are zero, side feature tensors cannot change scores.
    # Output should therefore stay fixed, proving there is no side-value path.
    with torch.no_grad():
        attn.feature_query[0].weight.zero_()
        attn.feature_query[0].bias.zero_()
        attn.feature_key[0].weight.zero_()
        attn.feature_key[0].bias.zero_()

    hidden = torch.randn(2, 5, 8)
    feature_a = torch.randn(2, 5, 8)
    feature_b = torch.randn(2, 5, 8) * 100
    mask = _causal_attention_mask(batch_size=2, seq_len=5)
    item_seq_mask = torch.ones(2, 5, dtype=torch.bool)

    out_a, _, _ = attn(hidden, [feature_a], mask, item_seq_mask=item_seq_mask)
    out_b, _, _ = attn(hidden, [feature_b], mask, item_seq_mask=item_seq_mask)

    assert torch.max(torch.abs(out_a - out_b)).item() <= 1e-6


def test_padaf_encoder_uses_one_shared_global_beta_parameter():
    encoder = PADAFTransformerEncoder(
        n_layers=2,
        n_heads=2,
        hidden_size=8,
        feat_num=3,
        inner_size=16,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        beta_init=0.01,
    )

    beta_params = [(name, param) for name, param in encoder.named_parameters() if name.endswith("beta")]

    assert len(beta_params) == 1
    assert beta_params[0][1].shape == (3,)
    assert (
        encoder.layer[0].multi_head_attention.beta
        is encoder.layer[1].multi_head_attention.beta
    )


def test_padaf_attention_matches_sasrec_attention_when_features_disabled():
    torch.manual_seed(19)
    sasrec_attn = MultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    padaf_attn = PADAFMultiHeadAttention(
        n_heads=2,
        hidden_size=8,
        feat_num=0,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    padaf_attn.load_state_dict(
        {
            key: value
            for key, value in sasrec_attn.state_dict().items()
            if key in padaf_attn.state_dict()
        },
        strict=False,
    )
    sasrec_attn.eval()
    padaf_attn.eval()

    hidden = torch.randn(2, 5, 8)
    mask = _causal_attention_mask(batch_size=2, seq_len=5)

    sasrec_output = sasrec_attn(hidden, mask)
    padaf_output, _, stats = padaf_attn(
        hidden,
        [],
        mask,
        return_attention_stats=True,
    )

    assert stats["alpha"].shape == (2, 0)
    assert stats["pair_scores"] is None
    assert torch.max(torch.abs(padaf_output - sasrec_output)).item() <= 1e-6
