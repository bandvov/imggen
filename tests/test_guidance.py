import pytest
import torch

from imggen.guidance import apply_caption_dropout, broadcast_null_embedding, combine_predictions


def test_p_zero_is_a_no_op():
    text_emb = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    null_emb = torch.randn(1, 8)
    null_mask = torch.ones(1, dtype=torch.bool)
    out_emb, out_mask = apply_caption_dropout(text_emb, mask, null_emb, null_mask, p=0.0)
    assert torch.equal(out_emb, text_emb) and torch.equal(out_mask, mask)


def test_p_one_replaces_every_sample_with_the_null_embedding():
    text_emb = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    null_emb = torch.randn(2, 8)
    null_mask = torch.tensor([True, True])
    out_emb, out_mask = apply_caption_dropout(text_emb, mask, null_emb, null_mask, p=1.0)
    for i in range(3):
        torch.testing.assert_close(out_emb[i, :2], null_emb)
        assert out_mask[i].tolist() == [True, True, False, False, False]


def test_dropout_selects_the_expected_fraction_given_a_seed():
    text_emb = torch.randn(200, 3, 8)
    mask = torch.ones(200, 3, dtype=torch.bool)
    null_emb = torch.randn(1, 8)
    null_mask = torch.ones(1, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    _, out_mask = apply_caption_dropout(text_emb, mask, null_emb, null_mask, p=0.3, generator=g)
    dropped = (out_mask.sum(dim=1) == 1).sum().item()  # null embedding has length 1
    assert 40 < dropped < 90  # expected 60 out of 200 at p=0.3; wide margin for a single seed


def test_output_length_is_max_of_real_and_null_lengths():
    text_emb = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    long_null_emb = torch.randn(5, 8)
    long_null_mask = torch.ones(5, dtype=torch.bool)
    out_emb, out_mask = apply_caption_dropout(
        text_emb,
        mask,
        long_null_emb,
        long_null_mask,
        p=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert out_emb.shape == (2, 5, 8) and out_mask.shape == (2, 5)


def test_dropped_and_kept_samples_are_independent_of_each_other():
    text_emb = torch.randn(4, 3, 8)
    mask = torch.ones(4, 3, dtype=torch.bool)
    null_emb = torch.randn(1, 8)
    null_mask = torch.ones(1, dtype=torch.bool)
    g = torch.Generator().manual_seed(1)
    out_emb, out_mask = apply_caption_dropout(
        text_emb, mask, null_emb, null_mask, p=0.5, generator=g
    )
    dropped = out_mask.sum(dim=1) == 1
    for i in range(4):
        if not dropped[i]:
            torch.testing.assert_close(out_emb[i, :3], text_emb[i])
            assert out_mask[i].tolist() == [True, True, True]


@pytest.mark.parametrize("scale,expected", [(1.0, "cond"), (0.0, "uncond"), (2.0, "extrapolate")])
def test_combine_predictions(scale, expected):
    cond = torch.full((2, 3), 2.0)
    uncond = torch.full((2, 3), 1.0)
    out = combine_predictions(cond, uncond, scale)
    if expected == "cond":
        torch.testing.assert_close(out, cond)
    elif expected == "uncond":
        torch.testing.assert_close(out, uncond)
    else:
        torch.testing.assert_close(out, torch.full((2, 3), 3.0))  # 1 + 2*(2-1) = 3


def test_broadcast_null_embedding_shape_and_values():
    null_emb = torch.randn(4, 8)
    null_mask = torch.tensor([True, True, False, False])
    emb, mask = broadcast_null_embedding(null_emb, null_mask, batch_size=3)
    assert emb.shape == (3, 4, 8) and mask.shape == (3, 4)
    for i in range(3):
        torch.testing.assert_close(emb[i], null_emb)
        assert torch.equal(mask[i], null_mask)
