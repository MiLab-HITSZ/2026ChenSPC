import math

import torch
import torch.nn.functional as F

from official_token_baselines import (
    _generate_with_branches,
    _pai_attention_forward,
    generate_native,
    mfcd_adjust_logits,
    pai_adjust_logits,
    vcd_adjust_logits,
)


def test_vcd_matches_released_full_vocab_equation_and_mask():
    native = torch.tensor([[3.0, 2.0, 0.0]])
    distorted = torch.tensor([[1.0, 2.5, -1.0]])
    actual = vcd_adjust_logits(native, distorted, alpha=1.0, beta=0.1)
    expected = 2.0 * native - distorted
    cutoff = math.log(0.1) + native.max(dim=-1, keepdim=True).values
    expected = expected.masked_fill(native < cutoff, -torch.inf)
    torch.testing.assert_close(actual, expected)


def test_mfcd_matches_released_full_vocab_equation_and_mask():
    native = torch.tensor([[4.0, 2.0, 0.0]])
    high = torch.tensor([[1.0, 2.0, 3.0]])
    low = torch.tensor([[2.0, 1.0, 0.0]])
    actual = mfcd_adjust_logits(
        native,
        high,
        low,
        high_alpha=1.0,
        low_alpha=1.0,
        beta=0.3,
    )
    expected = 3.0 * native - low - high
    probabilities = F.softmax(native, dim=-1)
    mask = probabilities < 0.3 * probabilities.max(dim=-1, keepdim=True).values
    expected = expected.masked_fill(mask, torch.finfo(expected.dtype).min)
    torch.testing.assert_close(actual, expected)


def test_pai_matches_released_cfg_equation_and_mask():
    multimodal = torch.tensor([[3.0, 2.0, 0.0]])
    text_only = torch.tensor([[1.0, 2.5, -1.0]])
    actual = pai_adjust_logits(
        multimodal,
        text_only,
        guidance_scale=1.1,
        plausibility_beta=0.1,
    )
    image_logp = F.log_softmax(multimodal, dim=-1)
    text_logp = F.log_softmax(text_only, dim=-1)
    expected = 1.1 * (image_logp - text_logp) + text_logp
    cutoff = math.log(0.1) + image_logp.max(dim=-1, keepdim=True).values
    expected = expected.masked_fill(image_logp < cutoff, -torch.inf)
    torch.testing.assert_close(actual, expected)


def test_pai_sdpa_port_matches_released_eager_attention_edit():
    class Attention:
        num_key_value_groups = 2
        is_causal = True
        training = False
        _pai_use_attention = True
        _pai_cfg_branch = False
        _pai_image_start = 1
        _pai_image_end = 3
        _pai_alpha = 0.2

    torch.manual_seed(3)
    query = torch.randn(1, 4, 5, 3)
    key = torch.randn(1, 2, 5, 3)
    value = torch.randn(1, 2, 5, 3)
    mask = torch.full((1, 1, 5, 5), -torch.inf)
    mask = torch.triu(mask, diagonal=1)
    actual, _ = _pai_attention_forward(
        Attention(),
        query,
        key,
        value,
        mask,
        scaling=3**-0.5,
    )

    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    weights = torch.matmul(query, repeated_key.transpose(2, 3)) * (3**-0.5)
    weights = weights + mask
    image_weights = weights[:, :, -1, 1:3]
    weights[:, :, -1, 1:3] = image_weights.abs() * 0.2 + image_weights
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    expected = torch.matmul(weights, repeated_value).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_token_generation_uses_full_vocab_and_stops_on_answer(monkeypatch):
    class Tokenizer:
        eos_token_id = 3

        @staticmethod
        def decode(values, **kwargs):
            return "".join(str(value) for value in values)

    class Processor:
        tokenizer = Tokenizer()

    def fake_forward(model, input_ids, kwargs, first_iteration):
        logits = torch.tensor([[0.0, 1.0, 5.0, -1.0]])
        return logits, kwargs

    monkeypatch.setattr(
        "official_token_baselines._forward_branch",
        fake_forward,
    )
    generation = _generate_with_branches(
        object(),
        Processor(),
        {
            "native": {"input_ids": torch.tensor([[7]])},
            "other": {"input_ids": torch.tensor([[8]])},
        },
        lambda logits: logits["native"],
        method="test",
        max_new_tokens=4,
        stop_token_ids=[2],
    )
    assert generation.token_ids == [2]
    assert generation.text == "2"
    assert generation.diagnostics["stopped_on_answer"] is True

    native = generate_native(
        object(),
        Processor(),
        {"input_ids": torch.tensor([[7]])},
        max_new_tokens=4,
        stop_token_ids=[2],
    )
    assert native.token_ids == [2]
    assert native.diagnostics["method"] == "native"


def test_later_answer_token_does_not_truncate_explanatory_text(monkeypatch):
    class Tokenizer:
        eos_token_id = 3

        @staticmethod
        def decode(values, **kwargs):
            return "".join(str(value) for value in values)

    class Processor:
        tokenizer = Tokenizer()

    def fake_forward(model, input_ids, kwargs, first_iteration):
        step = int(input_ids.shape[-1]) - 1
        winner = (1, 2, 3)[min(step, 2)]
        logits = torch.full((1, 4), -5.0)
        logits[0, winner] = 5.0
        return logits, kwargs

    monkeypatch.setattr(
        "official_token_baselines._forward_branch",
        fake_forward,
    )
    generation = generate_native(
        object(),
        Processor(),
        {"input_ids": torch.tensor([[7]])},
        max_new_tokens=4,
        stop_token_ids=[2],
    )
    assert generation.token_ids == [1, 2, 3]
    assert generation.diagnostics["stopped_on_answer"] is False
