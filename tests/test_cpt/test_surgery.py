'Tests for the foreign-tokenizer surgery (offline, tiny random models).'

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from llm_training.cpt.surgery import (
    MODE_PROJECTION,
    MODE_REMAP,
    MODE_WHOLESALE,
    apply_foreign_tokenizer,
    freeze_transformer_blocks,
    replace_vocab_layers,
)


def _tiny_model(vocab: int, hidden: int, tie: bool = True) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config)


class TestReplaceVocabLayers:
    def test_wholesale_copy_when_dims_and_vocab_coincide(self):
        model = _tiny_model(vocab=24, hidden=16)
        donor_embed = torch.randn(24, 16, generator=torch.Generator().manual_seed(1))
        report = replace_vocab_layers(
            model,
            new_to_donor=list(range(24)),
            donor_embed=donor_embed,
            donor_head=donor_embed.clone(),
            fit_pairs=[],
            seed=42,
        )
        assert report.mode == MODE_WHOLESALE
        assert report.copied_tokens == 24 and report.fallback_tokens == 0
        assert torch.equal(model.get_input_embeddings().weight.data, donor_embed)
        assert torch.equal(model.get_output_embeddings().weight.data, donor_embed)

    def test_remap_with_fallback_rows(self):
        model = _tiny_model(vocab=30, hidden=16)
        donor_embed = torch.randn(24, 16, generator=torch.Generator().manual_seed(2))
        new_to_donor = [*range(18), None, None]
        report = replace_vocab_layers(
            model,
            new_to_donor=new_to_donor,
            donor_embed=donor_embed,
            donor_head=None,
            fit_pairs=[],
            seed=42,
        )
        assert report.mode == MODE_REMAP
        assert report.copied_tokens == 18 and report.fallback_tokens == 2
        embed = model.get_input_embeddings().weight.data
        assert torch.equal(embed[:18], donor_embed[:18])
        # Fallback rows are near the mean of the matched rows, not zeros.
        
        assert not torch.equal(embed[18], torch.zeros(16))
        head = model.get_output_embeddings().weight.data
        assert torch.equal(head[:18], donor_embed[:18])

    def test_projection_when_hidden_sizes_differ(self):
        model = _tiny_model(vocab=40, hidden=16)
        g = torch.Generator().manual_seed(3)
        donor_embed = torch.randn(30, 12, generator=g)
        donor_head = torch.randn(30, 12, generator=g)
        old_embed = model.get_input_embeddings().weight.detach()
        # Perfectly consistent pairs: donor row j maps to old row j via a
        # fixed matrix — the projection must recover near-zero residual.
        
        
        
        proj = torch.randn(12, 16, generator=g)
        old = donor_embed[:20] @ proj
        with torch.no_grad():
            old_embed[:20] = old
        fit_pairs = [(j, j) for j in range(20)]
        new_to_donor = [*range(20), None]
        report = replace_vocab_layers(
            model,
            new_to_donor=new_to_donor,
            donor_embed=donor_embed,
            donor_head=donor_head,
            fit_pairs=fit_pairs,
            seed=42,
        )
        assert report.mode == MODE_PROJECTION
        assert report.copied_tokens == 20 and report.fallback_tokens == 1
        assert report.shared_fit_tokens == 20
        assert report.embed_projection_residual is not None
        assert report.embed_projection_residual < 1e-3
        embed = model.get_input_embeddings().weight.data
        assert embed.shape == (21, 16)
        assert torch.allclose(embed[:20], donor_embed[:20] @ proj, atol=1e-4)

    def test_projection_refuses_without_fit_pairs(self):
        model = _tiny_model(vocab=10, hidden=16)
        with pytest.raises(ValueError, match="projection|проекцию"):
            replace_vocab_layers(
                model,
                new_to_donor=list(range(8)),
                donor_embed=torch.randn(8, 12),
                donor_head=None,
                fit_pairs=[(i, i) for i in range(3)],
                seed=42,
            )

    def test_config_updates_and_untied_head(self):
        model = _tiny_model(vocab=20, hidden=16, tie=True)
        donor_embed = torch.randn(16, 16)
        replace_vocab_layers(
            model,
            new_to_donor=list(range(16)),
            donor_embed=donor_embed,
            donor_head=None,
            fit_pairs=[],
            seed=42,
            special_token_ids={"eos": 5, "pad": 6, "bos": None},
        )
        assert model.config.vocab_size == 16
        assert model.config.tie_word_embeddings is False
        # The new head is a standalone parameter, not a view of the embedding.
        
        assert (
            model.get_output_embeddings().weight.data_ptr()
            != model.get_input_embeddings().weight.data_ptr()
        )
        assert model.generation_config.eos_token_id == 5
        assert model.generation_config.pad_token_id == 6


class TestFreezeAndGradients:
    def test_freeze_keeps_only_vocab_layers_trainable(self):
        model = _tiny_model(vocab=12, hidden=8)
        replace_vocab_layers(
            model,
            new_to_donor=list(range(10)),
            donor_embed=torch.randn(10, 8),
            donor_head=None,
            fit_pairs=[],
            seed=42,
        )
        frozen = freeze_transformer_blocks(model)
        assert frozen > 0
        assert model.get_input_embeddings().weight.requires_grad
        assert model.get_output_embeddings().weight.requires_grad
        layer_weight = model.model.layers[0].self_attn.q_proj.weight
        assert not layer_weight.requires_grad

    def test_gradients_flow_through_frozen_blocks(self):
        # requires_grad=False must NOT detach the graph: with frozen blocks,
        # the trainable embedding still receives gradient through them.
        # requires_grad=False
        
        model = _tiny_model(vocab=12, hidden=8)
        replace_vocab_layers(
            model,
            new_to_donor=list(range(10)),
            donor_embed=torch.randn(10, 8),
            donor_head=None,
            fit_pairs=[],
            seed=42,
        )
        freeze_transformer_blocks(model)
        input_ids = torch.randint(0, 10, (2, 6))
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        assert model.get_input_embeddings().weight.grad is not None
        assert model.get_input_embeddings().weight.grad.abs().sum() > 0
        assert model.get_output_embeddings().weight.grad is not None
        # The frozen blocks have no gradients of their own.
        
        assert model.model.layers[0].self_attn.q_proj.weight.grad is None


class _StubTokenizer:
    'Minimal tokenizer stub: string<->id vocab over token_0..token_N.'

    def __init__(self, n_tokens: int) -> None:
        self._vocab = {f"token_{i}": i for i in range(n_tokens)}
        self.eos_token_id = n_tokens - 1
        self.pad_token_id = 0
        self.bos_token_id = None

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_ids_to_tokens(self, ids):
        inv = {i: t for t, i in self._vocab.items()}
        return [inv[i] for i in ids]

    def __len__(self) -> int:
        return len(self._vocab)


class TestApplyForeignTokenizer:
    def test_wrapper_projection_path(self, monkeypatch):
        model = _tiny_model(vocab=20, hidden=16)
        model.name_or_path = "fake/model"
        donor = _tiny_model(vocab=26, hidden=8)
        donor.name_or_path = "donor/model"

        new_tok = _StubTokenizer(26)
        old_tok = _StubTokenizer(20)
        # Share 18 token strings between the vocabs → enough fit pairs.
        
        old_tok._vocab = {
            f"token_{i}": i for i in range(18)
        }
        old_tok.eos_token_id = 17

        from llm_training.cpt import surgery

        class _FakeAuto:
            @staticmethod
            def from_pretrained(model_id, *a, **kw):
                return donor if model_id == "donor/model" else old_tok

        monkeypatch.setattr(surgery, "AutoModelForCausalLM", _FakeAuto)
        monkeypatch.setattr(surgery, "AutoTokenizer", _FakeAuto)
        report = apply_foreign_tokenizer(
            model, new_tok, "donor/model", seed=42
        )
        assert report.mode == MODE_PROJECTION
        assert report.new_vocab == 26
        assert model.config.vocab_size == 26
        assert model.get_input_embeddings().weight.shape == (26, 16)
        assert report.shared_fit_tokens == 18

    def test_wrapper_wholesale_path(self, monkeypatch):
        model = _tiny_model(vocab=26, hidden=16)
        model.name_or_path = "fake/model"
        donor = _tiny_model(vocab=26, hidden=16)
        donor.name_or_path = "donor/model"
        donor_embed = donor.get_input_embeddings().weight.detach().clone()

        tok = _StubTokenizer(26)
        old_tok = _StubTokenizer(20)
        from llm_training.cpt import surgery

        class _FakeAuto:
            @staticmethod
            def from_pretrained(model_id, *a, **kw):
                return donor if model_id == "donor/model" else old_tok

        monkeypatch.setattr(surgery, "AutoModelForCausalLM", _FakeAuto)
        monkeypatch.setattr(surgery, "AutoTokenizer", _FakeAuto)
        report = apply_foreign_tokenizer(model, tok, "donor/model", seed=42)
        assert report.mode == MODE_WHOLESALE
        assert torch.equal(
            model.get_input_embeddings().weight.data,
            donor_embed.to(model.get_input_embeddings().weight.dtype),
        )
