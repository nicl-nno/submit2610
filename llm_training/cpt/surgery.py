'Foreign-tokenizer surgery: replace the vocab-sized layers of a CPT model.\n\n\nThe trained model keeps its transformer blocks; only the vocabulary-sized\nlayers are rebuilt for the "foreign" tokenizer of another model (e.g. weights\nfrom ``Qwen/Qwen2.5-7B-Instruct`` + the tokenizer of\n``anonymous/base-model``):\n\n* **Case A — dimensions coincide** (the donor\'s hidden size equals the\n  trained model\'s AND the new vocab size equals the donor\'s vocab size): the\n  donor\'s embedding layer and classification layer (lm_head) are copied\n  **wholesale**.\n* **Case B — otherwise** new ``[V_new, d_model]`` layers are built and\n  initialized from **projections** of the donor\'s layers:\n\n  - token rows are matched **by token string** between the new (foreign)\n    tokenizer and the donor vocab — since the foreign tokenizer *belongs* to\n    the donor model, matched rows are copied exactly when the hidden sizes\n    coincide;\n  - when the hidden sizes differ, a least-squares map is fitted on the\n    token-string intersection of the trained model\'s ORIGINAL vocab and the\n    donor vocab (pairs of rows: donor embedding → trained-model embedding;\n    ditto for the heads) and every donor row is projected through it;\n  - tokens the donor vocab lacks (added tokens, padding) fall back to the\n    mean of the built rows plus small deterministic noise.\n\n``anonymous/base-model``):\n\n\nFreezing\n    :func:`freeze_transformer_blocks` freezes everything except the embedding\n    and the classification layer: gradients still FLOW THROUGH the frozen\n    blocks (plain autograd — ``requires_grad=False`` does not detach the\n    graph), but the blocks themselves are not updated. This is the companion\n    of the tokenizer swap: train only the two rebuilt layers while the rest\n    of the network acts as a fixed feature extractor.\n    :func:`freeze_transformer_blocks`'

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

#: Std of the deterministic fallback noise for tokens the donor vocab lacks.

FALLBACK_NOISE_STD = 0.02

#: Minimum number of shared token strings to fit the hidden-dim projection;
#: below this the surgery refuses to run (a projection fitted on a handful of
#: pairs would silently produce garbage init).



MIN_FIT_PAIRS = 16

# : Human-readable surgery modes
MODE_WHOLESALE = "wholesale"
MODE_REMAP = "remap"
MODE_PROJECTION = "projection"


@dataclass
class SurgeryReport:
    'Outcome of the vocab-layer surgery (logged + saved as JSON).'

    #: ``wholesale`` | ``remap`` | ``projection``.
    mode: str
    old_vocab: int
    new_vocab: int
    model_hidden: int
    donor_vocab: int
    donor_hidden: int
    #: rows copied from the donor (exactly or through the projection).
    
    copied_tokens: int
    #: rows initialized via mean + noise (absent from the donor vocab).
    
    fallback_tokens: int
    #: token-string pairs used to fit the projections (0 when dims coincide).
    
    shared_fit_tokens: int
    embed_projection_residual: float | None = None
    head_projection_residual: float | None = None

    def to_dict(self) -> dict:
        'Plain-dict form for ``json.dump``.'
        return asdict(self)


def _fit_projection(
    donor_rows: torch.Tensor, model_rows: torch.Tensor
) -> tuple[torch.Tensor, float]:
    'Least-squares map ``A`` so that ``donor_rows @ A ≈ model_rows``.\n\n\n    Inputs are float32 CPU matrices ``[n, d_donor]`` / ``[n, d_model]``; the\n    returned ``A`` is ``[d_donor, d_model]`` and the residual is the relative\n    Frobenius error ``||X A − Y||_F / ||Y||_F`` (0 = exact).'
    a = torch.linalg.lstsq(donor_rows, model_rows).solution
    residual = torch.linalg.norm(donor_rows @ a - model_rows) / (
        torch.linalg.norm(model_rows) + 1e-12
    )
    return a, float(residual)


def _build_rows(
    new_to_donor: list[int | None],
    donor_matrix: torch.Tensor,
    old_matrix: torch.Tensor,
    fit_pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, float | None, int]:
    'Build the new ``[V_new, d_model]`` matrix from a donor matrix.\n\n\n    ``fit_pairs`` carries ``(donor_idx, old_idx)`` row correspondences over\n    the shared token strings; it is used only when the hidden sizes differ\n    (the projection path). Returns ``(matrix, residual, n_copied)``.'
    v_new = len(new_to_donor)
    d_model = old_matrix.shape[1]
    d_donor = donor_matrix.shape[1]
    residual: float | None = None
    out = torch.zeros(v_new, d_model, dtype=torch.float32)
    if d_donor == d_model:
        for i, donor_idx in enumerate(new_to_donor):
            if donor_idx is not None:
                out[i] = donor_matrix[donor_idx]
    else:
        if len(fit_pairs) < MIN_FIT_PAIRS:
            raise ValueError(
                f'Hidden sizes differ (model {d_model} vs donor {d_donor}) but only {len(fit_pairs)} shared token strings were found (minimum {MIN_FIT_PAIRS}'
            )
        donor_idx = torch.tensor([p[0] for p in fit_pairs], dtype=torch.long)
        old_idx = torch.tensor([p[1] for p in fit_pairs], dtype=torch.long)
        projection, residual = _fit_projection(
            donor_matrix[donor_idx], old_matrix[old_idx]
        )
        for i, didx in enumerate(new_to_donor):
            if didx is not None:
                out[i] = donor_matrix[didx] @ projection
    n_copied = sum(1 for didx in new_to_donor if didx is not None)
    return out, residual, n_copied


def _apply_fallback_noise(
    matrix: torch.Tensor, new_to_donor: list[int | None], seed: int
) -> int:
    'Fill rows absent from the donor vocab with mean + deterministic noise.'
    fallback = [i for i, didx in enumerate(new_to_donor) if didx is None]
    if not fallback:
        return 0
    matched = [i for i, didx in enumerate(new_to_donor) if didx is not None]
    mean = matrix[matched].mean(dim=0) if matched else torch.zeros(
        matrix.shape[1]
    )
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        len(fallback), matrix.shape[1], generator=gen
    ) * FALLBACK_NOISE_STD
    idx = torch.tensor(fallback, dtype=torch.long)
    matrix[idx] = mean.unsqueeze(0) + noise
    return len(fallback)


def replace_vocab_layers(
    model,
    *,
    new_to_donor: list[int | None],
    donor_embed: torch.Tensor,
    donor_head: torch.Tensor | None,
    fit_pairs: list[tuple[int, int]],
    seed: int = 42,
    special_token_ids: dict[str, int | None] | None = None,
) -> SurgeryReport:
    'Rebuild the embedding and classification layers of ``model`` in place.\n\n\n    Args:\n        model: the (already loaded) causal LM being trained — its transformer\n            blocks stay untouched.\n        new_to_donor: ``new_to_donor[i]`` is the donor row index for new\n            token ``i`` (``None`` → fallback mean + noise).\n        donor_embed: the donor\'s input-embedding matrix ``[V_donor, d_donor]``.\n        donor_head: the donor\'s lm_head matrix or ``None`` (a tied donor head\n            falls back to the embedding matrix).\n        fit_pairs: ``(donor_idx, old_idx)`` row pairs over shared token\n            strings, used when the hidden sizes differ.\n        seed: RNG seed for the deterministic fallback noise (logged, repo\n            convention).\n        special_token_ids: optional ``{"eos": …, "pad": …, "bos": …}`` from\n            the new tokenizer, propagated into ``generation_config``.\n\n    Returns:\n        The :class:`SurgeryReport` (also updates ``model.config.vocab_size``\n        and unties ``tie_word_embeddings``).'
    old_embed = model.get_input_embeddings().weight.detach()
    old_head_module = model.get_output_embeddings()
    if old_head_module is None:
        raise ValueError(
            'The model has no output embeddings (lm_head) to replace.'
        )
    old_head = old_head_module.weight.detach()
    donor_head = donor_head if donor_head is not None else donor_embed

    v_new = len(new_to_donor)
    d_model = old_embed.shape[1]
    d_donor = donor_embed.shape[1]
    if v_new == 0:
        raise ValueError('new_to_donor is empty; nothing to build.')

    embed_f32 = donor_embed.detach().to(torch.float32).cpu()
    head_f32 = donor_head.detach().to(torch.float32).cpu()

    new_embed, embed_res, n_copied = _build_rows(
        new_to_donor, embed_f32, old_embed.to(torch.float32).cpu(), fit_pairs
    )
    n_fallback_embed = _apply_fallback_noise(new_embed, new_to_donor, seed)
    new_head, head_res, _ = _build_rows(
        new_to_donor, head_f32, old_head.to(torch.float32).cpu(), fit_pairs
    )
    _apply_fallback_noise(new_head, new_to_donor, seed)

    dtype = old_embed.dtype
    embedding = nn.Embedding(v_new, d_model, dtype=dtype)
    with torch.no_grad():
        embedding.weight.copy_(new_embed.to(dtype))
    model.set_input_embeddings(embedding)

    head = nn.Linear(d_model, v_new, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(new_head.to(dtype))
    model.set_output_embeddings(head)

    model.config.vocab_size = v_new
    model.config.tie_word_embeddings = False
    if special_token_ids:
        gen_cfg = model.generation_config
        for name, key in (("eos", "eos_token_id"), ("pad", "pad_token_id"),
                          ("bos", "bos_token_id")):
            value = special_token_ids.get(name)
            if value is not None:
                setattr(gen_cfg, key, value)

    donor_vocab = donor_embed.shape[0]
    if (
        d_donor == d_model
        and donor_vocab == v_new
        and all(i == didx for i, didx in enumerate(new_to_donor))
    ):
        mode = MODE_WHOLESALE
    elif d_donor == d_model:
        mode = MODE_REMAP
    else:
        mode = MODE_PROJECTION
    report = SurgeryReport(
        mode=mode,
        old_vocab=int(old_embed.shape[0]),
        new_vocab=int(v_new),
        model_hidden=int(d_model),
        donor_vocab=int(donor_vocab),
        donor_hidden=int(d_donor),
        copied_tokens=int(n_copied),
        fallback_tokens=int(n_fallback_embed),
        shared_fit_tokens=len(fit_pairs),
        embed_projection_residual=embed_res,
        head_projection_residual=head_res,
    )
    logger.info(
        "Tokenizer surgery: mode=%s vocab %d→%d hidden model=%d donor=%d, "
        "copied=%d fallback=%d fit_pairs=%d residuals(embed/head)=%.4f/%.4f",
        report.mode, report.old_vocab, report.new_vocab, report.model_hidden,
        report.donor_hidden, report.copied_tokens, report.fallback_tokens,
        report.shared_fit_tokens,
        report.embed_projection_residual or 0.0,
        report.head_projection_residual or 0.0,
    )
    return report


def apply_foreign_tokenizer(
    model,
    tokenizer,
    tokenizer_model_id: str,
    *,
    seed: int = 42,
) -> SurgeryReport:
    'Swap ``model`` onto ``tokenizer`` (the tokenizer of another model).\n\n\n    ``tokenizer_model_id`` names the DONOR model: both the tokenizer and the\n    embedding/classification layers used for initialization are loaded from\n    it (the ``--tokenizer`` flag of ``cpt_train.py`` / ``cpt_tokenize.py``).\n    The donor model is loaded on the CPU in its stored dtype (only its two\n    vocab-sized matrices are needed; the math runs in float32) and released\n    afterwards.\n\nDonor-row bounds\n        ``len(tokenizer)`` may exceed the donor\'s matrix rows (trimmed\n        donors, or padded vocabs the other way round): a token string is\n        only "copied" when its id is a valid donor row; everything else\n        falls back to mean + noise. The same bound applies to the projection\n        fit pairs.\n        ``len(tokenizer)``'
    vocab: dict[str, int] = dict(tokenizer.get_vocab())
    v_new = len(tokenizer)
    token_strs = tokenizer.convert_ids_to_tokens(list(range(v_new)))

    logger.info(
        "Loading the donor model %s on CPU for the vocab-layer surgery...",
        tokenizer_model_id,
    )
    donor = AutoModelForCausalLM.from_pretrained(
        tokenizer_model_id, torch_dtype="auto", low_cpu_mem_usage=True
    )
    try:
        donor_embed = donor.get_input_embeddings().weight
        donor_head_module = donor.get_output_embeddings()
        if donor_head_module is None or donor_head_module.weight is donor_embed:
            donor_head = None  # tied donor head → reuse the embedding matrix
        else:
            donor_head = donor_head_module.weight
        donor_rows = donor_embed.shape[0]

        new_to_donor = [
            didx if (didx is not None and didx < donor_rows) else None
            for didx in (vocab.get(tok) for tok in token_strs)
        ]
        n_missing = sum(1 for didx in new_to_donor if didx is None)
        if n_missing == v_new:
            raise ValueError(
                'The tokenizer vocab maps to no donor rows at all; refusing to build an all-fallback model.'
            )
        if n_missing:
            logger.info("Surgery: %d/%d tokens fall back to mean+noise (absent from the donor's %d rows).", n_missing, v_new, donor_rows)

        # Projection pairs: shared token strings between the DONOR vocab and
        # the trained model's ORIGINAL vocab (only needed when hidden sizes
        # differ; cheap to compute regardless). Both indices are bounded by
        # the corresponding matrix rows.
        
        
        
        
        fit_pairs: list[tuple[int, int]] = []
        old_rows = model.get_input_embeddings().weight.shape[0]
        old_tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        old_vocab: dict[str, int] = dict(old_tokenizer.get_vocab())
        for tok, donor_idx in vocab.items():
            old_idx = old_vocab.get(tok)
            if (
                old_idx is not None
                and donor_idx is not None
                and donor_idx < donor_rows
                and old_idx < old_rows
            ):
                fit_pairs.append((donor_idx, old_idx))
    finally:
        del donor
        gc.collect()

    special = {
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
    }
    return replace_vocab_layers(
        model,
        new_to_donor=new_to_donor,
        donor_embed=donor_embed,
        donor_head=donor_head,
        fit_pairs=fit_pairs,
        seed=seed,
        special_token_ids=special,
    )


def freeze_transformer_blocks(model) -> int:
    'Freeze everything except the embedding and classification layers.\n\n\n    ``requires_grad=False`` keeps the autograd graph intact — activations\n    still flow through the frozen blocks during the forward/backward pass,\n    so the two trainable layers receive correct gradients; the blocks\n    themselves are simply not updated by the optimizer (HF ``Trainer`` builds\n    param groups only over ``requires_grad=True``). ``enable_input_require_grads``\n    is called defensively for gradient-checkpointing compatibility. Returns\n    the number of frozen parameter tensors.'
    trainable_ids = {id(model.get_input_embeddings().weight)}
    head = model.get_output_embeddings()
    if head is not None:
        trainable_ids.add(id(head.weight))
    frozen = 0
    for _, param in model.named_parameters():
        if id(param) in trainable_ids:
            continue
        if param.requires_grad:
            param.requires_grad_(False)
            frozen += 1
    model.enable_input_require_grads()
    logger.info(
        "Froze %d parameter tensors outside embed/lm_head "
        "(gradients still flow through them).",
        frozen,
    )
    return frozen
