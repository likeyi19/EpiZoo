"""Strict structural qualification around the public species-transfer APIs.

No training replay, alternate transfer algorithm, or permissive checkpoint load.
Architecture and resource selection remain the caller's responsibility.
"""
from dataclasses import asdict
import torch

from .models.epizoo import EpiZoo
from .models.epizoo_x import EpiZooX
from .models.transfer import transfer_epizoox_state_dict, transfer_epizoox_state_dict_with_map


LOSS_BUFFERS = ('cca_loss_fn.pos_weight', 'signal_loss_fn.pos_weight')


def checked_state(model, state, *, allow_missing_loss_buffers=False, floating_dtype=None):
    """Match every tensor, including buffers, before strict loading.

    Only the two config-owned scalar BCE buffers may be synthesized for old
    source checkpoints. Learned parameters never receive defaults.
    """
    expected = model.state_dict()
    state = dict(state)
    if allow_missing_loss_buffers:
        for key, value in zip(LOSS_BUFFERS, (model.cfg.cca_pos_weight, model.cfg.signal_pos_weight)):
            if key not in state:
                state[key] = torch.tensor(value, dtype=torch.float32)
    if set(state) != set(expected):
        raise ValueError('Checkpoint tensor keys differ from the explicit architecture.')
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape:
            raise ValueError(f'Incompatible tensor shape: {key}')
        dtype = torch.float32 if key in LOSS_BUFFERS else floating_dtype
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f'Incompatible tensor dtype: {key}')
        if not tensor.is_floating_point():
            raise ValueError(f'Non-floating model tensor: {key}')
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_048_576):
            if not torch.isfinite(flat[start:start+1_048_576]).all():
                raise ValueError(f'Nonfinite tensor: {key}')
    for key, value in zip(LOSS_BUFFERS, (model.cfg.cca_pos_weight, model.cfg.signal_pos_weight)):
        if state[key].item() != value:
            raise ValueError(f'Unexpected loss buffer: {key}')
    return state


def qualify_source(state, cfg):
    with torch.device('meta'):
        model = EpiZoo(cfg)
    state = checked_state(model, state, allow_missing_loss_buffers=True, floating_dtype=torch.float16)
    model.load_state_dict(state, strict=True, assign=True)
    return state


def initialize_target(source, sequence_embeddings, cfg, *, strategy, mapping=None, source_species=None):
    """Invoke public transfer and verify inheritance before returning an FP32 model."""
    seq = torch.as_tensor(sequence_embeddings, dtype=torch.float32)
    n = cfg.signal_vocab_size
    if seq.shape != (n, cfg.emb_dim) or not torch.isfinite(seq).all():
        raise ValueError('Invalid ordered target sequence embeddings.')
    if strategy == 'de_novo':
        if mapping is not None or source_species is not None:
            raise ValueError('De-novo transfer cannot consume mapping resources.')
        state = transfer_epizoox_state_dict(source, seq, num_ccres=n)
    elif strategy == 'mapped_reference':
        if not isinstance(mapping, dict) or source_species not in ('human', 'mouse'):
            raise ValueError('Mapped transfer requires an explicit source species and correspondence.')
        if any(type(k) is not int or type(v) is not int for k,v in mapping.items()):
            raise ValueError('Correspondence indices must be exact integers.')
        state = transfer_epizoox_state_dict_with_map(source, seq, mapping,
            source_species=source_species, num_ccres=n)
    else:
        raise ValueError('Unknown transfer strategy; no fallback is permitted.')
    for key in state:
        if key not in ('ccre_emb.weight','seq_emb.weight','signal_decoder.weight','signal_decoder.bias'):
            if key not in source or not torch.equal(state[key], source[key]):
                raise ValueError(f'Shared parameter inheritance failed: {key}')
    for key in ('ccre_emb.weight','seq_emb.weight'):
        if not torch.equal(state[key][:4], source[key][:4].float()):
            raise ValueError('Special-token inheritance failed.')
    if not torch.equal(state['seq_emb.weight'][4:], seq):
        raise ValueError('Target sequence embedding inheritance failed.')
    if strategy == 'mapped_reference':
        offset = 4 + (source['signal_decoder.decoders.human.weight'].shape[0] if source_species=='mouse' else 0)
        for target, reference in mapping.items():
            for dst, src, ti, ri in (
                ('ccre_emb.weight','ccre_emb.weight',target+4,reference+offset),
                ('signal_decoder.weight',f'signal_decoder.decoders.{source_species}.weight',target,reference),
                ('signal_decoder.bias',f'signal_decoder.decoders.{source_species}.bias',target,reference)):
                if not torch.equal(state[dst][ti],source[src][ri].float()):
                    raise ValueError('Mapped parameter inheritance failed.')
    with torch.device('meta'):
        model = EpiZooX(cfg)
    state = {k:v.float() for k,v in state.items()}
    checked_state(model,state,floating_dtype=torch.float32)
    model.load_state_dict(state,strict=True,assign=True)
    model.seq_emb.requires_grad_(False)
    return model


def strict_target(state, cfg):
    with torch.device('meta'):
        model = EpiZooX(cfg)
    checked_state(model,state,floating_dtype=torch.float32)
    model.load_state_dict(state,strict=True,assign=True)
    model.seq_emb.requires_grad_(False)
    return model


def parameter_scopes(model):
    return {kind:sorted(name for name,p in model.named_parameters() if p.requires_grad==trainable)
            for kind,trainable in (('trainable',True),('frozen',False))}


@torch.no_grad()
def check_forward(model, input_ids, *, device, use_amp):
    model.to(device).eval()
    with torch.amp.autocast(device_type=torch.device(device).type,
                            enabled=use_amp and torch.device(device).type=='cuda'):
        value=model(input_ids.to(device),return_transformer_out=False)['cell_emb']
        decoded=model.signal_decoder(value)
    if value.shape!=(input_ids.shape[0],model.cfg.emb_dim) or decoded.shape!=(input_ids.shape[0],model.cfg.signal_vocab_size):
        raise ValueError('Target forward dimensions differ.')
    if not torch.isfinite(value).all() or not torch.isfinite(decoded).all():
        raise ValueError('Nonfinite target forward output.')
    return dict(cells=input_ids.shape[0],embedding_dim=model.cfg.emb_dim,decoder_size=model.cfg.signal_vocab_size)
