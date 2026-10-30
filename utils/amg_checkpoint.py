"""AMG VQ training checkpoints, including non-buffer EMA quantizer state."""
import os
import random
import numpy as np
import torch


def save_training_state(path, net, optimizer, scheduler, iteration, args, best):
    quantizer = net.vqvae.quantizer
    ema = {}
    for name in ('init', 'code_sum', 'code_count'):
        if hasattr(quantizer, name):
            value = getattr(quantizer, name)
            ema[name] = value.detach().cpu() if torch.is_tensor(value) else value
    state = {'format_version': 1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'iter': iteration, 'args': vars(args),
             'best': best, 'quantizer_ema': ema,
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}
    temporary = str(path) + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path, net, optimizer, scheduler, args):
    state = torch.load(path, map_location='cpu')
    if state.get('format_version') != 1 or 'quantizer_ema' not in state:
        raise ValueError('Use an AMG training_last.pth with EMA state; old weight-only checkpoints cannot resume this run')
    for key in ('dataname', 'quantizer', 'nb_code', 'code_dim', 'output_emb_width',
                'down_t', 'stride_t', 'width', 'depth', 'dilation_growth_rate', 'vq_act', 'vq_norm',
                'batch_size', 'window_size', 'amg_pairs_dir', 'amg_neighbors', 'amg_temperature',
                'amg_contrastive_weight', 'lr', 'lr_scheduler', 'gamma', 'mu', 'commit',
                'loss_vel', 'recons_loss', 'weight_decay'):
        if state['args'][key] != getattr(args, key):
            raise ValueError(f'Cannot resume with changed {key}')
    net.load_state_dict(state['net'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    quantizer = net.vqvae.quantizer
    device = next(net.parameters()).device
    for name, value in state['quantizer_ema'].items():
        setattr(quantizer, name, value.to(device) if torch.is_tensor(value) else value)
    if args.quantizer == 'ema_reset' and (not quantizer.init or quantizer.code_sum is None or quantizer.code_count is None):
        raise ValueError('Incomplete EMA state')
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['iter'], state['best']
