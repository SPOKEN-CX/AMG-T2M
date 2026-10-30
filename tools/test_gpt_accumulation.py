"""Check that accumulated per-sequence GPT loss gives the same optimizer update."""

import sys as _project_sys
from pathlib import Path as _project_path
_PROJECT_ROOT = _project_path(__file__).resolve().parents[1]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
_project_sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import copy
import torch
from models.t2m_trans import Text2Motion_Transformer


def sequence_loss(model, inputs, condition, target, lengths, pad):
    logits = model(inputs, condition)
    valid = torch.arange(target.shape[1])[None, :] < (lengths[:, None] + 1)
    token_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target,
                                                   reduction='none', ignore_index=pad)
    return ((token_loss * valid).sum(1) / valid.sum(1)).mean()


def main():
    torch.manual_seed(142)
    codebook = 8
    model = Text2Motion_Transformer(num_vq=codebook, embed_dim=16, clip_dim=16,
                                   block_size=9, num_layers=1, n_head=2, drop_out_rate=0., fc_rate=2)
    reference = copy.deepcopy(model)
    inputs = torch.randint(0, codebook, (8, 7))
    condition = torch.randn(8, 16)
    target = torch.randint(0, codebook, (8, 8))
    lengths = torch.tensor([2, 7, 4, 5, 3, 6, 1, 7])
    for i, length in enumerate(lengths):
        target[i, length] = codebook
        target[i, length + 1:] = codebook + 1
    opt_full = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    opt_accum = torch.optim.AdamW(model.parameters(), lr=1e-3)
    full_loss = sequence_loss(reference, inputs, condition, target, lengths, codebook + 1)
    full_loss.backward()
    accum_loss = 0.
    for indices in (slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)):
        loss = sequence_loss(model, inputs[indices], condition[indices], target[indices],
                             lengths[indices], codebook + 1)
        accum_loss += loss.item() / 4
        (loss / 4).backward()
    assert abs(full_loss.item() - accum_loss) < 1e-6
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
        else:
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=2e-5), name
    opt_full.step()
    opt_accum.step()
    for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters()):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-5), name
    print('PASS: varying-length GPT loss, gradients and AdamW update match full batch')


if __name__ == '__main__':
    main()
