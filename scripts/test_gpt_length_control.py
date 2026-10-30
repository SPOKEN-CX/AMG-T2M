"""Test duration boundaries and forced EOS suppression with deterministic logits."""
import torch
from scripts.generate_amg_length_controlled import frames_to_tokens, sample_exact_tokens


class EosFirstModel:
    block_size = 51
    num_vq = 8

    def __call__(self, tokens, feature):
        logits = torch.zeros((1, tokens.shape[1] + 1, 9))
        logits[:, :, 8] = 100  # Without length control this stops immediately.
        logits[:, :, 3] = 50
        return logits


def main():
    assert frames_to_tokens(1) == 1
    assert frames_to_tokens(98.94) == 25
    assert frames_to_tokens(196) == 49
    assert frames_to_tokens(200, allow_clip=True) == 49
    for invalid in (0, -1, float('nan'), float('inf'), 200):
        try:
            frames_to_tokens(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected rejection: {invalid}')
    for count in (1, 25, 49):
        result = sample_exact_tokens(EosFirstModel(), torch.zeros((1, 512)), count)
        assert result.shape == (1, count) and torch.all(result == 3)
    print('PASS: frame/token boundaries, overflow policy and exact-duration EOS suppression')


if __name__ == '__main__':
    main()
