import json

import torch

from hausafs2.vocoder.hifigan import AttrDict, Generator, _remap_legacy_keys

CFG = AttrDict(upsample_rates=[4, 4], upsample_kernel_sizes=[8, 8], upsample_initial_channel=16,
               resblock_kernel_sizes=[3, 7], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], num_mels=80)


def test_generator_output_shape():
    g = Generator(CFG, legacy=True).eval()
    y = g(torch.randn(1, 80, 10))
    assert y.shape == (1, 1, 10 * 16) and y.abs().max() <= 1.0


def test_legacy_weightnorm_checkpoint_loads_into_parametrized_model():
    """Released HiFi-GAN checkpoints use weight_g/weight_v keys; the fallback must remap them."""
    legacy = Generator(CFG, legacy=True).eval()
    new = Generator(CFG, legacy=False).eval()
    new.load_state_dict(_remap_legacy_keys(legacy.state_dict()))
    x = torch.randn(1, 80, 7)
    with torch.no_grad():
        assert torch.allclose(legacy(x), new(x), atol=1e-5)
