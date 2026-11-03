import numpy as np
import torch
from src.preprocessing.frame_sampling import sample_frame_indices
from src.preprocessing.sequence import make_sequence
from src.models.bilstm_attention import BiLSTMAttention


def test_uniform_sampling_shape_and_bounds():
    idx = sample_frame_indices(101, 48)
    assert idx.shape == (48,)
    assert idx.min() >= 0 and idx.max() < 101
    assert idx[0] == 0 and idx[-1] == 100


def test_sequence_contract():
    features = np.zeros((3, 16, 2, 21, 3), dtype=np.float32)
    flags = np.zeros((3, 16, 2), dtype=np.uint8)
    seq = make_sequence(features, flags)
    assert seq.shape == (3, 16, 128)
    assert seq.dtype == np.float32


def test_bilstm_attention_output_shape():
    model = BiLSTMAttention(classes=7)
    x = torch.zeros(2, 48, 128)
    y = model(x)
    assert y.shape == (2, 7)
