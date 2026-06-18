"""Shared pure-PyTorch transformer backbone (VTN + VALL-E).

Adapted from SlangLab-NU/VallE @ e65a69b. See per-module headers.
"""
from .activation import MultiheadAttention
from .embedding import SinePositionalEmbedding, TokenEmbedding
from .transformer import (
    AdaptiveLayerNorm,
    BalancedBasicNorm,
    BasicNorm,
    IdentityNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .utils import Transpose

__all__ = [
    "MultiheadAttention",
    "SinePositionalEmbedding", "TokenEmbedding",
    "LayerNorm", "AdaptiveLayerNorm", "BasicNorm", "BalancedBasicNorm", "IdentityNorm",
    "TransformerEncoder", "TransformerEncoderLayer", "TransformerDecoderLayer",
    "Transpose",
]
