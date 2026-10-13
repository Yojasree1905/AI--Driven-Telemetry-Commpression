"""
src/compression
===============
Advanced Data Compression Package:
- Entropy Coding: Shannon-Fano Algorithm, Huffman Algorithm
- Dictionary & Transform Coding: LZ77, LZW Algorithms, Gzip/Bzip Lossless Standards
- Quantization: Distortion Criteria, Uniform Scalar Quantizer, LBG Vector Quantizer
"""

from src.compression.shannon_fano import ShannonFanoCoder
from src.compression.huffman import HuffmanCoder
from src.compression.lzw import LZWCoder
from src.compression.lossless_benchmark import LosslessBenchmarkSuite
from src.compression.scalar_quantization import UniformScalarQuantizer
from src.compression.vector_quantization import LBGVectorQuantizer, VectorQuantizer

__all__ = [
    "ShannonFanoCoder",
    "HuffmanCoder",
    "LZWCoder",
    "LosslessBenchmarkSuite",
    "UniformScalarQuantizer",
    "LBGVectorQuantizer",
    "VectorQuantizer",
]
