"""
src/compression/lossless_benchmark.py
=====================================
Lossless compression benchmarking suite for network telemetry records.
Compares:
1. Custom Canonical Huffman Coding
2. Lempel-Ziv 77 + Huffman (Deflate / Gzip)
3. LZMA (Lempel-Ziv-Markov chain algorithm)
4. Bzip2 (Burrows-Wheeler Transform)
5. Run-Length Encoding (RLE)
"""

import bz2
import gzip
import lzma
import time
from typing import Any, Dict, List
import numpy as np

from src.compression.huffman import HuffmanCoder


def run_length_encode(data: bytes) -> bytes:
    """Byte-level Run-Length Encoder."""
    if not data:
        return b""
    encoded = bytearray()
    i = 0
    n = len(data)
    while i < n:
        byte_val = data[i]
        run_len = 1
        while i + 1 < n and data[i + 1] == byte_val and run_len < 255:
            i += 1
            run_len += 1
        encoded.append(run_len)
        encoded.append(byte_val)
        i += 1
    return bytes(encoded)


def run_length_decode(data: bytes) -> bytes:
    """Byte-level Run-Length Decoder."""
    if not data:
        return b""
    decoded = bytearray()
    for i in range(0, len(data), 2):
        run_len = data[i]
        byte_val = data[i + 1]
        decoded.extend([byte_val] * run_len)
    return bytes(decoded)


class LosslessBenchmarkSuite:
    """Benchmarks multiple lossless compression algorithms on tabular byte streams."""

    def __init__(self):
        self.huffman = HuffmanCoder()

    def benchmark_all(self, raw_data: bytes) -> List[Dict[str, Any]]:
        orig_size = len(raw_data)
        results = []

        # 1. Custom Huffman
        t0 = time.perf_counter()
        huff_comp = self.huffman.encode(raw_data)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        huff_dec = self.huffman.decode(huff_comp)
        t_dec = time.perf_counter() - t0
        assert huff_dec == raw_data, "Huffman decoding failed bit-exact match!"

        huff_stats = self.huffman.get_stats(raw_data, huff_comp)
        huff_stats["enc_time_sec"] = round(t_enc, 5)
        huff_stats["dec_time_sec"] = round(t_dec, 5)
        huff_stats["throughput_mb_s"] = round((orig_size / (1024 * 1024)) / (t_enc + 1e-9), 2)
        results.append(huff_stats)

        # 2. Deflate / Gzip (LZ77 + Huffman)
        t0 = time.perf_counter()
        gzip_comp = gzip.compress(raw_data, compresslevel=9)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        gzip_dec = gzip.decompress(gzip_comp)
        t_dec = time.perf_counter() - t0
        assert gzip_dec == raw_data

        results.append({
            "algorithm": "Gzip / Deflate (LZ77 + Huffman)",
            "type": "Lossless",
            "original_size_bytes": orig_size,
            "compressed_size_bytes": len(gzip_comp),
            "compression_ratio": round(orig_size / len(gzip_comp), 3),
            "decrease_percentage": round((orig_size - len(gzip_comp)) / orig_size * 100.0, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
            "enc_time_sec": round(t_enc, 5),
            "dec_time_sec": round(t_dec, 5),
            "throughput_mb_s": round((orig_size / (1024 * 1024)) / (t_enc + 1e-9), 2)
        })

        # 3. LZMA
        t0 = time.perf_counter()
        lzma_comp = lzma.compress(raw_data)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        lzma_dec = lzma.decompress(lzma_comp)
        t_dec = time.perf_counter() - t0
        assert lzma_dec == raw_data

        results.append({
            "algorithm": "LZMA (Lempel-Ziv-Markov)",
            "type": "Lossless",
            "original_size_bytes": orig_size,
            "compressed_size_bytes": len(lzma_comp),
            "compression_ratio": round(orig_size / len(lzma_comp), 3),
            "decrease_percentage": round((orig_size - len(lzma_comp)) / orig_size * 100.0, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
            "enc_time_sec": round(t_enc, 5),
            "dec_time_sec": round(t_dec, 5),
            "throughput_mb_s": round((orig_size / (1024 * 1024)) / (t_enc + 1e-9), 2)
        })

        # 4. Bzip2 (BWT + Huffman)
        t0 = time.perf_counter()
        bz2_comp = bz2.compress(raw_data)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        bz2_dec = bz2.decompress(bz2_comp)
        t_dec = time.perf_counter() - t0
        assert bz2_dec == raw_data

        results.append({
            "algorithm": "Bzip2 (Burrows-Wheeler + Huffman)",
            "type": "Lossless",
            "original_size_bytes": orig_size,
            "compressed_size_bytes": len(bz2_comp),
            "compression_ratio": round(orig_size / len(bz2_comp), 3),
            "decrease_percentage": round((orig_size - len(bz2_comp)) / orig_size * 100.0, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
            "enc_time_sec": round(t_enc, 5),
            "dec_time_sec": round(t_dec, 5),
            "throughput_mb_s": round((orig_size / (1024 * 1024)) / (t_enc + 1e-9), 2)
        })

        # 5. Run-Length Encoding
        t0 = time.perf_counter()
        rle_comp = run_length_encode(raw_data)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        rle_dec = run_length_decode(rle_comp)
        t_dec = time.perf_counter() - t0
        assert rle_dec == raw_data

        results.append({
            "algorithm": "Run-Length Encoding (RLE)",
            "type": "Lossless",
            "original_size_bytes": orig_size,
            "compressed_size_bytes": len(rle_comp),
            "compression_ratio": round(orig_size / len(rle_comp), 3),
            "decrease_percentage": round((orig_size - len(rle_comp)) / orig_size * 100.0, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
            "enc_time_sec": round(t_enc, 5),
            "dec_time_sec": round(t_dec, 5),
            "throughput_mb_s": round((orig_size / (1024 * 1024)) / (t_enc + 1e-9), 2)
        })

        return results
