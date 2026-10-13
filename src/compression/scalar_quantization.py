"""
src/compression/scalar_quantization.py
======================================
Uniform Scalar Quantization:
- Maps continuous scalar values into a finite set of discrete levels.
- Step size: Delta = (X_max - X_min) / (2^B - 1).
- Uniform quantization error is bounded by [-Delta / 2, +Delta / 2].
- Theoretical SQNR = 6.02 * B + 1.76 dB.
- Used to benchmark against Vector Quantization.
"""

import struct
from typing import Any, Dict
import numpy as np


class UniformScalarQuantizer:
    """
    Uniform Scalar Quantizer.
    """

    def __init__(self, bit_depth: int = 8):
        """
        Args:
            bit_depth: Number of bits per scalar (e.g. 4-bit, 8-bit, 16-bit).
        """
        self.bit_depth = bit_depth
        self.x_min: float = 0.0
        self.x_max: float = 0.0
        self.delta: float = 0.0
        self.levels: int = 1 << bit_depth

    def fit(self, X: np.ndarray) -> "UniformScalarQuantizer":
        """Calculates dynamic range and uniform step size Delta."""
        X = np.asarray(X, dtype=np.float32)
        self.x_min = float(np.min(X))
        self.x_max = float(np.max(X))
        self.delta = max(1e-7, (self.x_max - self.x_min) / (self.levels - 1))
        return self

    def compress(self, X: np.ndarray) -> bytes:
        """Quantizes scalar features into discrete integers."""
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape

        # Quantization rule: q = round((x - x_min) / Delta)
        q = np.clip(np.round((X - self.x_min) / self.delta), 0, self.levels - 1)

        dtype = np.uint8 if self.bit_depth <= 8 else np.uint16
        quant_bytes = q.astype(dtype).tobytes()

        # Header: N (4B), D (2B), bit_depth (1B), x_min (4B float), delta (4B float)
        fmt = ">IHBff"
        header = struct.pack(fmt, N, D, self.bit_depth, self.x_min, self.delta)
        return header + quant_bytes

    def decompress(self, payload: bytes) -> np.ndarray:
        """Dequantizes indices back into continuous approximations: x_hat = x_min + q * Delta."""
        fmt = ">IHBff"
        header_len = struct.calcsize(fmt)
        N, D, bit_depth, x_min, delta = struct.unpack(fmt, payload[:header_len])
        quant_bytes = payload[header_len:]

        dtype = np.uint8 if bit_depth <= 8 else np.uint16
        q = np.frombuffer(quant_bytes, dtype=dtype).reshape((N, D))

        X_hat = x_min + q.astype(np.float32) * delta
        return X_hat.astype(np.float32)

    def evaluate_quality(self, X_orig: np.ndarray, X_hat: np.ndarray, payload: bytes) -> Dict[str, Any]:
        """Calculates Distortion Criteria (MSE, RMSE, PSNR dB, SQNR dB)."""
        X_orig = np.asarray(X_orig, dtype=np.float32)
        X_hat = np.asarray(X_hat, dtype=np.float32)

        orig_sz = X_orig.nbytes
        comp_sz = len(payload)
        cr = orig_sz / comp_sz if comp_sz > 0 else 1.0
        decrease_pct = ((orig_sz - comp_sz) / orig_sz * 100.0)

        mse = float(np.mean((X_orig - X_hat) ** 2))
        rmse = float(np.sqrt(mse))
        signal_pwr = float(np.mean(X_orig ** 2))

        peak = float(np.max(X_orig) - np.min(X_orig))
        psnr = float(10 * np.log10((peak ** 2) / mse)) if mse > 0 else float("inf")
        sqnr = float(10 * np.log10(max(1e-9, signal_pwr) / mse)) if mse > 0 else float("inf")

        return {
            "algorithm": f"Uniform Scalar Quantizer ({self.bit_depth}-bit)",
            "category": "Scalar Quantization",
            "type": "Lossy",
            "bit_depth": self.bit_depth,
            "step_size_delta": round(self.delta, 5),
            "original_size_bytes": orig_sz,
            "compressed_size_bytes": comp_sz,
            "compression_ratio": round(cr, 3),
            "decrease_percentage": round(decrease_pct, 2),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "psnr_db": round(psnr, 2) if psnr != float("inf") else 999.0,
            "sqnr_db": round(sqnr, 2) if sqnr != float("inf") else 999.0,
            "exact_match": False,
        }
