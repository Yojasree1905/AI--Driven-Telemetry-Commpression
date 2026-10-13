"""
src/compression/transform_klt.py
================================
Karhunen-Loève Transform (KLT / PCA) with Uniform Quantization for Lossy Compression.

In data compression theory, KLT is the optimal linear orthogonal transform that:
1. Completely decorrelates input features (diagonal covariance).
2. Maximizes energy compaction into the leading transform coefficients.
3. Quantizes transform coefficients into discrete b-bit integers for high space reduction.
4. Performs exact inverse reconstruction with controlled reconstruction distortion (MSE / PSNR).
"""

import struct
from typing import Any, Dict, Tuple
import numpy as np


class KLTCompressor:
    """
    Karhunen-Loève Transform (PCA) Lossy Compressor with Uniform Quantization.
    """

    def __init__(self, n_components: int = 16, quant_bits: int = 8):
        """
        Args:
            n_components: Number of principal transform coefficients to retain (K <= D).
            quant_bits: Bit depth for uniform quantization (8 or 16 bits).
        """
        self.n_components = n_components
        self.quant_bits = quant_bits
        self.mean_: np.ndarray = None
        self.components_: np.ndarray = None  # Shape (D, K)
        self.explained_variance_ratio_: np.ndarray = None
        self.total_energy_ratio: float = 0.0

    def fit(self, X: np.ndarray) -> "KLTCompressor":
        """
        Learns the empirical covariance matrix, eigenvectors, and eigenvalues on training data.
        X shape: (N, D)
        """
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD for numerically stable Karhunen-Loève Transform
        # X_centered = U * S * Vt -> Covariance = V * (S^2 / N) * V^T
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_variance = np.sum(S ** 2)
        explained_variance = (S ** 2) / total_variance

        # Retain top K components
        k = min(self.n_components, X.shape[1])
        self.components_ = Vt[:k, :].T  # Shape (D, K)
        self.explained_variance_ratio_ = explained_variance[:k]
        self.total_energy_ratio = float(np.sum(self.explained_variance_ratio_))
        return self

    def compress(self, X: np.ndarray) -> bytes:
        """
        Compresses feature matrix X into a compact quantized bitstream payload.
        Returns:
            bytes: serialized self-contained binary payload
        """
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        K = self.components_.shape[1]

        # 1. Forward KLT Transform: Y = (X - mu) * W
        Y = np.dot(X - self.mean_, self.components_)  # Shape (N, K)

        # 2. Uniform Quantization
        # Determine global or per-component min/max
        y_min = float(np.min(Y))
        y_max = float(np.max(Y))
        y_range = max(1e-7, y_max - y_min)

        max_quant_val = (1 << self.quant_bits) - 1
        # Quantize to [0, max_quant_val]
        Y_norm = (Y - y_min) / y_range
        Y_quant = np.clip(np.round(Y_norm * max_quant_val), 0, max_quant_val)

        dtype = np.uint8 if self.quant_bits <= 8 else np.uint16
        quant_bytes = Y_quant.astype(dtype).tobytes()

        HEADER_FORMAT = ">IHhBff"
        header = struct.pack(HEADER_FORMAT, N, D, K, self.quant_bits, y_min, y_max)
        return header + quant_bytes

    def decompress(self, payload: bytes) -> np.ndarray:
        """
        Decompresses payload back into reconstructed feature space X_hat (N, D).
        """
        HEADER_FORMAT = ">IHhBff"
        header_len = struct.calcsize(HEADER_FORMAT)
        N, D, K, quant_bits, y_min, y_max = struct.unpack(HEADER_FORMAT, payload[:header_len])
        dtype = np.uint8 if quant_bits <= 8 else np.uint16
        quant_bytes = payload[header_len:]

        # 1. Recover quantized grid
        Y_quant = np.frombuffer(quant_bytes, dtype=dtype).reshape((N, K))
        max_quant_val = (1 << quant_bits) - 1

        # 2. De-quantization
        y_range = y_max - y_min
        Y_hat = (Y_quant.astype(np.float32) / max_quant_val) * y_range + y_min

        # 3. Inverse KLT Transform: X_hat = Y_hat * W^T + mu
        X_hat = np.dot(Y_hat, self.components_.T) + self.mean_
        return X_hat.astype(np.float32)

    def evaluate_quality(self, X_orig: np.ndarray, X_hat: np.ndarray, payload: bytes) -> Dict[str, Any]:
        """Calculates MSE, RMSE, PSNR (dB), SQNR (dB), Decrease % and Compression Ratio."""
        X_orig = np.asarray(X_orig, dtype=np.float32)
        X_hat = np.asarray(X_hat, dtype=np.float32)

        orig_size_bytes = X_orig.nbytes
        comp_size_bytes = len(payload)
        cr = orig_size_bytes / comp_size_bytes if comp_size_bytes > 0 else 1.0
        decrease_pct = ((orig_size_bytes - comp_size_bytes) / orig_size_bytes * 100.0)

        mse = float(np.mean((X_orig - X_hat) ** 2))
        rmse = float(np.sqrt(mse))
        signal_power = float(np.mean(X_orig ** 2))

        # Peak Signal to Noise Ratio (PSNR)
        peak_val = float(np.max(X_orig) - np.min(X_orig))
        if mse == 0.0:
            psnr = float("inf")
        else:
            psnr = float(10 * np.log10((peak_val ** 2) / mse))

        # Signal-to-Quantization-Noise Ratio (SQNR)
        if mse == 0.0:
            sqnr = float("inf")
        else:
            sqnr = float(10 * np.log10(max(1e-9, signal_power) / mse))

        return {
            "algorithm": f"KLT / PCA Transform ({self.n_components} components, {self.quant_bits}-bit)",
            "type": "Lossy",
            "components_retained": self.n_components,
            "quant_bits": self.quant_bits,
            "energy_retained_pct": round(self.total_energy_ratio * 100.0, 2),
            "original_size_bytes": orig_size_bytes,
            "compressed_size_bytes": comp_size_bytes,
            "compression_ratio": round(cr, 3),
            "decrease_percentage": round(decrease_pct, 2),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "psnr_db": round(psnr, 2) if psnr != float("inf") else 999.0,
            "sqnr_db": round(sqnr, 2) if sqnr != float("inf") else 999.0,
            "exact_match": False,
        }
