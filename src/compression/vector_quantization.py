"""
src/compression/vector_quantization.py
======================================
Linde-Buzo-Gray (LBG) Vector Quantization:
- Shannon's rate-distortion theorem: Coding vectors achieves superior efficiency
  over coding scalars individually.
- LBG algorithm iteratively clusters training vectors to determine K reproduction codebook centroids.
- Replaces each D-dimensional vector with a single codebook index (log2 K bits).
"""

import struct
from typing import Any, Dict
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class LBGVectorQuantizer:
    """
    Linde-Buzo-Gray (LBG) Vector Quantizer.
    """

    def __init__(self, codebook_size: int = 128, random_state: int = 42):
        """
        Args:
            codebook_size: Number of centroid vectors K in the codebook (e.g., 64, 128, 256, 512).
        """
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(
            n_clusters=codebook_size,
            random_state=random_state,
            batch_size=1024,
            max_iter=100,
            n_init=3
        )
        self.codebook_: np.ndarray = None  # Shape (K, D)

    def fit(self, X: np.ndarray) -> "VectorQuantizer":
        """Learns the optimal codebook centroids using LBG/K-Means."""
        X = np.asarray(X, dtype=np.float32)
        self.kmeans.fit(X)
        self.codebook_ = self.kmeans.cluster_centers_.astype(np.float32)
        return self

    def compress(self, X: np.ndarray, include_codebook: bool = False) -> bytes:
        """
        Encodes vectors into centroid indices.
        If include_codebook is True, bundles the codebook in the stream (self-contained mode).
        If False, acts as codebook-shared transmission (standard in streaming VQ).
        """
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        K = self.codebook_size

        indices = self.kmeans.predict(X)  # Shape (N,)
        dtype = np.uint8 if K <= 256 else np.uint16
        idx_bytes = indices.astype(dtype).tobytes()

        # Format:
        # N (4B), D (2B), K (2B), has_codebook (1B)
        fmt = ">IHH?"
        header = struct.pack(fmt, N, D, K, include_codebook)
        codebook_bytes = self.codebook_.tobytes() if include_codebook else b""
        return header + codebook_bytes + idx_bytes

    def decompress(self, payload: bytes) -> np.ndarray:
        """
        Decodes index payload into reconstructed feature matrix X_hat (N, D).
        """
        fmt = ">IHH?"
        header_len = struct.calcsize(fmt)
        N, D, K, has_codebook = struct.unpack(fmt, payload[:header_len])
        offset = header_len

        if has_codebook:
            cb_size = K * D * 4  # float32 = 4 bytes
            codebook = np.frombuffer(payload[offset:offset + cb_size], dtype=np.float32).reshape((K, D))
            offset += cb_size
        else:
            codebook = self.codebook_

        dtype = np.uint8 if K <= 256 else np.uint16
        indices = np.frombuffer(payload[offset:], dtype=dtype)
        X_hat = codebook[indices]
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

        peak_val = float(np.max(X_orig) - np.min(X_orig))
        if mse == 0.0:
            psnr = float("inf")
            sqnr = float("inf")
        else:
            psnr = float(10 * np.log10((peak_val ** 2) / mse))
            sqnr = float(10 * np.log10(max(1e-9, signal_power) / mse))

        bits_per_vector = np.ceil(np.log2(self.codebook_size))
        bits_per_dim = bits_per_vector / X_orig.shape[1]

        return {
            "algorithm": f"LBG Vector Quantization (K={self.codebook_size})",
            "category": "Vector Quantization",
            "type": "Lossy",
            "codebook_size": self.codebook_size,
            "bits_per_dimension": round(float(bits_per_dim), 3),
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


# Backward-compatibility alias
VectorQuantizer = LBGVectorQuantizer

