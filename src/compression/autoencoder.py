 

import struct
from typing import Any, Dict, Tuple
import numpy as np
from sklearn.neural_network import MLPRegressor


class NeuralAutoencoderCompressor:
    
    //Non-linear Autoencoder Compressor with Quantized Latent Bottleneck.
    

    def __init__(self, bottleneck_dim: int = 16, quant_bits: int = 8, max_iter: int = 60, random_state: int = 42):
        self.bottleneck_dim = bottleneck_dim
        self.quant_bits = quant_bits
        self.max_iter = max_iter
        self.random_state = random_state

        # Hidden architecture: e.g. [64, bottleneck_dim, 64]
        # We will train an MLP autoencoder mapping X -> X
        self.hidden_dims = (64, bottleneck_dim, 64)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=256,
            verbose=False
        )
        self.input_dim_: int = 0

    def fit(self, X: np.ndarray) -> "NeuralAutoencoderCompressor":
        """Trains the autoencoder on normal/reference traffic."""
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]
        self.model.fit(X, X)
        return self

    def _encode_latent(self, X: np.ndarray) -> np.ndarray:
        """Forward pass up to the bottleneck layer (layer 2)."""
        X = np.asarray(X, dtype=np.float32)
        # Weights: coefs_[0] (D, 64), coefs_[1] (64, bottleneck), coefs_[2] (bottleneck, 64), coefs_[3] (64, D)
        # Layer 1: ReLU(X @ W0 + b0)
        h1 = np.maximum(0, np.dot(X, self.model.coefs_[0]) + self.model.intercepts_[0])
        # Layer 2 (Bottleneck): linear or ReLU
        z = np.dot(h1, self.model.coefs_[1]) + self.model.intercepts_[1]
        return z.astype(np.float32)

    def _decode_latent(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass from bottleneck to output."""
        Z = np.asarray(Z, dtype=np.float32)
        # Layer 3: ReLU(Z @ W2 + b2)
        h3 = np.maximum(0, np.dot(Z, self.model.coefs_[2]) + self.model.intercepts_[2])
        # Output: h3 @ W3 + b3
        x_hat = np.dot(h3, self.model.coefs_[3]) + self.model.intercepts_[3]
        return x_hat.astype(np.float32)

    def compress(self, X: np.ndarray) -> bytes:
        """Compresses feature matrix into quantized latent bottleneck bytes."""
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        B = self.bottleneck_dim

        Z = self._encode_latent(X)  # Shape (N, B)

        z_min = float(np.min(Z))
        z_max = float(np.max(Z))
        z_range = max(1e-7, z_max - z_min)
        max_quant = (1 << self.quant_bits) - 1

        Z_norm = (Z - z_min) / z_range
        Z_quant = np.clip(np.round(Z_norm * max_quant), 0, max_quant)

        dtype = np.uint8 if self.quant_bits <= 8 else np.uint16
        quant_bytes = Z_quant.astype(dtype).tobytes()

        fmt = ">IHHBff"
        header = struct.pack(fmt, N, D, B, self.quant_bits, z_min, z_max)
        return header + quant_bytes

    def decompress(self, payload: bytes) -> np.ndarray:
        """Decompresses quantized latent payload through the decoder."""
        fmt = ">IHHBff"
        header_len = struct.calcsize(fmt)
        N, D, B, quant_bits, z_min, z_max = struct.unpack(fmt, payload[:header_len])
        quant_bytes = payload[header_len:]

        dtype = np.uint8 if quant_bits <= 8 else np.uint16
        Z_quant = np.frombuffer(quant_bytes, dtype=dtype).reshape((N, B))

        max_quant = (1 << quant_bits) - 1
        z_range = z_max - z_min
        Z_hat = (Z_quant.astype(np.float32) / max_quant) * z_range + z_min

        X_hat = self._decode_latent(Z_hat)
        return X_hat.astype(np.float32)

    def compute_anomaly_scores(self, X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
        """
        MDL Anomaly Detection: Reconstruction error per packet.
        Higher error = failure to compress on learned manifold = zero-day anomaly.
        """
        return np.mean((X - X_hat) ** 2, axis=1)

    def evaluate_quality(self, X_orig: np.ndarray, X_hat: np.ndarray, payload: bytes) -> Dict[str, Any]:
        """Calculates MSE, RMSE, PSNR, Decrease % and Compression Ratio."""
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

        return {
            "algorithm": f"Neural Autoencoder (Bottleneck={self.bottleneck_dim}, {self.quant_bits}-bit)",
            "type": "Lossy",
            "bottleneck_dim": self.bottleneck_dim,
            "quant_bits": self.quant_bits,
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
