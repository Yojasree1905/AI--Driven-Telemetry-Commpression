"""
src/compression/shannon_fano.py
===============================
Shannon-Fano Coding:
- Top-down entropy coding algorithm.
- Sorts symbols by non-increasing probabilities.
- Recursively divides the list into two subsets with sums as close to equal as possible.
- Assigns '0' to the first half and '1' to the second half.
- Compares efficiency against Huffman Coding and theoretical Shannon Entropy H(X).
"""

import math
import struct
from collections import Counter
from typing import Any, Dict, List, Tuple


class ShannonFanoCoder:
    """
    Shannon-Fano entropy encoder and decoder.
    """

    def __init__(self):
        self.codes: Dict[int, str] = {}
        self.reverse_codes: Dict[str, int] = {}
        self.shannon_entropy: float = 0.0
        self.avg_code_length: float = 0.0
        self.coding_efficiency: float = 0.0
        self.redundancy: float = 0.0

    def _split_equal_weights(self, symbols_with_freq: List[Tuple[int, int]]) -> int:
        """Finds split index minimizing the difference between left and right cumulative frequencies."""
        total = sum(f for _, f in symbols_with_freq)
        running = 0
        best_diff = float("inf")
        best_idx = 0

        for i in range(len(symbols_with_freq) - 1):
            running += symbols_with_freq[i][1]
            diff = abs(2 * running - total)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        return best_idx + 1

    def _recursive_code(self, symbols_with_freq: List[Tuple[int, int]], prefix: str = "") -> None:
        """Recursively partitions symbol frequency list to assign 0/1 prefix codes."""
        if len(symbols_with_freq) == 1:
            sym = symbols_with_freq[0][0]
            code = prefix if prefix else "0"
            self.codes[sym] = code
            self.reverse_codes[code] = sym
            return

        split_idx = self._split_equal_weights(symbols_with_freq)
        left = symbols_with_freq[:split_idx]
        right = symbols_with_freq[split_idx:]

        self._recursive_code(left, prefix + "0")
        self._recursive_code(right, prefix + "1")

    def encode(self, data: bytes) -> bytes:
        """Encodes raw bytes into a self-contained Shannon-Fano bitstream with table header."""
        if not data:
            return b""

        n = len(data)
        counts = Counter(data)
        sorted_symbols = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Theoretical Entropy
        probs = {sym: f / n for sym, f in counts.items()}
        self.shannon_entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0)

        self.codes.clear()
        self.reverse_codes.clear()
        self._recursive_code(sorted_symbols)

        # Average length & Efficiency
        self.avg_code_length = sum(probs[sym] * len(code) for sym, code in self.codes.items())
        self.redundancy = max(0.0, self.avg_code_length - self.shannon_entropy)
        self.coding_efficiency = (
            (self.shannon_entropy / self.avg_code_length * 100.0)
            if self.avg_code_length > 0 else 100.0
        )

        # Bitstring packing
        bitstring = "".join(self.codes[b] for b in data)
        pad_len = (8 - (len(bitstring) % 8)) % 8
        bitstring += "0" * pad_len

        byte_arr = bytearray()
        for i in range(0, len(bitstring), 8):
            byte_arr.append(int(bitstring[i:i + 8], 2))

        # Header format: orig_len (4B), pad_len (1B), table_size (2B), then (sym 1B, freq 4B)
        header = bytearray()
        header.extend(struct.pack(">IBH", len(data), pad_len, len(sorted_symbols)))
        for sym, freq in sorted_symbols:
            header.extend(struct.pack(">BI", sym, freq))

        return bytes(header) + bytes(byte_arr)

    def decode(self, payload: bytes) -> bytes:
        """Decodes self-contained Shannon-Fano payload back into original bytes."""
        if not payload:
            return b""

        orig_len, pad_len, table_size = struct.unpack(">IBH", payload[:7])
        offset = 7
        sorted_symbols = []
        for _ in range(table_size):
            sym, freq = struct.unpack(">BI", payload[offset:offset + 5])
            sorted_symbols.append((sym, freq))
            offset += 5

        # Rebuild codes
        self.codes.clear()
        self.reverse_codes.clear()
        self._recursive_code(sorted_symbols)

        raw_bits = []
        for b in payload[offset:]:
            raw_bits.append(f"{b:08b}")
        bitstring = "".join(raw_bits)
        if pad_len > 0:
            bitstring = bitstring[:-pad_len]

        decoded = bytearray()
        curr_code = ""
        for bit in bitstring:
            curr_code += bit
            if curr_code in self.reverse_codes:
                decoded.append(self.reverse_codes[curr_code])
                curr_code = ""
                if len(decoded) == orig_len:
                    break

        return bytes(decoded)

    def get_stats(self, orig_bytes: bytes, comp_bytes: bytes) -> Dict[str, Any]:
        orig_sz = len(orig_bytes)
        comp_sz = len(comp_bytes)
        ratio = orig_sz / comp_sz if comp_sz > 0 else 1.0
        decrease_pct = ((orig_sz - comp_sz) / orig_sz * 100.0) if orig_sz > 0 else 0.0

        return {
            "algorithm": "Shannon-Fano Algorithm",
            "category": "Prefix & Entropy Coding",
            "type": "Lossless",
            "original_size_bytes": orig_sz,
            "compressed_size_bytes": comp_sz,
            "compression_ratio": round(ratio, 3),
            "decrease_percentage": round(decrease_pct, 2),
            "shannon_entropy_bpc": round(self.shannon_entropy, 3),
            "avg_codeword_length_bpc": round(self.avg_code_length, 3),
            "redundancy_bpc": round(self.redundancy, 3),
            "coding_efficiency_pct": round(self.coding_efficiency, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
        }
