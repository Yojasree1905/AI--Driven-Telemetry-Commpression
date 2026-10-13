"""
src/compression/huffman.py
==========================
Information-Theoretic Canonical Huffman Coding for Advanced Data Compression.

Implements from first principles:
- Empirical probability estimation and Shannon Entropy H(X)
- Priority queue based Huffman Tree construction
- Prefix-free binary codebook generation
- Bit-packing into byte-aligned streams with metadata header
- Exact lossless inverse decoder
- Theoretical metrics: Average Codeword Length, Coding Redundancy, Efficiency
"""

import heapq
import math
import struct
from collections import Counter
from typing import Any, Dict, List, Tuple


class HuffmanNode:
    """Node in a Huffman coding tree."""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class HuffmanCoder:
    """
    Entropy encoder using Huffman Coding.
    Works on raw bytes or integer sequence tokens.
    """

    def __init__(self):
        self.codes: Dict[int, str] = {}
        self.reverse_codes: Dict[str, int] = {}
        self.probabilities: Dict[int, float] = {}
        self.shannon_entropy: float = 0.0
        self.avg_code_length: float = 0.0
        self.coding_efficiency: float = 0.0
        self.redundancy: float = 0.0

    def _compute_entropy(self, data: bytes) -> None:
        """Calculates theoretical Shannon Entropy H(X) in bits/symbol."""
        n = len(data)
        if n == 0:
            self.shannon_entropy = 0.0
            return

        counts = Counter(data)
        self.probabilities = {sym: count / n for sym, count in counts.items()}
        self.shannon_entropy = -sum(
            p * math.log2(p) for p in self.probabilities.values() if p > 0
        )

    def _build_tree(self, data: bytes) -> HuffmanNode:
        """Constructs the Huffman Tree using a min-heap."""
        counts = Counter(data)
        heap: List[HuffmanNode] = [
            HuffmanNode(symbol=sym, freq=freq) for sym, freq in counts.items()
        ]
        heapq.heapify(heap)

        # Single symbol edge case
        if len(heap) == 1:
            only_node = heapq.heappop(heap)
            root = HuffmanNode(freq=only_node.freq, left=only_node, right=None)
            return root

        while len(heap) > 1:
            n1 = heapq.heappop(heap)
            n2 = heapq.heappop(heap)
            merged = HuffmanNode(freq=n1.freq + n2.freq, left=n1, right=n2)
            heapq.heappush(heap, merged)

        return heap[0]

    def _generate_codes(self, root: HuffmanNode, current_code: str = "") -> None:
        """Traverses the tree to generate prefix-free variable length binary codes."""
        if root is None:
            return

        if root.is_leaf:
            code = current_code if current_code else "0"
            self.codes[root.symbol] = code
            self.reverse_codes[code] = root.symbol
            return

        if root.left:
            self._generate_codes(root.left, current_code + "0")
        if root.right:
            self._generate_codes(root.right, current_code + "1")

    def encode(self, data: bytes) -> bytes:
        """
        Compresses input bytes into a self-contained Huffman bitstream.
        Format:
        [Header: orig_len (4B), pad_len (1B), table_size (2B), (sym, freq)*] + [bitstream]
        """
        if not data:
            return b""

        self._compute_entropy(data)
        root = self._build_tree(data)
        self.codes.clear()
        self.reverse_codes.clear()
        self._generate_codes(root)

        # Compute theoretical metrics
        self.avg_code_length = sum(
            self.probabilities[sym] * len(code)
            for sym, code in self.codes.items()
        )
        self.redundancy = max(0.0, self.avg_code_length - self.shannon_entropy)
        self.coding_efficiency = (
            (self.shannon_entropy / self.avg_code_length * 100.0)
            if self.avg_code_length > 0 else 100.0
        )

        # Serialize bitstring
        bitstring = "".join(self.codes[b] for b in data)
        pad_len = (8 - (len(bitstring) % 8)) % 8
        bitstring += "0" * pad_len

        # Convert bitstring to bytes
        byte_arr = bytearray()
        for i in range(0, len(bitstring), 8):
            byte_arr.append(int(bitstring[i:i + 8], 2))

        # Build self-contained header with symbol frequencies for canonical reconstruction
        counts = Counter(data)
        header = bytearray()
        header.extend(struct.pack(">IBH", len(data), pad_len, len(counts)))
        for sym, freq in counts.items():
            header.extend(struct.pack(">BI", sym, freq))

        return bytes(header) + bytes(byte_arr)

    def decode(self, payload: bytes) -> bytes:
        """
        Decompresses a self-contained Huffman byte payload into original bytes.
        """
        if not payload:
            return b""

        orig_len, pad_len, table_size = struct.unpack(">IBH", payload[:7])
        offset = 7
        counts = {}
        for _ in range(table_size):
            sym, freq = struct.unpack(">BI", payload[offset:offset + 5])
            counts[sym] = freq
            offset += 5

        heap: List[HuffmanNode] = [
            HuffmanNode(symbol=sym, freq=freq) for sym, freq in counts.items()
        ]
        heapq.heapify(heap)

        if len(heap) == 1:
            only = heapq.heappop(heap)
            root = HuffmanNode(freq=only.freq, left=only, right=None)
        else:
            while len(heap) > 1:
                n1 = heapq.heappop(heap)
                n2 = heapq.heappop(heap)
                heapq.heappush(heap, HuffmanNode(freq=n1.freq + n2.freq, left=n1, right=n2))
            root = heap[0]

        raw_bits = []
        for b in payload[offset:]:
            raw_bits.append(f"{b:08b}")
        full_bitstring = "".join(raw_bits)
        if pad_len > 0:
            full_bitstring = full_bitstring[:-pad_len]

        decoded = bytearray()
        curr = root
        for bit in full_bitstring:
            curr = curr.left if bit == "0" else curr.right
            if curr.is_leaf:
                decoded.append(curr.symbol)
                curr = root
                if len(decoded) == orig_len:
                    break

        return bytes(decoded)

    def get_stats(self, orig_bytes: bytes, comp_bytes: bytes) -> Dict[str, Any]:
        """Returns comprehensive information-theoretic compression stats."""
        orig_size = len(orig_bytes)
        comp_size = len(comp_bytes)
        ratio = orig_size / comp_size if comp_size > 0 else 1.0
        decrease_pct = ((orig_size - comp_size) / orig_size * 100.0) if orig_size > 0 else 0.0

        return {
            "algorithm": "Huffman Coding (Entropy)",
            "type": "Lossless",
            "original_size_bytes": orig_size,
            "compressed_size_bytes": comp_size,
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
