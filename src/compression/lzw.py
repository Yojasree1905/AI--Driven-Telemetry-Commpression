"""
src/compression/lzw.py
======================
Lempel-Ziv-Welch (LZW) Compression:
- Universal dictionary-based lossless compression.
- Initializes dictionary with 256 single-byte entries (0-255).
- Dynamically discovers recurring byte substrings and assigns new integer codes.
- Outputs a sequence of 16-bit dictionary index codes.
- The decompressor rebuilds the identical dictionary dynamically during decoding without
  transmitting the dictionary itself.
"""

import struct
from typing import Any, Dict, List


class LZWCoder:
    """
    Lempel-Ziv-Welch (LZW) Dictionary Encoder and Decoder.
    """

    def __init__(self, max_dict_size: int = 65536):
        self.max_dict_size = max_dict_size

    def encode(self, data: bytes) -> bytes:
        """Compresses data bytes into an LZW 16-bit code sequence stream."""
        if not data:
            return b""

        # Initialize dictionary with all 256 single byte values
        dictionary: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        dict_size = 256

        current = bytearray()
        result_codes: List[int] = []

        for byte_val in data:
            current_plus_c = bytes(current) + bytes([byte_val])
            if current_plus_c in dictionary:
                current.append(byte_val)
            else:
                result_codes.append(dictionary[bytes(current)])
                if dict_size < self.max_dict_size:
                    dictionary[current_plus_c] = dict_size
                    dict_size += 1
                current = bytearray([byte_val])

        if current:
            result_codes.append(dictionary[bytes(current)])

        # Pack codes into 16-bit integers
        header = struct.pack(">II", len(data), len(result_codes))
        code_bytes = struct.pack(f">{len(result_codes)}H", *result_codes)
        return header + code_bytes

    def decode(self, payload: bytes) -> bytes:
        """Decompresses LZW code stream back into original bytes."""
        if not payload:
            return b""

        header_len = 8
        orig_len, num_codes = struct.unpack(">II", payload[:header_len])
        result_codes = struct.unpack(f">{num_codes}H", payload[header_len:])

        dictionary: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        dict_size = 256

        if not result_codes:
            return b""

        old_code = result_codes[0]
        decoded = bytearray(dictionary[old_code])
        character = bytes([dictionary[old_code][0]])

        for code in result_codes[1:]:
            if code in dictionary:
                entry = dictionary[code]
            elif code == dict_size:
                entry = dictionary[old_code] + character
            else:
                raise ValueError(f"Bad LZW code: {code}")

            decoded.extend(entry)
            character = bytes([entry[0]])

            if dict_size < self.max_dict_size:
                dictionary[dict_size] = dictionary[old_code] + character
                dict_size += 1

            old_code = code

        return bytes(decoded[:orig_len])

    def get_stats(self, orig_bytes: bytes, comp_bytes: bytes) -> Dict[str, Any]:
        orig_sz = len(orig_bytes)
        comp_sz = len(comp_bytes)
        ratio = orig_sz / comp_sz if comp_sz > 0 else 1.0
        decrease_pct = ((orig_sz - comp_sz) / orig_sz * 100.0) if orig_sz > 0 else 0.0

        return {
            "algorithm": "LZW Algorithm (Lempel-Ziv-Welch)",
            "category": "Dictionary-Based Coding",
            "type": "Lossless",
            "original_size_bytes": orig_sz,
            "compressed_size_bytes": comp_sz,
            "compression_ratio": round(ratio, 3),
            "decrease_percentage": round(decrease_pct, 2),
            "exact_match": True,
            "mse": 0.0,
            "psnr_db": float("inf"),
        }
