import math
from itertools import product
from typing import List, Optional


class FloatingPoint:
    def __init__(
        self,
        sign_bits: int,
        exponent_bits: int,
        mantissa_bits: int,
        bias: int,
        bits: int,
        max_mantissa_at_max_exponent: Optional[int] = None,
        reserved_exponent: bool = True,
    ):
        assert bits > 0 and bits == sign_bits + exponent_bits + mantissa_bits and 0 <= sign_bits <= 1
        self.bits = bits
        self.sign_bits = sign_bits
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.reserved_exponent = reserved_exponent
        self.max_mantissa_at_max_exponent = max_mantissa_at_max_exponent or 2**mantissa_bits - 1

    @property
    def is_signed(self) -> bool:
        return self.sign_bits > 0

    @property
    def epsilon(self) -> float:
        return float(2 ** (-self.mantissa_bits))

    @property
    def minimum(self) -> float:
        return -self.maximum if self.is_signed else 0.0

    @property
    def maximum(self) -> float:
        if self.exponent_bits == 0:
            max_exponent, max_mantissa = 1 - self.bias, (2**self.mantissa_bits) - 1
            return float((max_mantissa / (2**self.mantissa_bits)) * (2**max_exponent))
        else:
            max_stored_exponent = (2**self.exponent_bits - 2) if self.reserved_exponent else (2**self.exponent_bits - 1)
            max_exponent = max_stored_exponent - self.bias
            return float((1 + self.max_mantissa_at_max_exponent / (2**self.mantissa_bits)) * (2**max_exponent))

    def generate_bit_combinations(self) -> List[int]:
        return [int("".join(map(str, bits)), 2) for bits in product([0, 1], repeat=self.bits)]

    def bit_pattern_to_custom_fp(self, bit_pattern: int) -> float:
        total_bits = self.sign_bits + self.exponent_bits + self.mantissa_bits
        sign_mask = (1 << (total_bits - 1)) if self.is_signed else 0
        exponent_mask = ((1 << self.exponent_bits) - 1) << self.mantissa_bits
        mantissa_mask = (1 << self.mantissa_bits) - 1
        sign = (bit_pattern & sign_mask) >> (self.exponent_bits + self.mantissa_bits) if self.is_signed else 0
        exponent = (bit_pattern & exponent_mask) >> self.mantissa_bits
        mantissa = bit_pattern & mantissa_mask
        sign_factor = -1 if sign else 1
        if self.exponent_bits == 0:
            exponent_value, mantissa_value = 1 - self.bias, mantissa / (2**self.mantissa_bits)
            return sign_factor * 0.0 if mantissa == 0 else float(sign_factor * mantissa_value * (2**exponent_value))
        else:
            max_exponent = (1 << self.exponent_bits) - 1
            if self.reserved_exponent and exponent == max_exponent:
                return sign_factor * math.inf if mantissa == 0 else math.nan
            elif exponent == 0:
                if mantissa == 0:
                    return sign_factor * 0.0
                mantissa_value = mantissa / (2**self.mantissa_bits)
                return float(sign_factor * mantissa_value * (2 ** (1 - self.bias)))
            else:
                exponent_value, mantissa_value = exponent - self.bias, 1 + (mantissa / (2**self.mantissa_bits))
                return float(sign_factor * mantissa_value * (2**exponent_value))

    def generate_all_custom_fp_values(self):
        bit_combinations = self.generate_bit_combinations()
        values = [self.bit_pattern_to_custom_fp(b) for b in bit_combinations]
        def sort_func(x):
            return (math.inf if math.isnan(x) else math.copysign(1, x), math.inf if math.isnan(x) else x)
        values = sorted(values, key=sort_func)
        assert len(values) == 2**self.bits
        return values

    @property
    def values(self) -> List[float]:
        return self.generate_all_custom_fp_values()

    def __repr__(self) -> str:
        return (
            f"Float{self.bits}-"
            f"S{self.sign_bits}"
            f"E{self.exponent_bits}"
            f"M{self.mantissa_bits}"
            f"B{self.bias}"
            f"MaxM{self.max_mantissa_at_max_exponent}"
            f"{'R' if self.reserved_exponent else ''}"
        )
