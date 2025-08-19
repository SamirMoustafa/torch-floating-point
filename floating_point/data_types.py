import math
from itertools import product
from typing import List, Optional


class FloatingPoint:
    """
    A class representing a custom floating-point format.
    Parameters:
    - sign_bits: Number of bits for the sign (0 or 1).
    - exponent_bits: Number of bits for the exponent.
    - mantissa_bits: Number of bits for the mantissa.
    - bias: Bias for the exponent.
    - bits: Total number of bits (sign + exponent + mantissa).
    - max_mantissa_at_max_exponent: Maximum mantissa value at maximum exponent.
    - reserved_exponent: Whether the exponent has reserved values (default is True).
    """

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
        assert bits > 0, "Total bits must be positive."
        assert bits == sign_bits + exponent_bits + mantissa_bits, (
            "Sum of sign, exponent, and mantissa bits must equal bits."
        )
        assert 0 <= sign_bits <= 1, "Sign bits must be 0 or 1."
        self.bits = bits
        self.sign_bits = sign_bits
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.reserved_exponent = reserved_exponent
        if max_mantissa_at_max_exponent is not None:
            self.max_mantissa_at_max_exponent = max_mantissa_at_max_exponent
        else:
            self.max_mantissa_at_max_exponent = 2**mantissa_bits - 1

    @property
    def is_signed(self) -> bool:
        return self.sign_bits > 0

    @property
    def epsilon(self) -> float:
        # Smallest positive subnormal value
        return float(2 ** (-self.mantissa_bits))

    @property
    def minimum(self) -> float:
        # Negative of the max finite value
        return -self.maximum if self.is_signed else 0.0

    @property
    def maximum(self) -> float:
        if self.exponent_bits == 0:
            # Subnormal maximum: (max_mantissa / 2^mantissa_bits) * 2^(1 - bias)
            max_exponent = 1 - self.bias
            max_mantissa = (2**self.mantissa_bits) - 1
            return float((max_mantissa / (2**self.mantissa_bits)) * (2**max_exponent))
        else:
            # Calculate max stored exponent based on reserved status
            max_stored_exponent = (2**self.exponent_bits - 2) if self.reserved_exponent else (2**self.exponent_bits - 1)
            max_exponent = max_stored_exponent - self.bias
            return float((1 + self.max_mantissa_at_max_exponent / (2**self.mantissa_bits)) * (2**max_exponent))

    def generate_bit_combinations(self) -> List[int]:
        """Generate all possible bit patterns for the given configuration."""
        total_bits = self.bits
        bit_combinations = list(product([0, 1], repeat=total_bits))
        return [int("".join(map(str, bits)), 2) for bits in bit_combinations]

    def bit_pattern_to_custom_fp(self, bit_pattern: int) -> float:
        total_bits = self.sign_bits + self.exponent_bits + self.mantissa_bits
        # Mask definitions
        sign_mask = (1 << (total_bits - 1)) if self.is_signed else 0
        exponent_mask = ((1 << self.exponent_bits) - 1) << self.mantissa_bits
        mantissa_mask = (1 << self.mantissa_bits) - 1
        sign = (bit_pattern & sign_mask) >> (self.exponent_bits + self.mantissa_bits) if self.is_signed else 0
        exponent = (bit_pattern & exponent_mask) >> self.mantissa_bits
        mantissa = bit_pattern & mantissa_mask
        # Decode components
        sign_factor = -1 if sign else 1
        if self.exponent_bits == 0:
            # Only subnormals
            exponent_value = 1 - self.bias
            mantissa_value = mantissa / (2**self.mantissa_bits)
            if mantissa == 0:
                return sign_factor * 0.0
            return float(sign_factor * mantissa_value * (2**exponent_value))
        else:
            max_exponent = (1 << self.exponent_bits) - 1
            if self.reserved_exponent and exponent == max_exponent:
                if mantissa == 0:
                    return sign_factor * math.inf
                else:
                    return math.nan
            elif exponent == 0:
                if mantissa == 0:
                    return sign_factor * 0.0
                else:
                    mantissa_value = mantissa / (2**self.mantissa_bits)
                    return float(sign_factor * mantissa_value * (2 ** (1 - self.bias)))
            else:
                exponent_value = exponent - self.bias
                mantissa_value = 1 + (mantissa / (2**self.mantissa_bits))
                return float(sign_factor * mantissa_value * (2**exponent_value))

    def generate_all_custom_fp_values(self) -> List[float]:
        bit_combinations = self.generate_bit_combinations()
        values = [self.bit_pattern_to_custom_fp(b) for b in bit_combinations]
        values = sorted(
            values,
            key=lambda x: (
                math.inf if math.isnan(x) else math.copysign(1, x),
                math.inf if math.isnan(x) else x,
            ),
        )
        assert len(values) == 2**self.bits, f"Incorrect number of values generated: {len(values)} != {2**self.bits}"
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
