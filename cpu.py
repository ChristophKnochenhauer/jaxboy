from dataclasses import dataclass
from jaxtyping import UInt8

@dataclass
class Registers:
    A: UInt8
    B: UInt8
    C: UInt8
    D: UInt8
    E: UInt8
    F: UInt8
    H: UInt8
    L: UInt8
