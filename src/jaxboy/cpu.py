import jax.numpy as jnp
from jaxtyping import UInt8, UInt16


class Registers:
    a: UInt8
    b: UInt8
    c: UInt8
    d: UInt8
    e: UInt8
    f: UInt8
    h: UInt8
    l: UInt8

    def get_bc(self) -> UInt16:
        b = jnp.astype(self.b, jnp.uint16) << 8
        c = jnp.astype(self.c, jnp.uint16)
        return b | c

    def set_bc(self, val: UInt16):
        self.b = jnp.astype(((val & 0xFF00) >> 8), jnp.uint8)
        self.c = jnp.astype((val & 0xFF), jnp.uint8)

