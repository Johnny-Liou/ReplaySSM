# SPDX-License-Identifier: Apache-2.0
"""The nine precision configurations used in the main text."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Precision:
    state: int
    stochastic: bool
    read: int | None = None

    @property
    def two_precision(self):
        return self.read is not None

    @property
    def nvfp4(self):
        return self.state == 3 or self.read == 3


# Format IDs: FP32=0, FP16=1, FP8 E4M3=2, NVFP4 E2M1=3.
MODES = {
    "s32_rs": Precision(0, False),
    "f16_sr_rs": Precision(1, True),
    "f8_sr_rs": Precision(2, True),
    "f4_sr_rs": Precision(3, True),
    "tp8_hi32": Precision(0, False, 2),
    "tp8_hi16_sr": Precision(1, True, 2),
    "tp4_hi32": Precision(0, False, 3),
    "tp4_hi16_sr": Precision(1, True, 3),
    "tp4_hi8_sr": Precision(2, True, 3),
}
