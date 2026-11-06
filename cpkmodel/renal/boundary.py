"""已知心端、水输入与肾滤过容量机制扰动。"""
from dataclasses import dataclass
from ..circulation.boundary import BoundaryValues
from ..circulation.parameters import finite_real


@dataclass(frozen=True)
class RenalBoundary(BoundaryValues):
    water_input: float = 1500/86400
    filtration_scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        finite_real(self.water_input,'净水输入',nonnegative=True)
        finite_real(self.filtration_scale,'滤过容量比例',nonnegative=True)
