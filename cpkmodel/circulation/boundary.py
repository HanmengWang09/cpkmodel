"""已知心端边界；不包含心脏主动收缩方程。"""
from bisect import bisect_right
from dataclasses import dataclass
from .parameters import finite_real


@dataclass(frozen=True)
class BoundaryValues:
    systemic_inflow: float
    pulmonary_inflow: float
    right_atrial_pressure: float
    left_atrial_pressure: float

    def __post_init__(self):
        for name, value in vars(self).items():
            finite_real(value, name, nonnegative=name.endswith('inflow'))


@dataclass(frozen=True)
class PiecewiseBoundary:
    switch_times: tuple[float, ...]
    values: tuple[BoundaryValues, ...]

    def __post_init__(self):
        object.__setattr__(self, 'switch_times', tuple(self.switch_times))
        object.__setattr__(self, 'values', tuple(self.values))
        if len(self.values) != len(self.switch_times)+1 or not all(isinstance(b,BoundaryValues) for b in self.values):
            raise ValueError('每段边界须提供一组BoundaryValues')
        last = 0.
        for t in self.switch_times:
            finite_real(t, '切换时刻', positive=True)
            if t <= last:
                raise ValueError('切换时刻必须严格递增')
            last = t

    def at(self, time):
        finite_real(time, '时间', nonnegative=True)
        return self.values[bisect_right(self.switch_times,time)]
