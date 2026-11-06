"""自主呼吸输入；fluid房压在本阶段表示呼气末参考表压。"""
from bisect import bisect_right
from dataclasses import dataclass,field
from ..circulation.parameters import finite_real
from ..fluid import FluidBoundary


@dataclass(frozen=True)
class RespiratoryBoundary:
    fluid: FluidBoundary = field(default_factory=FluidBoundary)
    respiratory_rate: float = 12.
    muscle_pressure_min: float = -5.

    def __post_init__(self):
        if not isinstance(self.fluid,FluidBoundary):raise ValueError('呼吸输入需包含FluidBoundary')
        finite_real(self.respiratory_rate,'呼吸频率',positive=True)
        finite_real(self.muscle_pressure_min,'肌压力幅度')
        if self.muscle_pressure_min>0:raise ValueError('自主吸气肌压力须非正；0表示关闭肌驱动')


@dataclass(frozen=True)
class RespiratoryProtocol:
    switch_times: tuple[float,...]
    values: tuple[RespiratoryBoundary,...]

    def __post_init__(self):
        object.__setattr__(self,'switch_times',tuple(self.switch_times))
        object.__setattr__(self,'values',tuple(self.values))
        if len(self.values)!=len(self.switch_times)+1 or not all(isinstance(b,RespiratoryBoundary) for b in self.values):
            raise ValueError('每段需一组RespiratoryBoundary')
        previous=0.
        for time in self.switch_times:
            finite_real(time,'切换时刻',positive=True)
            if time<=previous:raise ValueError('切换时刻须严格递增')
            previous=time

    def at(self,time):
        finite_real(time,'时间',nonnegative=True)
        return self.values[bisect_right(self.switch_times,time)]
