"""心端守恒转接只需房压；外部质量输入为净水与净钠。"""
from bisect import bisect_right
from dataclasses import dataclass
from ..circulation.parameters import finite_real


@dataclass(frozen=True)
class FluidBoundary:
    right_atrial_pressure: float = 4.5
    left_atrial_pressure: float = 8.
    water_input: float = 1500/86400
    sodium_input: float = 150/86400
    filtration_scale: float = 1.
    water_excretion_shift: float = 0.
    sodium_excretion_shift: float = 0.

    def __post_init__(self):
        for name,value in vars(self).items():
            finite_real(value,name,nonnegative=name=='filtration_scale')


@dataclass(frozen=True)
class FluidProtocol:
    switch_times: tuple[float,...]
    values: tuple[FluidBoundary,...]

    def __post_init__(self):
        object.__setattr__(self,'switch_times',tuple(self.switch_times)); object.__setattr__(self,'values',tuple(self.values))
        if len(self.values)!=len(self.switch_times)+1 or not all(isinstance(b,FluidBoundary) for b in self.values):
            raise ValueError('每段需一组FluidBoundary')
        previous=0.
        for time in self.switch_times:
            finite_real(time,'切换时刻',positive=True)
            if time<=previous: raise ValueError('切换时刻须严格递增')
            previous=time

    def at(self,time):
        finite_real(time,'时间',nonnegative=True)
        return self.values[bisect_right(self.switch_times,time)]
