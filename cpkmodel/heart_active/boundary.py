"""第6步外界输入；房压完全由心脏状态计算。"""
from dataclasses import dataclass
from bisect import bisect_right
from ..circulation.parameters import finite_real
from ..fluid.boundary import FluidBoundary
from ..respiratory.boundary import RespiratoryBoundary

@dataclass(frozen=True)
class ActiveBoundary:
    water_input: float=1500/86400
    sodium_input: float=150/86400
    filtration_scale: float=1.
    water_excretion_shift: float=0.
    sodium_excretion_shift: float=0.
    respiratory_rate: float=12.
    muscle_pressure_min: float=-5.

    def __post_init__(self):
        for k,v in vars(self).items():finite_real(v,k,positive=k=='respiratory_rate',nonnegative=k=='filtration_scale')
        if self.muscle_pressure_min>0:raise ValueError('自主肌压力须非正')

    def respiratory(self,ra=0.,la=0.):
        fluid=FluidBoundary(ra,la,self.water_input,self.sodium_input,self.filtration_scale,self.water_excretion_shift,self.sodium_excretion_shift)
        return RespiratoryBoundary(fluid,self.respiratory_rate,self.muscle_pressure_min)

@dataclass(frozen=True)
class ActiveProtocol:
    switch_times: tuple
    values: tuple
    def __post_init__(self):
        object.__setattr__(self,'switch_times',tuple(self.switch_times));object.__setattr__(self,'values',tuple(self.values))
        if len(self.values)!=len(self.switch_times)+1 or not all(isinstance(x,ActiveBoundary) for x in self.values):raise ValueError('外界输入段数非法')
        last=0.
        for t in self.switch_times:
            finite_real(t,'切换时刻',positive=True)
            if t<=last:raise ValueError('切换时刻须递增')
            last=t
    def at(self,t):
        finite_real(t,'时间',nonnegative=True)
        return self.values[bisect_right(self.switch_times,t)]
