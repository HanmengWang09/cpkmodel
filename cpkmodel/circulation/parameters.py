"""阶段1血管参数；单位为 mL、mmHg、s。"""
from dataclasses import dataclass
from importlib.resources import files
import json
from math import isfinite
from numbers import Real

POOL_NAMES = ('SA','EP','SP','MP','HP','BP','EV','SV','MV','HV','BV','TV','PA','PP','PV','PS')


def finite_real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{name} 必须为有限实数')
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f'{name} 超出允许范围')


@dataclass(frozen=True)
class LinearPool:
    name: str
    compliance: float
    unstressed_volume: float
    thoracic: bool = False

    def __post_init__(self):
        finite_real(self.compliance, '顺应性', positive=True)
        finite_real(self.unstressed_volume, '无应力容积', nonnegative=True)
        if type(self.thoracic) is not bool:
            raise ValueError('胸腔外压标志thoracic必须为布尔值')


@dataclass(frozen=True)
class Resistor:
    name: str
    upstream: str
    downstream: str
    resistance: float
    one_way: bool = False

    def __post_init__(self):
        finite_real(self.resistance, '阻力', positive=True)
        if type(self.one_way) is not bool:
            raise ValueError('单向标志one_way必须为布尔值')


@dataclass(frozen=True)
class ThoracicVein:
    d1: float = .3855
    k1: float = .15
    d2: float = -5.
    k2: float = .4
    unstressed_volume: float = 130.
    vmin: float = 50.
    vmax: float = 350.
    kxp: float = 2.
    kxv: float = 8.
    kr: float = .001
    r0: float = .025

    def __post_init__(self):
        for key, value in vars(self).items():
            finite_real(value, key, positive=key not in ('d1','d2'))


@dataclass(frozen=True)
class CircuitParameters:
    linear_pools: tuple[LinearPool, ...]
    resistors: tuple[Resistor, ...]
    thoracic_vein: ThoracicVein
    pleural_pressure: float
    pulmonary_outflow_resistance: float
    metadata: dict
    pool_names: tuple[str, ...] = POOL_NAMES

    def __post_init__(self):
        object.__setattr__(self, 'linear_pools', tuple(self.linear_pools))
        object.__setattr__(self, 'resistors', tuple(self.resistors))
        if self.pool_names != POOL_NAMES:
            raise ValueError('阶段1必须保持固定16血管池顺序')
        if {p.name for p in self.linear_pools} != set(POOL_NAMES)-{'TV'} or len(self.linear_pools) != 15:
            raise ValueError('线性血池应恰好覆盖除TV外的15池')
        expected = {(a,b) for a,b in [('SA',j+'P') for j in 'ESMHB'] +
                    [(j+'P',j+'V') for j in 'ESMHB'] + [(j+'V','TV') for j in 'ESMHB'] +
                    [('PA','PP'),('PP','PV'),('PA','PS'),('PS','PV')]}
        if {(e.upstream,e.downstream) for e in self.resistors} != expected or len(self.resistors) != 19:
            raise ValueError('阶段1内部连接拓扑不完整')
        if any(e.name != e.upstream+'_'+e.downstream or e.one_way != (e.downstream=='TV') for e in self.resistors):
            raise ValueError('连接名称或单向规则错误')
        finite_real(self.pleural_pressure, '胸膜压')
        finite_real(self.pulmonary_outflow_resistance, '肺静脉流出阻力', positive=True)


def load_reference_parameters():
    raw = json.loads(files(__package__).joinpath('reference_parameters.json').read_text(encoding='utf-8'))
    return CircuitParameters(
        tuple(LinearPool(**p) for p in raw['linear_pools']),
        tuple(Resistor(**e) for e in raw['resistors']),
        ThoracicVein(**raw['thoracic_vein']), raw['pleural_pressure'],
        raw['pulmonary_outflow_resistance'], raw['metadata'])
