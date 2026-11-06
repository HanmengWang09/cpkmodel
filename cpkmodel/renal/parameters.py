"""第2步双肾合计参数；候选机制闭合，非患者拟合参数。"""
from dataclasses import dataclass
from importlib.resources import files
import json
from ..circulation.parameters import CircuitParameters,LinearPool,Resistor,ThoracicVein,finite_real


@dataclass(frozen=True)
class RenalParameters:
    afferent_resistance: float
    efferent_resistance: float
    reference_venous_pressure: float
    reference_perfusion_pressure: float
    filtration_coefficient: float = .2
    bowman_pressure: float = 15.
    oncotic_pressure: float = 25.
    oncotic_factor: float = 1.2
    congestion_gain: float = 1.
    autoregulation_gain: float = .5
    resistance_min: float = .5
    resistance_max: float = 2.
    pressure_time_constant: float = 10.
    excretion_fraction: float = 1500/86400/2

    def __post_init__(self):
        signed={'bowman_pressure','reference_venous_pressure'}
        nonnegative={'filtration_coefficient','congestion_gain','autoregulation_gain','excretion_fraction'}
        for name,value in vars(self).items():
            finite_real(value,name,positive=name not in signed|nonnegative,nonnegative=name in nonnegative)
        if not 0<self.resistance_min<=1<=self.resistance_max or self.excretion_fraction>=1:
            raise ValueError('阻力比例界须包含1，排水分数须在[0,1)')


@dataclass(frozen=True)
class RenalCircuitParameters:
    vascular: CircuitParameters
    renal: RenalParameters
    metadata: dict

    def __post_init__(self):
        if not isinstance(self.vascular,CircuitParameters) or not isinstance(self.renal,RenalParameters):
            raise ValueError('须提供血管参数和肾参数')


def load_reference_parameters():
    raw=json.loads(files(__package__).joinpath('reference_parameters.json').read_text(encoding='utf-8'))
    v=raw['vascular']
    vascular=CircuitParameters(tuple(LinearPool(**p) for p in v['linear_pools']),
        tuple(Resistor(**r) for r in v['resistors']),ThoracicVein(**v['thoracic_vein']),
        v['pleural_pressure'],v['pulmonary_outflow_resistance'],v['metadata'])
    return RenalCircuitParameters(vascular,RenalParameters(**raw['renal']),raw['metadata'])
