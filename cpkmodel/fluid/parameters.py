"""第3步候选体液参数；水量mL、钠量mmol、压力mmHg、时间s。"""
from dataclasses import dataclass
from importlib.resources import files
import json
from ..circulation.parameters import CircuitParameters,LinearPool,Resistor,ThoracicVein,finite_real
from ..renal.parameters import RenalParameters


@dataclass(frozen=True)
class ExchangeBed:
    name: str
    inlet_resistance: float
    outlet_resistance: float
    filtration_coefficient: float
    lymph_resistance: float
    lymph_pump_pressure: float
    lymph_maxflow: float

    def __post_init__(self):
        if self.name not in tuple('ESMHB'): raise ValueError('交换床须为原五分区之一')
        for name,value in vars(self).items():
            if name=='name': continue
            finite_real(value,name,positive=name not in ('filtration_coefficient','lymph_pump_pressure','lymph_maxflow'),
                nonnegative=name in ('filtration_coefficient','lymph_maxflow'))


@dataclass(frozen=True)
class FluidParameters:
    rbc_volume: float
    plasma_reference: float
    systemic_reference: float
    pulmonary_reference: float = 200.
    intracellular_reference: float = 28000.
    systemic_pressure: float = -2.
    pulmonary_tm_pressure: float = -5.
    systemic_compliance: float = 1000.
    pulmonary_compliance: float = 30.
    plasma_oncotic: float = 25.
    systemic_oncotic: float = 5.
    pulmonary_oncotic: float = 10.
    systemic_reflection: float = 1.
    pulmonary_reflection: float = 1.
    pulmonary_filtration: float = .00035224911057099594
    pulmonary_lymph_resistance: float = 1000.
    pulmonary_lymph_pump: float = 20.
    pulmonary_lymph_maxflow: float = .01
    intracellular_osmoles: float = 7840.
    osmotic_conductance: float = 555.5555555555555

    def __post_init__(self):
        signed={'systemic_pressure','pulmonary_tm_pressure','pulmonary_lymph_pump'}
        nonnegative={'pulmonary_filtration','pulmonary_lymph_maxflow','osmotic_conductance','systemic_oncotic','pulmonary_oncotic','systemic_reflection','pulmonary_reflection'}
        for name,value in vars(self).items():
            finite_real(value,name,positive=name not in signed|nonnegative,nonnegative=name in nonnegative)
        if self.systemic_reflection>1 or self.pulmonary_reflection>1: raise ValueError('反射系数须在[0,1]')


@dataclass(frozen=True)
class FeedbackParameters:
    sodium_reference: float = .14
    sodium_fraction_reference: float = 150/86400/(2*.14)
    sodium_time_constant: float = 1800.
    water_time_constant: float = 300.
    sodium_perfusion_gain: float = 1.
    water_osmotic_gain: float = 20.
    water_perfusion_gain: float = 1.
    pressure_natriuresis_gain: float = 2.
    sodium_retention_gain: float = 1.
    sodium_limit: float = 2.
    water_limit: float = 4.

    def __post_init__(self):
        for name,value in vars(self).items():
            finite_real(value,name,positive=not name.endswith('gain'),nonnegative=name.endswith('gain'))
        if not 0<self.sodium_fraction_reference<1: raise ValueError('参考排钠分数须在(0,1)')


@dataclass(frozen=True)
class FluidCircuitParameters:
    vascular: CircuitParameters
    renal: RenalParameters
    fluid: FluidParameters
    feedback: FeedbackParameters
    beds: tuple[ExchangeBed,...]
    metadata: dict

    def __post_init__(self):
        object.__setattr__(self,'beds',tuple(self.beds))
        if tuple(b.name for b in self.beds)!=tuple('ESMHB'): raise ValueError('交换床顺序须为ESMHB')
        if not 0<self.renal.excretion_fraction<1: raise ValueError('反馈需要(0,1)内参考排水分数')


def load_reference_parameters():
    r=json.loads(files(__package__).joinpath('reference_parameters.json').read_text(encoding='utf-8'))
    v=r['vascular']
    vascular=CircuitParameters(tuple(LinearPool(**p) for p in v['linear_pools']),tuple(Resistor(**e) for e in v['resistors']),
        ThoracicVein(**v['thoracic_vein']),v['pleural_pressure'],v['pulmonary_outflow_resistance'],v['metadata'])
    return FluidCircuitParameters(vascular,RenalParameters(**r['renal']),FluidParameters(**r['fluid']),
        FeedbackParameters(**r['feedback']),tuple(ExchangeBed(**b) for b in r['beds']),r['metadata'])
