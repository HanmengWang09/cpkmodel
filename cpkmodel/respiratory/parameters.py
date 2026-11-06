"""源呼吸力学参数；气体容积L、压力cmH2O、时间s。"""
from dataclasses import dataclass
from importlib.resources import files
import json
from ..circulation.parameters import finite_real
from ..fluid.parameters import FluidCircuitParameters,load_reference_parameters as load_fluid

CMH2O_TO_MMHG=.73556


@dataclass(frozen=True)
class RespiratoryParameters:
    laryngeal_compliance: float = .00127
    tracheal_compliance: float = .00238
    bronchial_compliance: float = .0131
    alveolar_compliance: float = .2
    chest_wall_compliance: float = .2445
    abdominal_compliance: float = .183
    laryngeal_unstressed: float = .0344
    tracheal_unstressed: float = .00663
    bronchial_unstressed: float = .0187
    alveolar_unstressed: float = 1.263
    mouth_resistance: float = 1.021
    larynx_trachea_resistance: float = .3369
    trachea_bronchi_resistance: float = .3063
    bronchi_alveoli_resistance: float = .0817
    inspiratory_expiratory_ratio: float = .6
    expiration_tau_fraction: float = .2
    pleural_end_expiratory: float = -5.
    abdominal_end_expiratory: float = 2.45
    nominal_frc: float = 2.4

    def __post_init__(self):
        for name,value in vars(self).items():
            finite_real(value,name,positive=name not in ('pleural_end_expiratory','abdominal_end_expiratory'))


@dataclass(frozen=True)
class RespiratoryCircuitParameters:
    fluid: FluidCircuitParameters
    mechanics: RespiratoryParameters
    metadata: dict


def load_reference_parameters():
    r=json.loads(files(__package__).joinpath('reference_parameters.json').read_text(encoding='utf-8'))
    return RespiratoryCircuitParameters(load_fluid(),RespiratoryParameters(**r['mechanics']),r['metadata'])
