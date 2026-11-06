"""心脏被动参考；公开容积mL/面积cm2，几何核显式转换SI。"""
from dataclasses import dataclass
from importlib.resources import files
import json
from ..circulation.parameters import finite_real

PA_PER_MMHG=133.322368
WALL_NAMES=('LA','RA','LW','SW','RW')


@dataclass(frozen=True)
class WallParameters:
    name: str
    wall_volume_ml: float
    reference_area_cm2: float
    stress_passive_pa: float
    stress_scale_pa: float
    sarcomere_reference_um: float = 2.
    sarcomere_zero_um: float = 1.8
    titin_length_um: float = .6
    ecm_exponent: float = 10.
    titin_fraction: float = .01

    def __post_init__(self):
        if self.name not in WALL_NAMES:raise ValueError('心壁须为LA/RA/LW/SW/RW')
        for name,value in vars(self).items():
            if name!='name':finite_real(value,name,positive=name!='titin_fraction',nonnegative=name=='titin_fraction')


@dataclass(frozen=True)
class PassiveParameters:
    walls: tuple[WallParameters,...]
    metadata: dict
    pericardial_reference_pa: float = 1000.
    pericardial_reference_ml: float = 542.67
    pericardial_exponent: float = 10.
    maximum_geometry_z: float = .6
    minimum_tension: float = 1e-8
    maximum_condition: float = 1e8

    def __post_init__(self):
        object.__setattr__(self,'walls',tuple(self.walls))
        if tuple(w.name for w in self.walls)!=WALL_NAMES:raise ValueError('心壁顺序须为LA/RA/LW/SW/RW')
        for name,value in vars(self).items():
            if name in ('walls','metadata'):continue
            finite_real(value,name,positive=name!='pericardial_reference_pa',nonnegative=name=='pericardial_reference_pa')
        if self.maximum_geometry_z>=.95:raise ValueError('接受几何域须小于工作解析域.95')


def load_reference_parameters():
    r=json.loads(files(__package__).joinpath('reference_parameters.json').read_text(encoding='utf-8'))
    return PassiveParameters(tuple(WallParameters(**w) for w in r['walls']),r['metadata'],**r['global'])
