"""主动心脏开发闭合；未核定映射以显式输入公开。"""
from dataclasses import dataclass, field
from ..circulation.parameters import finite_real
from ..heart_passive.parameters import PassiveParameters, load_reference_parameters as load_passive
from ..respiratory.parameters import load_reference_parameters as load_respiratory

@dataclass(frozen=True)
class ValveParameters:
    name: str
    area_cm2: float
    length_cm: float
    opening_rate: float
    closing_rate: float
    proximal_resistance: float = 0.
    density_g_cm3: float = 1.04
    minimum_area_fraction: float = 1e-5
    open_threshold: float = 0.
    close_threshold: float = 0.

    def __post_init__(self):
        for k,v in vars(self).items():
            if k=='name':continue
            finite_real(v,k,positive=k in ('area_cm2','length_cm','density_g_cm3','minimum_area_fraction'),
                        nonnegative=k in ('opening_rate','closing_rate','proximal_resistance'))
        if self.close_threshold>self.open_threshold or self.minimum_area_fraction>1:
            raise ValueError('瓣膜阈值或最小面积比例非法')

@dataclass(frozen=True)
class ActiveParameters:
    passive: PassiveParameters
    respiratory: object
    valves: tuple
    cycle: float = .85
    av_delay_s: float = .150025
    series_length_um: float = .04
    intrinsic_zero_um: float = 1.51
    ldcc: float = 4.
    active_scale: float = 1.
    ventricular_scale: float = 1.
    buckling_stress_pa: float = 100.
    stress_mode: str = 'smooth_tension'
    maximum_z: float = .8
    durations_s: tuple = (.15,.15,.425,.425,.425)
    rise_s: tuple = (.033,.033,.0584375,.0584375,.0584375)
    decay_s: tuple = (.0198,.0198,.0350625,.0350625,.0350625)
    vmax_um_s: tuple = (14.,14.,7.,7.,7.)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.stress_mode not in ('raw','smooth_tension'):raise ValueError('未知低张力处理')
        for k in ('cycle','series_length_um','intrinsic_zero_um','ldcc','buckling_stress_pa','maximum_z'):
            finite_real(getattr(self,k),k,positive=True)
        for k in ('av_delay_s','active_scale','ventricular_scale'):finite_real(getattr(self,k),k,nonnegative=True)
        if self.av_delay_s>=self.cycle or self.maximum_z>=.95:raise ValueError('激活延迟或接受几何域非法')
        if tuple(v.name for v in self.valves)!=('MV','AV','TV','PVL'):raise ValueError('瓣膜顺序须MV/AV/TV/PVL')
        for k in ('durations_s','rise_s','decay_s','vmax_um_s'):
            vals=tuple(getattr(self,k));object.__setattr__(self,k,vals)
            if len(vals)!=5:raise ValueError('五壁参数长度须为5')
            for v in vals:finite_real(v,k,positive=True)

def load_reference_parameters():
    valves=(ValveParameters('MV',8.6,1.,26.,53.3),ValveParameters('AV',4.05,1.,15.,15.,.005),
            ValveParameters('TV',4.96,1.,40.,53.3),ValveParameters('PVL',5.18,1.5,26.7,26.7,.001))
    return ActiveParameters(load_passive(),load_respiratory(),valves,metadata=dict(
        definition='开发闭合：有效时标及lse_iso显式给定；smooth_tension非原生屈曲',
        source_urls=['https://framework.circadapt.org/2409/userguide/components/patch/patch.html',
          'https://framework.circadapt.org/2409/_modules/circadapt/model/vanosta2024.html'],
        valve_source='runs/coupling_tables_20261021/source_archive/Cardio-respiratory-model-main/test/params/cardiovascular.dat',
        inactive_mappings=['LDAD','ADO','native Timings','native buckling','native lse0 to lse_iso']))
