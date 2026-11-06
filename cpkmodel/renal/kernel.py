"""K02—K06、固定排水分数的K12—K13及四压力解析偏导。"""
from dataclasses import dataclass
import numpy as np
from ..circulation.parameters import finite_real


@dataclass
class KidneyEvaluation:
    values: dict[str,float]
    # 顺序为瞬时入球/回流压力、感知入球/回流压力。
    partials: dict[str,np.ndarray]


def evaluate_renal(parameters,arterial_pressure,venous_pressure,sensed_arterial_pressure,
                   sensed_venous_pressure,filtration_scale=1.):
    p=parameters
    for name,value in [('入球压力',arterial_pressure),('回流压力',venous_pressure),
                       ('感知入球压力',sensed_arterial_pressure),('感知回流压力',sensed_venous_pressure)]:
        finite_real(value,name)
    finite_real(filtration_scale,'滤过容量比例',nonnegative=True)
    delta=sensed_arterial_pressure-sensed_venous_pressure
    if delta<=0 or arterial_pressure<venous_pressure:
        raise ValueError('肾工作域要求感知灌注差为正且瞬时血流不反向')
    ratio=(delta/p.reference_perfusion_pressure)**p.autoregulation_gain
    ra=p.afferent_resistance*np.clip(ratio,p.resistance_min,p.resistance_max)
    dra=np.zeros(4)
    if p.resistance_min<ratio<p.resistance_max:
        dra[2:]=[ra*p.autoregulation_gain/delta,-ra*p.autoregulation_gain/delta]
    pb=p.bowman_pressure+p.congestion_gain*max(sensed_venous_pressure-p.reference_venous_pressure,0)
    dpb=np.zeros(4)
    if sensed_venous_pressure>p.reference_venous_pressure:
        dpb[3]=p.congestion_gain
    threshold=pb+p.oncotic_factor*p.oncotic_pressure
    ga,ge=1/ra,1/p.efferent_resistance
    total_resistance=ra+p.efferent_resistance
    free_flow=(arterial_pressure-venous_pressure)/total_resistance
    free=venous_pressure+p.efferent_resistance*free_flow
    kg=p.filtration_coefficient*filtration_scale if free>threshold else 0.
    dga=-dra/ra**2
    unit_a,unit_v=np.eye(4)[0],np.eye(4)[1]
    if kg==0:
        # 直接用压差求无滤过串联支路：零压差必须精确得到零流量，避免加权平均舍入造成假反流。
        pg=free
        qa=qe=free_flow
        fg=0.
        dqa=dqe=(unit_a-unit_v-free_flow*dra)/total_resistance
        dpg=unit_v+p.efferent_resistance*dqa
        dfg=np.zeros(4)
    else:
        pg=(ga*arterial_pressure+ge*venous_pressure+kg*threshold)/(ga+ge+kg)
        dpg=(ga*unit_a+ge*unit_v+kg*dpb+(arterial_pressure-pg)*dga)/(ga+ge+kg)
        qa,qe=ga*(arterial_pressure-pg),ge*(pg-venous_pressure)
        fg=kg*(pg-threshold)
        dqa=ga*(unit_a-dpg)+(arterial_pressure-pg)*dga
        dqe=ge*(dpg-unit_v)
        dfg=kg*(dpg-dpb)
    if qa<0 or qe<0 or fg<0:
        raise ValueError('肾节点出现反向血流或负滤过，超出当前模型工作域')
    urine=p.excretion_fraction*fg
    durine=p.excretion_fraction*dfg
    values=dict(R_A=float(ra),P_B=float(pb),P_G=float(pg),Q_A=float(qa),Q_E=float(qe),
        GFR=float(fg),Q_urine=float(urine),Q_reabs=float(fg-urine),Q_return=float(qe+fg-urine),
        sensed_perfusion=float(delta),node_residual=float(qa-qe-fg))
    partials=dict(R_A=dra,P_B=dpb,P_G=dpg,Q_A=dqa,Q_E=dqe,GFR=dfg,Q_urine=durine,
        Q_reabs=dfg-durine,Q_return=dqe+dfg-durine,sensed_perfusion=np.array([0,0,1.,-1.]),node_residual=dqa-dqe-dfg)
    if not np.all(np.isfinite(list(values.values()))) or not all(np.all(np.isfinite(x)) for x in partials.values()):
        raise ValueError('肾节点计算超出有限数值范围')
    return KidneyEvaluation(values,partials)
