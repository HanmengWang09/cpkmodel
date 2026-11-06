"""R01—R08的容积状态实现；纯当前相位，无肌压时间数值微分。

方程来源：用户提供Cardio-respiratory-model-main/source_files/lungMechanics.cc。
源作者Bianca Maria Laudenzi、Caterina Dalmaso、Lucas Omar Muller (2025)，Apache-2.0。
项目采用已汇总的容积状态等价形式；来源与哈希见reference_parameters.json。
"""
from dataclasses import dataclass
import numpy as np
from ..circulation.parameters import finite_real
from .parameters import CMH2O_TO_MMHG
from .boundary import RespiratoryBoundary


@dataclass
class RespiratoryEvaluation:
    derivative: np.ndarray
    jacobian: np.ndarray
    values: dict[str,float]
    partials: dict[str,np.ndarray]


def muscle_pressure(parameters,phase,amplitude):
    finite_real(phase,'呼吸累计相位')
    finite_real(amplitude,'肌压力幅度')
    if amplitude>0:raise ValueError('自主肌压力须非正')
    if amplitude==0:return 0.,0.
    p=parameters;q=phase-np.floor(phase)
    beta=1/(1+p.inspiratory_expiratory_ratio);alpha=1-beta
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:
            if q<=alpha:
                value=amplitude*q*(1-q)/(alpha*beta)
                derivative=amplitude*(1-2*q)/(alpha*beta)
            else:
                scale=p.expiration_tau_fraction*beta
                decay=np.exp(-(q-alpha)/scale);end=np.exp(-1/p.expiration_tau_fraction)
                denominator=-np.expm1(-1/p.expiration_tau_fraction)
                value=amplitude*(decay-end)/denominator
                derivative=-amplitude*decay/(scale*denominator)
        except (FloatingPointError,OverflowError,ZeroDivisionError) as exc:
            raise ValueError('呼吸肌驱动超出有限数值工作域') from exc
    if not np.all(np.isfinite([value,derivative])):raise ValueError('呼吸肌驱动或导数非有限')
    return float(value),float(derivative)


def reference_gas_state(parameters):
    p=parameters
    return np.array([p.laryngeal_unstressed,
        p.tracheal_unstressed-p.tracheal_compliance*p.pleural_end_expiratory,
        p.bronchial_unstressed-p.bronchial_compliance*p.pleural_end_expiratory,
        p.alveolar_unstressed-p.alveolar_compliance*p.pleural_end_expiratory,0.,0.])


def evaluate_respiration(parameters,state,boundary):
    x=np.asarray(state,dtype=float)
    if x.shape!=(6,) or not np.all(np.isfinite(x)) or np.any(x[:4]<=0):
        raise ValueError('呼吸需六有限状态与四正气室容积')
    if not isinstance(boundary,RespiratoryBoundary):raise ValueError('需RespiratoryBoundary')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:result=_evaluate(parameters,x,boundary)
        except (FloatingPointError,OverflowError,ZeroDivisionError) as exc:
            raise ValueError('呼吸代数超出有限数值工作域，不裁剪气室') from exc
    if not all(np.all(np.isfinite(v)) for v in (result.derivative,result.jacobian,list(result.values.values()),*result.partials.values())):
        raise ValueError('呼吸代数或Jacobian非有限')
    return result


def _evaluate(p,x,b):
    eye=np.eye(6);pmus,dpmus=muscle_pressure(p,x[5],b.muscle_pressure_min)
    ppl=x[4]/p.chest_wall_compliance+pmus+p.pleural_end_expiratory
    gp=eye[4]/p.chest_wall_compliance+dpmus*eye[5]
    c=np.array([p.laryngeal_compliance,p.tracheal_compliance,p.bronchial_compliance,p.alveolar_compliance])
    vu=np.array([p.laryngeal_unstressed,p.tracheal_unstressed,p.bronchial_unstressed,p.alveolar_unstressed])
    pressures=(x[:4]-vu)/c+np.array([0,1,1,1])*ppl
    gradients=eye[:4]/c[:,None]+np.array([0,1,1,1])[:,None]*gp
    resistance=np.array([p.mouth_resistance,p.larynx_trachea_resistance,p.trachea_bronchi_resistance,p.bronchi_alveoli_resistance])
    q=np.r_[-pressures[0],pressures[:-1]-pressures[1:]]/resistance
    gq=np.vstack((-gradients[0],gradients[:-1]-gradients[1:]))/resistance[:,None]
    derivative=np.r_[q[:3]-q[1:],q[3],q[1],b.respiratory_rate/60]
    jacobian=np.vstack((gq[:3]-gq[1:],gq[3],gq[1],np.zeros(6)))
    values=dict(zip(('p_l','p_tr','p_br','p_A'),map(float,pressures)))
    partials=dict(zip(('p_l','p_tr','p_br','p_A'),gradients))
    values.update(zip(('q_mouth','q_lt','q_tb','q_bA'),map(float,q)))
    partials.update(zip(('q_mouth','q_lt','q_tb','q_bA'),gq))
    values.update(p_mus=pmus,ppl_cmh2o=float(ppl),ppl_mmhg=float(CMH2O_TO_MMHG*ppl),
        V_gas=float(x[:4].sum()),V_dead=float(x[:3].sum()),chest_invariant=float(x[4]-x[1:4].sum()),
        p_abd=float(-x[4]/p.abdominal_compliance-pmus+p.abdominal_end_expiratory))
    partials.update(p_mus=dpmus*eye[5],ppl_cmh2o=gp,ppl_mmhg=CMH2O_TO_MMHG*gp,
        V_gas=eye[:4].sum(axis=0),V_dead=eye[:3].sum(axis=0),chest_invariant=eye[4]-eye[1:4].sum(axis=0),
        p_abd=-eye[4]/p.abdominal_compliance-dpmus*eye[5])
    return RespiratoryEvaluation(derivative,jacobian,values,partials)
