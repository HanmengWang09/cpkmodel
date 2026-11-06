"""C09、W01—W07、K07—K11的代数与解析梯度。"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from ..circulation.parameters import finite_real


@dataclass
class KernelEvaluation:
    values: dict[str,float]
    partials: dict[str,np.ndarray]


def evaluate_chemistry(parameters,state,*,pleural_pressure=None,cardiac_volume=None):
    if pleural_pressure is not None:finite_real(pleural_pressure,'胸膜压')
    if cardiac_volume is not None:finite_real(cardiac_volume,'额外心腔血量',nonnegative=True)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:
            result=_chemistry(parameters,state,pleural_pressure,cardiac_volume)
        except (FloatingPointError,OverflowError,ZeroDivisionError) as exc:
            raise ValueError('体液代数超出有限数值工作域，不裁剪水池') from exc
    if not np.all(np.isfinite(list(result.values.values()))) or not all(np.all(np.isfinite(g)) for g in result.partials.values()):
        raise ValueError('体液代数或导数出现非有限值')
    return result


def _chemistry(parameters,state,pleural_pressure=None,cardiac_volume=None):
    x=np.asarray(state,dtype=float); f=parameters.fluid
    if x.shape!=(24,) or not np.all(np.isfinite(x)) or np.any(x[:16]<=0) or np.any(x[20:]<=0):
        raise ValueError('体液需要24有限状态、正血管容积及正水钠储量')
    n=24 if cardiac_volume is None else 25
    eye=np.eye(n); vb=float(x[:16].sum())+(0. if cardiac_volume is None else cardiac_volume)
    dvb=np.r_[np.ones(16),np.zeros(n-16)]
    if cardiac_volume is not None:dvb[24]=1.
    vpl=vb-f.rbc_volume
    if vpl<=0: raise ValueError('血浆有效容积须大于0')
    ws,wp,wi,ne=x[20:]; ve=vpl+ws+wp; dve=dvb+eye[20]+eye[21]
    c=ne/ve; dc=eye[23]/ve-c/ve*dve
    pib=f.plasma_oncotic*f.plasma_reference/vpl
    pis=f.systemic_oncotic*f.systemic_reference/ws
    pip=f.pulmonary_oncotic*f.pulmonary_reference/wp
    oe=2*c; oi=f.intracellular_osmoles/wi
    values=dict(V_b=vb,V_pl=vpl,V_e=ve,W=ve+wi,C_Na=float(c),pi_b=float(pib),pi_Is=float(pis),pi_Ip=float(pip),
        P_Is=float(f.systemic_pressure+(ws-f.systemic_reference)/f.systemic_compliance),
        P_Ip=float((parameters.vascular.pleural_pressure if pleural_pressure is None else pleural_pressure)+f.pulmonary_tm_pressure+(wp-f.pulmonary_reference)/f.pulmonary_compliance),
        O_e=float(oe),O_i=float(oi),J_i=float(f.osmotic_conductance*(oi-oe)),H_eff=f.rbc_volume/vb)
    partials=dict(V_b=dvb,V_pl=dvb,V_e=dve,W=dve+eye[22],C_Na=dc,
        pi_b=-pib/vpl*dvb,pi_Is=-pis/ws*eye[20],pi_Ip=-pip/wp*eye[21],
        P_Is=eye[20]/f.systemic_compliance,P_Ip=eye[21]/f.pulmonary_compliance,
        O_e=2*dc,O_i=-oi/wi*eye[22],J_i=f.osmotic_conductance*(-oi/wi*eye[22]-2*dc),H_eff=-f.rbc_volume/vb**2*dvb)
    return KernelEvaluation(values,partials)


def evaluate_bed(bed,p_up,p_down,p_interstitial,pi_b,pi_interstitial,reflection):
    inputs=[p_up,p_down,p_interstitial,pi_b,pi_interstitial,reflection]
    if not all(np.isfinite(inputs)): raise ValueError('交换床输入须有限')
    gi,go,k=1/bed.inlet_resistance,1/bed.outlet_resistance,bed.filtration_coefficient
    eye=np.eye(5); potential=p_interstitial+reflection*(pi_b-pi_interstitial)
    ds=eye[2]+reflection*(eye[3]-eye[4])
    pc=(gi*p_up+go*p_down+k*potential)/(gi+go+k)
    dpc=(gi*eye[0]+go*eye[1]+k*ds)/(gi+go+k)
    return KernelEvaluation(dict(P_c=float(pc),Q_in=float(gi*(p_up-pc)),Q_out=float(go*(pc-p_down)),J=float(k*(pc-potential))),
        dict(P_c=dpc,Q_in=gi*(eye[0]-dpc),Q_out=go*(dpc-eye[1]),J=k*(dpc-ds)))


def lymph_return(interstitial_pressure,venous_pressure,resistance,pump_pressure,maxflow):
    raw=(interstitial_pressure+pump_pressure-venous_pressure)/resistance
    return float(min(max(raw,0.),maxflow)),np.array([1.,-1.])/resistance if 0<raw<maxflow else np.zeros(2)


def evaluate_feedback(parameters,renal,pbar_a,pbar_v,sodium_concentration,a,h,water_shift=0.,sodium_shift=0.):
    f=parameters
    for name,value in [('barA',pbar_a),('barV',pbar_v),('CNa',sodium_concentration),('a',a),('h',h),('uW',water_shift),('uNa',sodium_shift)]:
        finite_real(value,name)
    delta=pbar_a-pbar_v
    if delta<=0 or sodium_concentration<=0: raise ValueError('反馈需要正灌注差和钠浓度')
    lp=np.log(delta/renal.reference_perfusion_pressure)
    eye=np.eye(5); dlp=(eye[0]-eye[1])/delta
    araw=-f.sodium_perfusion_gain*lp
    hraw=f.water_osmotic_gain*np.log(sodium_concentration/f.sodium_reference)-f.water_perfusion_gain*lp
    ai=float(np.clip(araw,-f.sodium_limit,f.sodium_limit)); hi=float(np.clip(hraw,-f.water_limit,f.water_limit))
    dai=-f.sodium_perfusion_gain*dlp if abs(araw)<f.sodium_limit else np.zeros(5)
    dhi=f.water_osmotic_gain/sodium_concentration*eye[2]-f.water_perfusion_gain*dlp if abs(hraw)<f.water_limit else np.zeros(5)
    fn=float(expit(logit(f.sodium_fraction_reference)+f.pressure_natriuresis_gain*lp-f.sodium_retention_gain*a+sodium_shift))
    fw=float(expit(logit(renal.excretion_fraction)-h+water_shift))
    return KernelEvaluation(dict(a_inf=ai,h_inf=hi,f_Na=fn,f_W=fw),
        dict(a_inf=dai,h_inf=dhi,f_Na=fn*(1-fn)*(f.pressure_natriuresis_gain*dlp-f.sodium_retention_gain*eye[3]),f_W=-fw*(1-fw)*eye[4]))
