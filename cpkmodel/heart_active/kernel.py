"""H08—H14及源包V01—V05；返回解析输入导数。"""
from dataclasses import dataclass
import numpy as np
from ..heart_passive.kernel import FieldEvaluation

@dataclass
class FiberEvaluation:
    derivative: np.ndarray
    jacobian: np.ndarray

def fiber_rates(p,ls,lsi,c,phase):
    z=np.r_[ls,lsi,c,phase].astype(float)
    if z.shape!=(16,) or not np.all(np.isfinite(z)) or np.any(z[:10]<=0):raise ValueError('肌纤维输入须为有限五长度/五lsi/五C/相位')
    ls,li,c=z[:5],z[5:10],z[10:15]
    age=p.cycle*np.mod(z[15]-np.r_[np.zeros(2),np.full(3,p.av_delay_s/p.cycle)],1.)
    tr=np.array(p.rise_s);td=np.array(p.decay_s);dur=np.array(p.durations_s)
    u=np.clip(age/tr,0,8);rise=.02*u**3*(8-u)**2*np.exp(-u)
    dr=.02*np.exp(-u)*(3*u*u*(8-u)**2-2*u**3*(8-u)-u**3*(8-u)**2)
    dr*=((age/tr)>0)&((age/tr)<8)
    eta=(age-dur)/td;eta_clip=np.clip(eta,-np.pi/2,np.pi/2)
    decay=.5+.5*np.sin(eta_clip)
    dd=.5*np.cos(eta_clip)*((eta>-np.pi/2)&(eta<np.pi/2))
    length=np.tanh(p.ldcc*(li-p.intrinsic_zero_um)**2)
    dl=2*p.ldcc*(li-p.intrinsic_zero_um)*(1-length**2)
    vmax=np.array(p.vmax_um_s);rate=np.r_[vmax*((ls-li)/p.series_length_um-1),length*rise/tr-c*decay/td]
    jac=np.zeros((10,16));idx=np.arange(5)
    jac[idx,idx]=vmax/p.series_length_um;jac[idx,5+idx]=-vmax/p.series_length_um
    jac[5+idx,5+idx]=dl*rise/tr;jac[5+idx,10+idx]=-decay/td
    jac[5:,15]=p.cycle*(length*dr/tr**2-c*dd/td**2)
    return FiberEvaluation(rate,jac)

def valve(p,delta_pressure,q,xi):
    # 隐式迭代试算允许暂时超出[0,1]；接受态由ActiveCircuit单独严格核查。
    if not np.all(np.isfinite([delta_pressure,q,xi])):raise ValueError('瓣膜状态或压差非法')
    area=p.area_cm2*(p.minimum_area_fraction+(1-p.minimum_area_fraction)*xi)
    da=p.area_cm2*(1-p.minimum_area_fraction)
    if xi<0:
        # 仅为隐式域外试算延拓；0点值和导数连续，物理[0,1]仍严格沿原面积式。
        amin=p.area_cm2*p.minimum_area_fraction;den=1-da/amin*xi
        area=amin/den;da/=den*den
    drop=delta_pressure-p.proximal_resistance*q
    inertia=p.density_g_cm3*p.length_cm/area
    bernoulli=p.density_g_cm3/(2*area*area)
    chi=1333.22368
    qdot=(chi*drop-bernoulli*q*abs(q))/inertia
    gq=np.array([chi/inertia,(-chi*p.proximal_resistance-2*bernoulli*abs(q))/inertia,
                 (chi*drop+bernoulli*q*abs(q))*da/(p.density_g_cm3*p.length_cm)])
    if drop>=p.open_threshold:
        xidot=(1-xi)*p.opening_rate*(drop-p.open_threshold)
        gx=np.array([(1-xi)*p.opening_rate,-p.proximal_resistance*(1-xi)*p.opening_rate,-p.opening_rate*(drop-p.open_threshold)])
    elif drop<p.close_threshold:
        xidot=xi*p.closing_rate*(drop-p.close_threshold)
        gx=np.array([xi*p.closing_rate,-p.proximal_resistance*xi*p.closing_rate,p.closing_rate*(drop-p.close_threshold)])
    else:xidot=0.;gx=np.zeros(3)
    return FieldEvaluation(dict(q_dot=qdot,xi_dot=xidot,area_cm2=area,pressure_drop=drop),dict(q_dot=gq,xi_dot=gx))
