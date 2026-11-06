"""与解析核同式的只求值后端；无 fastmath、历史根缓存或状态裁剪。

参数按字段名打包成不可变 namedtuple，数值核不读取参考常数或隐藏数据。
解析核继续负责雅可比与观测，快核仅减少 RHS/独立通量中的导数构建。
"""
from collections import namedtuple
from functools import lru_cache
from dataclasses import fields
import math
import numpy as np
from numba import njit
from .numeric import newton,geometry,wall
from ..heart_passive.parameters import PA_PER_MMHG
from ..respiratory.parameters import CMH2O_TO_MMHG


@lru_cache(maxsize=None)
def record_type(name,names):
    cls=namedtuple(name,names)
    globals()[name]=cls
    return cls


def record(obj):
    names=tuple(f.name for f in fields(obj) if isinstance(getattr(obj,f.name),(int,float)))
    cls=record_type(type(obj).__name__+'Values',names)
    return cls(*(getattr(obj,n) for n in names))


@njit(cache=True)
def respiration(m,x,b):
    q=x[29]-math.floor(x[29]);beta=1/(1+m.inspiratory_expiratory_ratio);alpha=1-beta
    if b.muscle_pressure_min==0:pmus=0.
    elif q<=alpha:pmus=b.muscle_pressure_min*q*(1-q)/(alpha*beta)
    else:
        decay=math.exp(-(q-alpha)/(m.expiration_tau_fraction*beta))
        end=math.exp(-1/m.expiration_tau_fraction)
        pmus=b.muscle_pressure_min*(decay-end)/(-math.expm1(-1/m.expiration_tau_fraction))
    ppl=x[28]/m.chest_wall_compliance+pmus+m.pleural_end_expiratory
    p=np.array([(x[24]-m.laryngeal_unstressed)/m.laryngeal_compliance,
        (x[25]-m.tracheal_unstressed)/m.tracheal_compliance+ppl,
        (x[26]-m.bronchial_unstressed)/m.bronchial_compliance+ppl,
        (x[27]-m.alveolar_unstressed)/m.alveolar_compliance+ppl])
    flows=np.array([-p[0]/m.mouth_resistance,(p[0]-p[1])/m.larynx_trachea_resistance,
        (p[1]-p[2])/m.trachea_bronchi_resistance,(p[2]-p[3])/m.bronchi_alveoli_resistance])
    dx=np.array([flows[0]-flows[1],flows[1]-flows[2],flows[2]-flows[3],flows[3],flows[1],b.respiratory_rate/60])
    return dx,CMH2O_TO_MMHG*ppl,flows[0]


@njit(cache=True)
def fluid_values(f,k,g,t,beds,vascular,x,ppl,b):
    """C09/W01—W07/K02—K11；无房压依赖的内部项和九通量。"""
    vb=x[:16].sum()+x[30:34].sum();vpl=vb-f.rbc_volume
    if vpl<=0:raise ValueError('血浆有效容积须大于0')
    ws,wp,wi,ne=x[20:24];ve=vpl+ws+wp;cn=ne/ve
    pib=f.plasma_oncotic*f.plasma_reference/vpl
    pis=f.systemic_oncotic*f.systemic_reference/ws;pip=f.pulmonary_oncotic*f.pulmonary_reference/wp
    ps=f.systemic_pressure+(ws-f.systemic_reference)/f.systemic_compliance
    pp=ppl+f.pulmonary_tm_pressure+(wp-f.pulmonary_reference)/f.pulmonary_compliance
    ji=f.osmotic_conductance*(f.intracellular_osmoles/wi-2*cn)
    compliance,unstressed,external,mask,up,down,resistance,oneway,incidence,pleural,rp=vascular
    vp=(x[:16]-unstressed)/compliance+external+mask*(ppl-pleural)
    v=x[11];a=math.exp(-v/t.kxv);den=-math.expm1(-v/t.kxv)
    base=t.d1+t.k1*(v-t.unstressed_volume) if v>=t.unstressed_volume else t.d2+t.k2*math.exp(v/t.vmin)
    vp[11]=ppl+base-t.kxp*a/den
    q=np.empty(len(up))
    for i in range(len(q)):
        drop=vp[up[i]]-vp[down[i]]
        q[i]=(max(drop,0.) if oneway[i] else drop)/resistance[i]
    dx=np.zeros(24);dx[:16]=incidence@q
    js=qls=0.
    for i in range(5):
        bed=beds[i];gi=1/bed.inlet_resistance;go=1/bed.outlet_resistance;fil=bed.filtration_coefficient
        potential=ps+f.systemic_reflection*(pib-pis)
        pc=(gi*vp[i+1]+go*vp[i+6]+fil*potential)/(gi+go+fil)
        qi=gi*(vp[i+1]-pc);qo=go*(pc-vp[i+6]);j=fil*(pc-potential)
        lymph=min(max((ps+bed.lymph_pump_pressure-vp[i+6])/bed.lymph_resistance,0.),bed.lymph_maxflow)
        # 原交换支路在第 5..9 列；通过上下游匹配，避免依赖源文件顺序。
        old=0.
        for e in range(len(q)):
            if up[e]==i+1 and down[e]==i+6:old=q[e]
        dx[i+1]+=old-qi;dx[i+6]+=-old+qo+lymph;js+=j;qls+=lymph
    jp=f.pulmonary_filtration*(vp[13]-pp-f.pulmonary_reflection*(pib-pip))
    qlp=min(max((pp+f.pulmonary_lymph_pump-vp[6])/f.pulmonary_lymph_resistance,0.),f.pulmonary_lymph_maxflow)
    delta=x[16]-x[17]
    if delta<=0 or vp[1]<vp[6]:raise ValueError('肾工作域要求正感知灌注与非反向瞬时血流')
    lp=math.log(delta/k.reference_perfusion_pressure)
    ai=min(max(-g.sodium_perfusion_gain*lp,-g.sodium_limit),g.sodium_limit)
    hi=min(max(g.water_osmotic_gain*math.log(cn/g.sodium_reference)-g.water_perfusion_gain*lp,-g.water_limit),g.water_limit)
    zn=math.log(g.sodium_fraction_reference/(1-g.sodium_fraction_reference))+g.pressure_natriuresis_gain*lp-g.sodium_retention_gain*x[18]+b.sodium_excretion_shift
    zw=math.log(k.excretion_fraction/(1-k.excretion_fraction))-x[19]+b.water_excretion_shift
    fn=1/(1+math.exp(-zn)) if zn>=0 else math.exp(zn)/(1+math.exp(zn))
    fw=1/(1+math.exp(-zw)) if zw>=0 else math.exp(zw)/(1+math.exp(zw))
    ra=k.afferent_resistance*min(max((delta/k.reference_perfusion_pressure)**k.autoregulation_gain,k.resistance_min),k.resistance_max)
    pb=k.bowman_pressure+k.congestion_gain*max(x[17]-k.reference_venous_pressure,0.)
    threshold=pb+k.oncotic_factor*pib;ga=1/ra;ge=1/k.efferent_resistance
    freeflow=(vp[1]-vp[6])/(ra+k.efferent_resistance);free=vp[6]+k.efferent_resistance*freeflow
    fil=k.filtration_coefficient*b.filtration_scale if free>threshold else 0.
    if fil==0:qa=qe=freeflow;gfr=0.
    else:
        pg=(ga*vp[1]+ge*vp[6]+fil*threshold)/(ga+ge+fil)
        qa=ga*(vp[1]-pg);qe=ge*(pg-vp[6]);gfr=fil*(pg-threshold)
    if qa<0 or qe<0 or gfr<0:raise ValueError('肾节点出现反流或负滤过')
    urine=fw*gfr;una=fn*gfr*cn
    dx[1]-=qa;dx[6]+=qa-urine+b.water_input+qlp;dx[13]-=jp
    dx[16]=(vp[1]-x[16])/k.pressure_time_constant;dx[17]=(vp[6]-x[17])/k.pressure_time_constant
    dx[18]=(ai-x[18])/g.sodium_time_constant;dx[19]=(hi-x[19])/g.water_time_constant
    dx[20:24]=np.array([js-qls-ji,jp-qlp,ji,b.sodium_input-una])
    flux=np.array([b.water_input,urine,b.sodium_input,una,js,jp,qls,qlp,ji])
    return dx,flux,vp


@njit(cache=True)
def heart_values(array,scales,p,passive,x,ppl):
    v=x[30:34];li=x[34:39];cc=x[39:44]
    soft=p.buckling_stress_pa if p.stress_mode else 0.
    initial=np.array([math.log(.0316/.03)+math.log((v[1]+v[3]+150)/410)/3,.181])
    q,ok,it=newton(array,v,li,cc,scales,p.series_length_um,p.intrinsic_zero_um,soft,initial)
    if not ok:
        for y in (.02,.03,.04):
            for vs in (-.2,0.,.2):
                q,ok,it=newton(array,v,li,cc,scales,p.series_length_um,p.intrinsic_zero_um,soft,np.array([math.log(y/.03),vs]))
                if ok:break
            if ok:break
    if not ok:raise ValueError('主动TriSeg未找到有效平衡根')
    residual,jac,pv,gpv,lsg,vals,geom=geometry(array,v,li,cc,scales,p.series_length_um,p.intrinsic_zero_um,soft,q)
    condition=np.linalg.cond(jac[:,:2])
    if not np.isfinite(condition) or condition>=1e8 or np.any(vals[:,5]<=1e-8) or np.any(np.abs(vals[:,6])>p.maximum_z):
        raise ValueError('主动TriSeg根超出接受域')
    pressure=np.zeros(4);pressure[1]=pv[0]/PA_PER_MMHG;pressure[3]=pv[1]/PA_PER_MMHG
    lengths=np.empty(5);lengths[2:]=vals[:,0]
    for n in range(2):
        i=n*2;r=(3*(v[i]*1e-6+.5*array[n,0])/(4*math.pi))**(1/3)
        w,grad=wall(array[n],4*math.pi*r*r,1/r,li[n],cc[n],scales[n],p.series_length_um,p.intrinsic_zero_um,soft)
        if not np.all(np.isfinite(w)) or w[5]<=1e-8 or abs(w[6])>p.maximum_z:raise ValueError('主动心房不在接受域')
        pressure[i]=2*w[5]/r/PA_PER_MMHG;lengths[n]=w[0]
    bag=v.sum()+passive[0]
    peri=passive[1]*(bag/passive[2])**passive[3]/PA_PER_MMHG
    pressure+=ppl+peri
    return pressure,lengths


@njit(cache=True)
def rates_numeric(x,b,packed):
    f,k,g,t,beds,vascular,m,array,scales,p,passive,fiber,valves=packed
    rd,ppl,qmouth=respiration(m,x,b)
    fd,ff,vp=fluid_values(f,k,g,t,beds,vascular,x,ppl,b)
    hp,ls=heart_values(array,scales,p,passive,x,ppl)
    dx=np.zeros(53);dx[:24]=fd;dx[24:30]=rd
    qt=max(vp[11]-hp[2],0.)/(t.kr*(t.vmax/x[11])**2+t.r0)
    ql=max(vp[14]-hp[0],0.)/vascular[-1]
    dx[11]-=qt;dx[14]-=ql;dx[0]+=x[45];dx[12]+=x[47]
    dx[30:34]=np.array([ql-x[44],x[44]-x[45],qt-x[46],x[46]-x[47]])
    for i in range(5):
        dur,tr,td,vmax=fiber[i];delay=0. if i<2 else p.av_delay_s/p.cycle
        age=p.cycle*((x[52]-delay)%1.)
        u=min(max(age/tr,0.),8.);rise=.02*u**3*(8-u)**2*math.exp(-u)
        decay=.5+.5*math.sin(min(max((age-dur)/td,-math.pi/2),math.pi/2))
        length=math.tanh(p.ldcc*(x[34+i]-p.intrinsic_zero_um)**2)
        dx[34+i]=vmax*((ls[i]-x[34+i])/p.series_length_um-1)
        dx[39+i]=length*rise/tr-x[39+i]*decay/td
    delta=np.array([hp[0]-hp[1],hp[1]-vp[0],hp[2]-hp[3],hp[3]-vp[12]])
    for i in range(4):
        v=valves[i];q=x[44+i];xi=x[48+i]
        area=v.area_cm2*(v.minimum_area_fraction+(1-v.minimum_area_fraction)*xi)
        if xi<0:
            amin=v.area_cm2*v.minimum_area_fraction
            area=amin/(1-v.area_cm2*(1-v.minimum_area_fraction)/amin*xi)
        drop=delta[i]-v.proximal_resistance*q
        inertia=v.density_g_cm3*v.length_cm/area;bernoulli=v.density_g_cm3/(2*area*area)
        dx[44+i]=(1333.22368*drop-bernoulli*q*abs(q))/inertia
        if drop>=v.open_threshold:dx[48+i]=(1-xi)*v.opening_rate*(drop-v.open_threshold)
        elif drop<v.close_threshold:dx[48+i]=xi*v.closing_rate*(drop-v.close_threshold)
    dx[52]=1/p.cycle
    flux=np.empty(12);flux[:9]=ff;flux[9]=qmouth;flux[10]=b.respiratory_rate/60;flux[11]=1/p.cycle
    return dx,flux


@njit(cache=True)
def flux_numeric(x,b,packed):
    f,k,g,t,beds,vascular,m=packed[:7]
    rd,ppl,qmouth=respiration(m,x,b)
    fd,ff,vp=fluid_values(f,k,g,t,beds,vascular,x,ppl,b)
    flux=np.empty(12);flux[:9]=ff;flux[9]=qmouth;flux[10]=b.respiratory_rate/60;flux[11]=1/packed[9].cycle
    return flux


def pack(circuit):
    p=circuit.params;fp=p.respiratory.fluid;v=circuit.fluid.vascular
    vascular=(v.compliance,v.unstressed,v.external,v.pleural_mask,v.up,v.down,v.resistance,v.one_way,v.incidence,
        fp.vascular.pleural_pressure,fp.vascular.pulmonary_outflow_resistance)
    active=record(p)
    # stress_mode 是显式枚举，单独布尔字段不丢弃参数依赖。
    cls=record_type('ActiveNumeric',active._fields+('stress_mode',))
    active=cls(*active,p.stress_mode=='smooth_tension')
    passive=np.array([sum(w.wall_volume_ml for w in p.passive.walls),p.passive.pericardial_reference_pa,
        p.passive.pericardial_reference_ml,p.passive.pericardial_exponent])
    return (record(fp.fluid),record(fp.renal),record(fp.feedback),record(fp.vascular.thoracic_vein),
        tuple(record(b) for b in fp.beds),vascular,record(p.respiratory.mechanics),
        circuit.heart.array,circuit.heart.scales,active,passive,
        np.array([p.durations_s,p.rise_s,p.decay_s,p.vmax_um_s]).T.copy(),tuple(record(v) for v in p.valves))


def validate_working(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(53,) or not np.all(np.isfinite(x)) or np.any(x[:16]<=0) or np.any(x[20:28]<=0) or np.any(x[30:39]<=0) or x[16]<=x[17]:
        raise ValueError('53态存在非法容积、储量或非有限状态')
    return x
