"""五壁主动心脏压力、肌节长度及14输入隐函数偏导。"""
from dataclasses import dataclass
import math
import numpy as np
from .numeric import wall,geometry,newton
from ..heart_passive.parameters import PA_PER_MMHG

@dataclass
class HeartEvaluation:
    pressures: np.ndarray
    pressure_partials: np.ndarray
    lengths: np.ndarray
    length_partials: np.ndarray
    residual: np.ndarray
    condition: float
    coordinates: np.ndarray
    walls: tuple
    pericardial_pressure: float
    iterations: int

class ActiveHeart:
    def __init__(self,params):
        self.params=params
        self.array=np.array([[w.wall_volume_ml*1e-6,w.reference_area_cm2*1e-4,w.stress_passive_pa,
            w.stress_scale_pa,w.sarcomere_reference_um,w.sarcomere_zero_um,w.ecm_exponent,
            2*w.sarcomere_reference_um/w.titin_length_um,w.titin_fraction] for w in params.passive.walls])
        self.scales=np.r_[np.ones(2),np.full(3,params.ventricular_scale)]*params.active_scale

    def evaluate(self,volumes,lsi,c,*,pleural_pressure_mmhg=-3.6778,geometry_guess=None):
        v=np.asarray(volumes,dtype=float);li=np.asarray(lsi,dtype=float);cc=np.asarray(c,dtype=float)
        if v.shape!=(4,) or li.shape!=(5,) or cc.shape!=(5,) or not np.all(np.isfinite(np.r_[v,li,cc,pleural_pressure_mmhg])) or np.any(v<=0) or np.any(li<=0):
            raise ValueError('主动心脏需要正四容积、正五lsi及有限五C/胸膜压')
        p=self.params;soft=p.buckling_stress_pa if p.stress_mode=='smooth_tension' else 0.
        args=(self.array,v,li,cc,self.scales,p.series_length_um,p.intrinsic_zero_um,soft)
        initial=np.array([math.log(.0316/.03)+math.log((v[1]+v[3]+150)/410)/3,.181])
        if geometry_guess is not None:initial=np.asarray(geometry_guess,dtype=float)
        q,ok,it=newton(*args,initial)
        if not ok:
            for y in (.02,.03,.04):
                for vs in (-.2,0.,.2):
                    q,ok,it=newton(*args,np.array([math.log(y/.03),vs]))
                    if ok:break
                if ok:break
        if not ok:raise ValueError('主动TriSeg未找到有效平衡根')
        residual,jac,pv,gpv,lsg,vals,geom=geometry(*args,q)
        condition=float(np.linalg.cond(jac[:,:2]))
        if not np.isfinite(condition) or condition>=1e8 or np.any(vals[:,5]<=1e-8) or np.any(abs(vals[:,6])>p.maximum_z):
            raise ValueError('主动TriSeg根不在张力/曲率/条件数接受域')
        response=-np.linalg.solve(jac[:,:2],jac[:,2:])
        dp=np.zeros((4,14));length_partials=np.zeros((5,14));lengths=np.zeros(5);pressures=np.zeros(4)
        dp[[1,3]]=(gpv[:,2:]+gpv[:,:2]@response)/PA_PER_MMHG;pressures[[1,3]]=pv/PA_PER_MMHG
        length_partials[2:]=lsg[:,2:]+lsg[:,:2]@response;lengths[2:]=vals[:,0]
        all_values=np.zeros((5,9));all_geom=np.zeros((5,5));all_values[2:]=vals;all_geom[2:]=geom
        for n,i in enumerate((0,2)):
            r=(3*(v[i]*1e-6+.5*self.array[n,0])/(4*math.pi))**(1/3);a=4*math.pi*r*r;cur=1/r
            w,g=wall(self.array[n],a,cur,li[n],cc[n],self.scales[n],p.series_length_um,p.intrinsic_zero_um,soft)
            if not np.all(np.isfinite(w)) or w[5]<=1e-8 or abs(w[6])>p.maximum_z:raise ValueError('主动心房不在接受域')
            dr=1e-6/(4*math.pi*r*r);chain=np.zeros((4,14));chain[0,i]=8*math.pi*r*dr;chain[1,i]=-dr/r**2
            chain[2,4+n]=1;chain[3,9+n]=1;grad=g@chain
            pressures[i]=2*w[5]/r/PA_PER_MMHG;dp[i]=2*grad[5]/r/PA_PER_MMHG;dp[i,i]-=2*w[5]*dr/r**2/PA_PER_MMHG
            lengths[n]=w[0];length_partials[n]=grad[0];all_values[n]=w;all_geom[n]=[2*r,0.,v[i]*1e-6+.5*self.array[n,0],a,cur]
        bag=float(v.sum()+sum(w.wall_volume_ml for w in p.passive.walls))
        peri=p.passive.pericardial_reference_pa*(bag/p.passive.pericardial_reference_ml)**p.passive.pericardial_exponent/PA_PER_MMHG
        pressures+=pleural_pressure_mmhg+peri;dp[:,:4]+=p.passive.pericardial_exponent*peri/bag
        names=('ls_um','sigma_passive_Pa','sigma_active_Pa','sigma_raw_Pa','sigma_effective_Pa','tension_N_m','z','u','strain')
        walls=tuple(dict(name=p.passive.walls[i].name,**dict(zip(names,map(float,w))),
                        x_m=float(all_geom[i,0]),y_m=float(all_geom[i,1]),Vm_m3=float(all_geom[i,2]),
                        area_m2=float(all_geom[i,3]),curvature_inv_m=float(all_geom[i,4])) for i,w in enumerate(all_values))
        if not all(np.all(np.isfinite(z)) for z in (pressures,dp,lengths,length_partials)):raise ValueError('主动压力/导数非有限')
        return HeartEvaluation(pressures,dp,lengths,length_partials,residual,condition,q.copy(),walls,peri,it)
