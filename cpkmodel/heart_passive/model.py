"""四腔容积输入→被动压力及解析容积导数；尚不替换循环心端。"""
from dataclasses import dataclass
import math
import numpy as np
from ..circulation.parameters import finite_real
from .parameters import PA_PER_MMHG
from .kernel import evaluate_wall
from .geometry import PassiveDomainError,solve_ventricles


@dataclass
class PassiveEvaluation:
    volumes_ml: np.ndarray
    pressures: np.ndarray
    transmural_pressures: np.ndarray
    pressure_jacobian: np.ndarray
    pericardial_pressure_mmhg: float
    bag_volume_ml: float
    external_pressure_mmhg: float
    walls: dict
    geometry: object


class PassiveHeart:
    def __init__(self,params,*,geometry_mode='series'):
        if geometry_mode not in ('series','analytic_reference'):raise ValueError('未知被动几何模式')
        self.params=params;self.geometry_mode=geometry_mode

    def evaluate(self,volumes_ml,*,pleural_pressure_mmhg=-3.6778,geometry_guess=None):
        v=np.asarray(volumes_ml,dtype=float)
        if v.shape!=(4,) or not np.all(np.isfinite(v)) or np.any(v<=0):raise ValueError('四腔容积须依LA/LV/RA/RV给出正有限mL')
        finite_real(pleural_pressure_mmhg,'胸膜压mmHg')
        p=self.params;tm=np.zeros(4);jac=np.zeros((4,4));walls={}
        for i,w in zip((0,2),p.walls[:2]):
            r=(3*(v[i]+.5*w.wall_volume_ml)*1e-6/(4*math.pi))**(1/3)
            a,c=4*math.pi*r*r,1/r;wall=evaluate_wall(w,a,c,mode=self.geometry_mode)
            if wall.values['tension']<=p.minimum_tension or abs(wall.values['z'])>p.maximum_geometry_z:
                raise PassiveDomainError(f'{w.name}不在已验证正张力/曲率域，未启用屈曲修正')
            dr=1e-6/(4*math.pi*r*r);chain=np.array([8*math.pi*r*dr,-dr/(r*r)])
            tension=wall.values['tension'];dt=wall.partials['tension']@chain
            tm[i]=2*tension/r/PA_PER_MMHG;jac[i,i]=(2*dt/r-2*tension*dr/(r*r))/PA_PER_MMHG
            walls[w.name]={**wall.values,'radius_m':r,'x':2*r,'y':0.,'Vm_m3':(v[i]+.5*w.wall_volume_ml)*1e-6}
        geometry=solve_ventricles(p,v[[1,3]],mode=self.geometry_mode,geometry_guess=geometry_guess)
        tm[[1,3]]=geometry.transmural_pressures;jac[np.ix_([1,3],[1,3])]=geometry.pressure_jacobian
        walls.update(geometry.walls)
        bag=float(v.sum()+sum(w.wall_volume_ml for w in p.walls))
        try:peri=p.pericardial_reference_pa*(bag/p.pericardial_reference_ml)**p.pericardial_exponent/PA_PER_MMHG
        except OverflowError as exc:raise ValueError('心包压力超出有限工作域') from exc
        dpi=p.pericardial_exponent*peri/bag;external=pleural_pressure_mmhg+peri
        pressure=tm+external;jac+=dpi
        if not all(np.all(np.isfinite(z)) for z in (pressure,tm,jac,peri,bag)):
            raise ValueError('被动心腔压力或导数非有限')
        return PassiveEvaluation(v.copy(),pressure,tm,jac,peri,bag,external,walls,geometry)
