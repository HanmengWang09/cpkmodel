"""H04—H07/H15—H17；纯被动材料与几何值、解析导数。"""
from dataclasses import dataclass
import math
import numpy as np
from ..circulation.parameters import finite_real


@dataclass
class FieldEvaluation:
    values: dict[str,float]
    partials: dict[str,np.ndarray]


def _finite(result):
    if not np.all(np.isfinite(list(result.values.values()))) or not all(np.all(np.isfinite(g)) for g in result.partials.values()):
        raise ValueError('被动材料或几何超出有限数值工作域')
    return result


def _exp_remainder(x):
    if abs(x)<1e-4:return x*x*(.5+x*(1/6+x*(1/24+x/120)))
    return math.expm1(x)-x


def evaluate_material(p,ls_um):
    finite_real(ls_um,'肌节长度μm',positive=True)
    try:
        d=math.log(ls_um/p.sarcomere_zero_um);k=p.ecm_exponent;t=2*p.sarcomere_reference_um/p.titin_length_um
        sf,st=p.stress_passive_pa,p.titin_fraction*p.stress_scale_pa
        e,et=math.exp(k*d),math.exp(t*d)
        se=sf*math.expm1(k*d);ss=st*math.expm1(t*d)
        stiffness=k*sf*e+t*st*et;third=k*k*sf*e+t*t*st*et
        energy=sf*_exp_remainder(k*d)/k+st*_exp_remainder(t*d)/t
        return _finite(FieldEvaluation(dict(sigma=se+ss,sigma_ecm=se,sigma_titin=ss,stiffness=stiffness,
            energy_density=energy,active_stress=0.),dict(sigma=np.array([stiffness/ls_um]),
            sigma_ecm=np.array([k*sf*e/ls_um]),sigma_titin=np.array([t*st*et/ls_um]),
            stiffness=np.array([third/ls_um]),energy_density=np.array([(se+ss)/ls_um]),active_stress=np.zeros(1))))
    except (OverflowError,ZeroDivisionError,FloatingPointError) as exc:
        raise ValueError('被动材料超出有限工作域') from exc


def evaluate_cap(volume_m3,y_m):
    finite_real(volume_m3,'有符号中壁体积');finite_real(y_m,'交界半径',positive=True)
    try:
        x=2*y_m*math.sinh(math.asinh(3*volume_m3/(math.pi*y_m**3))/3)
        d=x*x+y_m*y_m;gx=np.array([2/(math.pi*d),-2*x*y_m/d]);gy=np.array([0.,1.])
        gd=2*x*gx+2*y_m*gy
        return _finite(FieldEvaluation(dict(x=x,A=math.pi*d,C=2*x/d),dict(x=gx,A=math.pi*gd,C=2*gx/d-2*x*gd/d**2)))
    except (OverflowError,ZeroDivisionError,FloatingPointError) as exc:
        raise ValueError('球冠几何超出有限工作域') from exc


def tension_factor(z):
    finite_real(z,'曲率厚度z')
    if abs(z)>=.95:raise ValueError('解析张力参照要求|z|<.95')
    if abs(z)<.01:
        f=1+sum(z**(2*n)/(2*n+1) for n in range(1,6))
        g=sum(2*n*z**(2*n-1)/(2*n+1) for n in range(1,6))
        return f,g
    a=math.atanh(z)
    return a/z,(z/(1-z*z)-a)/z**2


def _g0(z):
    if abs(z)>=.95:raise ValueError('应变解析参照要求|u|和|z|<.95')
    if abs(z)<.05:
        return (-sum(z**(2*n)/(n*(2*n+1)) for n in range(1,9)),
                -sum(2*z**(2*n-1)/(2*n+1) for n in range(1,9)))
    a=math.atanh(z)
    return 2*a/z+math.log1p(-z*z)-2,2/z-2*a/z**2


def evaluate_wall(p,area_m2,curvature_inv_m,*,mode='series'):
    finite_real(area_m2,'中壁面积m2',positive=True);finite_real(curvature_inv_m,'有符号曲率')
    if mode not in ('series','analytic_reference'):raise ValueError('未知被动几何模式')
    try:
        a,c=area_m2,curvature_inv_m;v=p.wall_volume_ml*1e-6
        z=1.5*v*c/a;u=3*v*c**3/(8*math.pi)
        if abs(z)>=.95 or abs(u)>=.95:raise ValueError('曲率超出被动几何工作域')
        ga=np.array([1.,0.]);gc=np.array([0.,1.]);gz=-z/a*ga+1.5*v/a*gc
        gu=9*v*c*c/(8*math.pi)*gc
        strain=.5*math.log(a/(p.reference_area_cm2*1e-4))
        if mode=='series':
            strain-=z*z/12+.019*z**4
            ge=.5/a*ga-(z/6+.076*z**3)*gz
            factor=1+z*z/3+z**4/5;gf=(2*z/3+4*z**3/5)*gz
        else:
            g,dg=_g0(z);h,dh=_g0(u)
            strain+=g/4-h/12;ge=.5/a*ga+dg/4*gz-dh/12*gu
            factor,df=tension_factor(z);gf=df*gz
        ls=p.sarcomere_reference_um*math.exp(strain);material=evaluate_material(p,ls)
        sigma=material.values['sigma'];pref=v/(2*a);tension=pref*sigma*factor
        gt=pref*(material.values['stiffness']*ge*factor+sigma*gf-sigma*factor/a*ga)
        values={**material.values,'strain':strain,'ls_um':ls,'tension':tension,'z':z,'u':u,'A':a,'C':c}
        partials={key:gradient[0]*ls*ge for key,gradient in material.partials.items()}
        partials.update(strain=ge,ls_um=ls*ge,tension=gt,z=gz,u=gu,A=ga,C=gc)
        return _finite(FieldEvaluation(values,partials))
    except (OverflowError,ZeroDivisionError,FloatingPointError) as exc:
        raise ValueError('心壁计算超出有限工作域') from exc
