"""有符号TriSeg原始残差、解析导数及确定性多初猜阻尼Newton。"""
from dataclasses import dataclass
import math
import numpy as np
from .kernel import evaluate_cap,evaluate_wall
from .parameters import PA_PER_MMHG


class PassiveDomainError(ValueError):
    def __init__(self,message,diagnostics=()):
        super().__init__(message);self.diagnostics=tuple(diagnostics)


@dataclass
class GeometryEvaluation:
    residual: np.ndarray
    residual_jacobian: np.ndarray
    transmural_pressures_pa: np.ndarray
    pressure_partials_pa: np.ndarray
    walls: dict
    cap_volume_error_m3: float


@dataclass
class VentricularGeometry:
    y_m: float
    septal_volume_ml: float
    coordinates: np.ndarray
    internal_coordinates: np.ndarray
    residual: np.ndarray
    condition: float
    transmural_pressures: np.ndarray
    pressure_jacobian: np.ndarray
    walls: dict
    iterations: int
    attempts: tuple
    admissible_root_count: int
    geometry_mode: str
    cap_volume_error_m3: float


def residual_geometry(parameters,volumes_ml,coordinates,*,mode='series'):
    """输入(VLV,VRV)、q；导数列为q0/q1/VLV_mL/VRV_mL。"""
    v=np.asarray(volumes_ml,dtype=float);q=np.asarray(coordinates,dtype=float)
    if v.shape!=(2,) or q.shape!=(2,) or not np.all(np.isfinite(np.r_[v,q])) or np.any(v<=0):
        raise ValueError('TriSeg需两个正心室容积和两个有限内部坐标')
    try:y=.03*math.exp(q[0])
    except OverflowError as exc:raise ValueError('交界半径超出有限工作域') from exc
    if y<=0 or not np.isfinite(y):raise ValueError('交界半径超出正有限工作域')
    gy=np.array([y,0.,0.,0.]);vs=q[1]*1e-4;gvs=np.array([0.,1e-4,0.,0.])
    wl,ws,wr=parameters.walls[2:]
    volumes=[vs-(v[0]+.5*(wl.wall_volume_ml+ws.wall_volume_ml))*1e-6,vs,
             vs+(v[1]+.5*(wr.wall_volume_ml+ws.wall_volume_ml))*1e-6]
    gradients=[gvs-np.array([0.,0.,1e-6,0.]),gvs,gvs+np.array([0.,0.,0.,1e-6])]
    if volumes[0]>=0 or volumes[2]<=0:raise ValueError('左右自由壁方向与有符号几何不一致')
    walls={};tx=[];ty=[];gtx=[];gty=[];errors=[]
    for p,vm,gvm in zip((wl,ws,wr),volumes,gradients):
        cap=evaluate_cap(vm,y);cv=cap.values;cc=np.stack((gvm,gy))
        cg={key:g@cc for key,g in cap.partials.items()}
        wall=evaluate_wall(p,cv['A'],cv['C'],mode=mode);wv=wall.values
        wc=np.stack((cg['A'],cg['C']));wt={key:g@wc for key,g in wall.partials.items()}
        t=wv['tension'];s=cv['C']*y;co=1-cv['x']*cv['C']
        gs=cg['C']*y+cv['C']*gy;gco=-cg['x']*cv['C']-cv['x']*cg['C']
        tx.append(t*s);ty.append(t*co);gtx.append(wt['tension']*s+t*gs);gty.append(wt['tension']*co+t*gco)
        walls[p.name]={**wv,'x':cv['x'],'y':y,'Vm_m3':vm,'Tx':tx[-1],'Ty':ty[-1]}
        errors.append(abs(math.pi/6*cv['x']*(cv['x']**2+3*y*y)-vm))
    pressure=np.array([-2*tx[0]/y,2*tx[2]/y])
    gp=np.stack((-2*gtx[0]/y+2*tx[0]/y**2*gy,2*gtx[2]/y-2*tx[2]/y**2*gy))
    residual=np.array([sum(tx),sum(ty)]);jac=np.stack((np.sum(gtx,axis=0),np.sum(gty,axis=0)))
    if not all(np.all(np.isfinite(x)) for x in (residual,jac,pressure,gp)):
        raise ValueError('TriSeg残差或导数非有限')
    return GeometryEvaluation(residual,jac,pressure,gp,walls,max(errors))


def _newton(p,v,q,mode):
    q=np.array(q,dtype=float);last=None
    for iteration in range(61):
        try:e=residual_geometry(p,v,q,mode=mode)
        except ValueError as exc:return None,dict(status='工作域外',reason=str(exc),iterations=iteration)
        norm=float(np.max(np.abs(e.residual)));last=norm
        if norm<1e-10:
            condition=float(np.linalg.cond(e.residual_jacobian[:,:2]))
            admissible=all(w['tension']>p.minimum_tension and abs(w['z'])<=p.maximum_geometry_z for w in e.walls.values())
            if not admissible or not np.isfinite(condition) or condition>=p.maximum_condition:
                return None,dict(status='平衡根不在接受域',residual_N_m=norm,iterations=iteration)
            return (q,e,condition,iteration),dict(status='有效平衡根',residual_N_m=norm,iterations=iteration)
        if iteration==60:break
        try:step=np.linalg.solve(e.residual_jacobian[:,:2],e.residual)
        except np.linalg.LinAlgError:return None,dict(status='奇异几何Jacobian',iterations=iteration,residual_N_m=norm)
        if not np.all(np.isfinite(step)):return None,dict(status='非有限Newton步',iterations=iteration)
        # 约束的是数值步长，不裁剪物理状态；域外候选只回溯。
        step*=min(1.,1/max(float(np.max(np.abs(step))),1.))
        accepted=False
        for power in range(30):
            alpha=2.**(-power);candidate=q-alpha*step
            try:trial=residual_geometry(p,v,candidate,mode=mode)
            except ValueError:continue
            if np.max(np.abs(trial.residual))<norm*(1-1e-4*alpha):
                q=candidate;accepted=True;break
        if not accepted:return None,dict(status='回溯没有下降',iterations=iteration,residual_N_m=norm)
    return None,dict(status='达到迭代上限',iterations=60,residual_N_m=last)


def solve_ventricles(p,volumes_ml,*,mode='series',geometry_guess=None):
    v=np.asarray(volumes_ml,dtype=float)
    if v.shape!=(2,) or not np.all(np.isfinite(v)) or np.any(v<=0):raise ValueError('需两个正有限心室容积')
    guesses=[]
    if geometry_guess is not None:
        g=np.asarray(geometry_guess,dtype=float)
        if g.shape!=(2,) or not np.all(np.isfinite(g)) or g[0]<=0:raise ValueError('几何初猜须为正y(m)及有符号VmSW(mL)')
        guesses.append((float(g[0]),float(g[1])))
    guesses.extend((y,vs) for y in (.02,.03,.04) for vs in (-20.,0.,20.))
    roots=[];attempts=[]
    for y,vs in guesses:
        result,record=_newton(p,v,[math.log(y/.03),vs/100.],mode)
        record['initial_y_m']=y;record['initial_VmSW_ml']=vs;attempts.append(record)
        if result is not None and not any(np.max(np.abs(result[0]-old[0]))<1e-6 for old in roots):roots.append(result)
    if not roots:raise PassiveDomainError('未找到正张力且非退化的TriSeg平衡分支',attempts)
    if len(roots)>1:raise PassiveDomainError('多初猜找到多个有效平衡根，拒绝自动择根',attempts)
    q,e,condition,iterations=roots[0]
    response=-np.linalg.solve(e.residual_jacobian[:,:2],e.residual_jacobian[:,2:])
    derivative=(e.pressure_partials_pa[:,2:]+e.pressure_partials_pa[:,:2]@response)/PA_PER_MMHG
    y=.03*math.exp(q[0]);vs=q[1]*100.
    return VentricularGeometry(y,vs,np.array([y,vs]),q.copy(),e.residual.copy(),condition,
        e.transmural_pressures_pa/PA_PER_MMHG,derivative,e.walls,iterations,tuple(attempts),len(roots),mode,e.cap_volume_error_m3)
