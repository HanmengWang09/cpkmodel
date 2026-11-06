"""16血管池的压力—容积与电阻网络；状态无历史缓存。"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from .boundary import BoundaryValues
from .parameters import finite_real


@dataclass
class Evaluation:
    pressures: np.ndarray
    flows: dict[str, float]
    derivative: np.ndarray
    boundary_flux: np.ndarray
    diagnostics: dict[str,float] | None = None


class VascularCircuit:
    state_size = 16
    boundary_flux_size = 4

    def __init__(self, params):
        self.params = params
        self.index = {n:i for i,n in enumerate(params.pool_names)}
        self.compliance = np.ones(16)
        self.unstressed = np.zeros(16)
        self.external = np.zeros(16)
        self.pleural_mask = np.zeros(16)
        for pool in params.linear_pools:
            i = self.index[pool.name]
            self.compliance[i] = pool.compliance
            self.unstressed[i] = pool.unstressed_volume
            self.external[i] = params.pleural_pressure if pool.thoracic else 0.
            self.pleural_mask[i] = float(pool.thoracic)
        self.pleural_mask[11] = 1.
        self.up = np.array([self.index[e.upstream] for e in params.resistors])
        self.down = np.array([self.index[e.downstream] for e in params.resistors])
        self.resistance = np.array([e.resistance for e in params.resistors])
        self.one_way = np.array([e.one_way for e in params.resistors],dtype=bool)
        self.incidence = np.zeros((16,19))
        self.incidence[self.up,np.arange(19)] = -1
        self.incidence[self.down,np.arange(19)] = 1

    @staticmethod
    def _volumes(volumes):
        v = np.asarray(volumes,dtype=float)
        if v.shape != (16,) or not np.all(np.isfinite(v)) or np.any(v<=0):
            raise ValueError('血管容积须为16个有限正数；不使用裁剪修复非法状态')
        return v

    def _pleural(self, pressure):
        if pressure is None:return self.params.pleural_pressure
        finite_real(pressure,'胸膜压')
        return pressure

    def thoracic_pressure(self, volume, pleural_pressure=None):
        t = self.params.thoracic_vein
        a = np.exp(-volume/t.kxv)
        den = -np.expm1(-volume/t.kxv)
        psi = t.kxp*a/den
        if volume >= t.unstressed_volume:
            base = t.d1+t.k1*(volume-t.unstressed_volume)
            slope = t.k1
        else:
            expv = np.exp(volume/t.vmin)
            base = t.d2+t.k2*expv
            slope = t.k2*expv/t.vmin
        return self._pleural(pleural_pressure)+base-psi, slope+t.kxp*a/(t.kxv*den**2)

    def thoracic_resistance(self, volume):
        t = self.params.thoracic_vein
        return t.kr*(t.vmax/volume)**2+t.r0, -2*t.kr*t.vmax**2/volume**3

    def _pressures(self, v, pleural_pressure=None):
        pp = self._pleural(pleural_pressure)
        p = (v-self.unstressed)/self.compliance+self.external+self.pleural_mask*(pp-self.params.pleural_pressure)
        slopes = 1/self.compliance
        p[11],slopes[11] = self.thoracic_pressure(v[11],pp)
        return p,slopes

    def _drains(self, v, p, slopes, b):
        rt,drt = self.thoracic_resistance(v[11])
        drop_t = p[11]-b.right_atrial_pressure
        drop_p = p[14]-b.left_atrial_pressure
        rp = self.params.pulmonary_outflow_resistance
        qt,qp = max(drop_t,0)/rt,max(drop_p,0)/rp
        dt = slopes[11]/rt-drop_t*drt/rt**2 if drop_t>0 else 0.
        dp = slopes[14]/rp if drop_p>0 else 0.
        return qt,qp,dt,dp

    def evaluate(self, volumes, boundary: BoundaryValues, *, pleural_pressure=None):
        v = self._volumes(volumes)
        p,slopes = self._pressures(v,pleural_pressure)
        drops = p[self.up]-p[self.down]
        q = np.where(self.one_way,np.maximum(drops,0),drops)/self.resistance
        qt,qp,_,_ = self._drains(v,p,slopes,boundary)
        dv = self.incidence@q
        dv[0] += boundary.systemic_inflow
        dv[11] -= qt
        dv[12] += boundary.pulmonary_inflow
        dv[14] -= qp
        flows = dict(zip((e.name for e in self.params.resistors),q))
        flows.update(TV_RA=qt,PV_LA=qp)
        flux = np.array([boundary.systemic_inflow,qt,boundary.pulmonary_inflow,qp])
        return Evaluation(p,flows,dv,flux)

    def jacobian(self, volumes, boundary, *, pleural_pressure=None):
        v = self._volumes(volumes)
        p,slopes = self._pressures(v,pleural_pressure)
        active = (~self.one_way)|((p[self.up]-p[self.down])>0)
        conductance = active/self.resistance
        jac = -(self.incidence*conductance)@self.incidence.T*slopes[None,:]
        _,_,dt,dp = self._drains(v,p,slopes,boundary)
        jac[11,11] -= dt
        jac[14,14] -= dp
        return jac

    def outflow_slopes(self, volumes, boundary, *, pleural_pressure=None):
        v = self._volumes(volumes)
        p,slopes = self._pressures(v,pleural_pressure)
        return self._drains(v,p,slopes,boundary)[2:]

    def validate_state(self, state):
        return self._volumes(state)

    def ledger_jacobian(self, state, boundary, *, pleural_pressure=None):
        jac = np.zeros((4,16))
        dt,dp = self.outflow_slopes(state,boundary,pleural_pressure=pleural_pressure)
        jac[1,11],jac[3,14] = dt,dp
        return jac

    def pressure_response(self, state, boundary, *, pleural_pressure, atrial_transmission=0.):
        """固定容积时，对共同胸膜压的RHS/四通量解析偏导。"""
        finite_real(atrial_transmission,'房压外压传递',nonnegative=True)
        if atrial_transmission>1:raise ValueError('房压外压传递须在[0,1]')
        v=self._volumes(state);p,_=self._pressures(v,pleural_pressure);mask=self.pleural_mask
        active=(~self.one_way)|((p[self.up]-p[self.down])>0)
        dq=active*(mask[self.up]-mask[self.down])/self.resistance
        dt=(mask[11]-atrial_transmission)/self.thoracic_resistance(v[11])[0] if p[11]>boundary.right_atrial_pressure else 0.
        dp=(mask[14]-atrial_transmission)/self.params.pulmonary_outflow_resistance if p[14]>boundary.left_atrial_pressure else 0.
        rhs=self.incidence@dq;rhs[11]-=dt;rhs[14]-=dp
        return rhs,np.array([0.,dt,0.,dp])

    def equilibrium(self, boundary):
        """由已知入流、房压和冻结电阻网络求一致种子；不执行参数拟合。"""
        qs,qp = boundary.systemic_inflow,boundary.pulmonary_inflow
        if qs<=0 or qp<=0:
            raise ValueError('自动平衡种子要求两入流为正；零流情况请显式提供正初值')
        def residual(v):
            return self.thoracic_pressure(v)[0]-boundary.right_atrial_pressure-self.thoracic_resistance(v)[0]*qs
        upper = 1000.
        while residual(upper)<0 and upper<1e7:
            upper*=2
        vt = brentq(residual,1e-6,upper,xtol=1e-11)
        p = np.zeros(16)
        p[11] = self.thoracic_pressure(vt)[0]
        rs = {e.name:e.resistance for e in self.params.resistors}
        totals = {j:rs['SA_'+j+'P']+rs[j+'P_'+j+'V']+rs[j+'V_TV'] for j in 'ESMHB'}
        p[0] = p[11]+qs/sum(1/r for r in totals.values())
        for j in totals:
            q = (p[0]-p[11])/totals[j]
            p[self.index[j+'P']] = p[0]-rs['SA_'+j+'P']*q
            p[self.index[j+'V']] = p[11]+rs[j+'V_TV']*q
        p[14] = boundary.left_atrial_pressure+qp*self.params.pulmonary_outflow_resistance
        total_cap,total_sh = rs['PA_PP']+rs['PP_PV'],rs['PA_PS']+rs['PS_PV']
        p[12] = p[14]+qp/(1/total_cap+1/total_sh)
        p[13] = p[14]+(p[12]-p[14])/total_cap*rs['PP_PV']
        p[15] = p[14]+(p[12]-p[14])/total_sh*rs['PS_PV']
        v = self.unstressed+self.compliance*(p-self.external)
        v[11] = vt
        return self._volumes(v).copy()
