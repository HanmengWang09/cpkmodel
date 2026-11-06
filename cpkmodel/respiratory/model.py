"""30态呼吸—循环—肾—体液耦合；全部耦合导数解析计算。"""
import numpy as np
from ..circulation.model import Evaluation
from ..fluid import FluidCircuit
from .parameters import CMH2O_TO_MMHG
from .boundary import RespiratoryBoundary,RespiratoryProtocol
from .kernel import evaluate_respiration,reference_gas_state
from ..circulation.parameters import finite_real


def respiratory_breakpoints(protocol,phase,end_time,mechanics):
    """纯相位分段，不引用血液化学或任何血量定义。"""
    if not isinstance(protocol,RespiratoryProtocol):raise ValueError('呼吸需RespiratoryProtocol')
    finite_real(phase,'呼吸累计初相位',nonnegative=True)
    finite_real(end_time,'终点',positive=True)
    alpha=mechanics.inspiratory_expiratory_ratio/(1+mechanics.inspiratory_expiratory_ratio)
    external=[t for t in protocol.switch_times if t<end_time]
    stops=[0.]+external+[float(end_time)];cuts=list(external)
    for left,right in zip(stops[:-1],stops[1:]):
        rate=protocol.at(left).respiratory_rate/60;end_phase=phase+rate*(right-left)
        for cycle in range(int(np.floor(phase)),int(np.ceil(end_phase))+1):
            for offset in (0.,alpha):
                time=left+(cycle+offset-phase)/rate
                if left+1e-10<time<right-1e-10:cuts.append(time)
        phase=end_phase
    return sorted(set(cuts))


class RespiratoryCircuit:
    state_size=30
    boundary_flux_size=11
    balance_names=FluidCircuit.balance_names+('gas_volume','chest_wall_invariant','respiratory_phase')
    balance_units=FluidCircuit.balance_units+('L','L','cycle')

    def __init__(self,params):
        self.params=params;self.fluid=FluidCircuit(params.fluid);self.index=self.fluid.index
        self.balance_matrix=np.zeros((9,11));self.balance_matrix[:6,:9]=self.fluid.balance_matrix
        self.balance_matrix[6,9]=1.;self.balance_matrix[8,10]=1.

    def _working_state(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(30,) or not np.all(np.isfinite(x)) or np.any(x[24:28]<=0):
            raise ValueError('需30有限状态与四正气室容积')
        self.fluid.validate_state(x[:24])
        return x

    def validate_state(self,state):
        x=self._working_state(state)
        if x[29]<0:raise ValueError('初态及已接受状态的累计呼吸相位须非负')
        return x

    def reference_initial_state(self,boundary):
        if not isinstance(boundary,RespiratoryBoundary):raise ValueError('第4步需RespiratoryBoundary')
        p=self.params
        if abs(CMH2O_TO_MMHG*p.mechanics.pleural_end_expiratory-p.fluid.vascular.pleural_pressure)>1e-12:
            raise ValueError('呼气末胸膜压与冻结体液参考不一致，请显式提供30态initial_state')
        return self.validate_state(np.r_[self.fluid.equilibrium(boundary.fluid),reference_gas_state(p.mechanics)]).copy()

    def equilibrium(self,boundary):
        raise ValueError('呼吸驱动无静态相位平衡，请显式传入reference_initial_state或自定义initial_state')

    def balance_quantities(self,state):
        x=self.validate_state(state)
        return np.r_[self.fluid.balance_quantities(x[:24]),x[24:28].sum(),x[28]-x[25:28].sum(),x[29]]

    def _compute(self,state,boundary):
        # 周期RHS在任意有限相位有定义；不裁剪隐式误差估计的微小负试算相位。
        x=self._working_state(state)
        if not isinstance(boundary,RespiratoryBoundary):raise ValueError('第4步需RespiratoryBoundary')
        respiratory=evaluate_respiration(self.params.mechanics,x[24:],boundary)
        ppl=respiratory.values['ppl_mmhg']
        fluid,jf,lf=self.fluid.evaluate_with_pleural(x[:24],boundary.fluid,ppl)
        chain=np.zeros((25,30));chain[:24,:24]=np.eye(24);chain[24,24:]=respiratory.partials['ppl_mmhg']
        jac=np.zeros((30,30));jac[:24]=jf@chain;jac[24:,24:]=respiratory.jacobian
        ledger=np.zeros((11,30));ledger[:9]=lf@chain;ledger[9,24:]=respiratory.partials['q_mouth']
        shift=ppl-self.params.fluid.vascular.pleural_pressure
        diagnostics={**fluid.diagnostics,**respiratory.values,'phi_R':float(x[29]),'RR':boundary.respiratory_rate,
            'P_RA':boundary.fluid.right_atrial_pressure+shift,'P_LA':boundary.fluid.left_atrial_pressure+shift}
        e=Evaluation(fluid.pressures,fluid.flows,np.r_[fluid.derivative,respiratory.derivative],
            np.r_[fluid.boundary_flux,respiratory.values['q_mouth'],boundary.respiratory_rate/60],diagnostics)
        return e,jac,ledger

    def evaluate(self,state,boundary):return self._compute(state,boundary)[0]
    def jacobian(self,state,boundary):return self._compute(state,boundary)[1]
    def ledger_jacobian(self,state,boundary):return self._compute(state,boundary)[2]

    def integration_breakpoints(self,protocol,initial_state,end_time):
        """按累计相位解析插入周期首与吸气末；频率变化不重置相位。"""
        phase=float(self.validate_state(initial_state)[29])
        return respiratory_breakpoints(protocol,phase,end_time,self.params.mechanics)
