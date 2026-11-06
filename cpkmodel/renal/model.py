"""16血管池+2肾压力感知；肾节点无独立储量。"""
import numpy as np
from scipy.optimize import root
from ..circulation.model import VascularCircuit,Evaluation
from .boundary import RenalBoundary
from .kernel import evaluate_renal


class RenalCircuit:
    state_size=18
    boundary_flux_size=6

    def __init__(self,params):
        self.params=params
        self.vascular=VascularCircuit(params.vascular)
        self.index=self.vascular.index
        self._chain_scales=np.array([1/self.vascular.compliance[1],1/self.vascular.compliance[6],1.,1.])
        self._chain_columns=np.array([1,6,16,17])

    def validate_state(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(18,) or not np.all(np.isfinite(x)) or x[16]<=x[17]:
            raise ValueError('肾循环须提供18个有限状态，且感知灌注差为正')
        self.vascular.validate_state(x[:16])
        return x

    @staticmethod
    def _boundary(boundary):
        if not isinstance(boundary,RenalBoundary):
            raise ValueError('肾循环必须使用RenalBoundary明确净水输入和滤过容量')

    def _kidney(self,x,b):
        self._boundary(b)
        pressure,_=self.vascular._pressures(x[:16])
        return evaluate_renal(self.params.renal,pressure[1],pressure[6],x[16],x[17],b.filtration_scale)

    def evaluate(self,state,boundary):
        x=self.validate_state(state); self._boundary(boundary)
        e=self.vascular.evaluate(x[:16],boundary)
        k=evaluate_renal(self.params.renal,e.pressures[1],e.pressures[6],x[16],x[17],boundary.filtration_scale)
        dv=e.derivative.copy()
        dv[1]-=k.values['Q_A']
        dv[6]+=k.values['Q_return']+boundary.water_input
        sensed=(e.pressures[[1,6]]-x[16:])/self.params.renal.pressure_time_constant
        flux=np.r_[e.boundary_flux,boundary.water_input,k.values['Q_urine']]
        diagnostics={**k.values,'P_EP':float(e.pressures[1]),'P_EV':float(e.pressures[6]),
            'Pbar_EP':float(x[16]),'Pbar_EV':float(x[17]),'GFR_ml_min':60*k.values['GFR'],'RBF_ml_min':60*k.values['Q_A']}
        flows={**e.flows,**{name:k.values[name] for name in ('Q_A','Q_E','GFR','Q_reabs','Q_return','Q_urine')}}
        return Evaluation(e.pressures,flows,np.r_[dv,sensed],flux,diagnostics)

    def jacobian(self,state,boundary):
        x=self.validate_state(state); k=self._kidney(x,boundary)
        j=np.zeros((18,18)); j[:16,:16]=self.vascular.jacobian(x[:16],boundary)
        j[1,self._chain_columns]-=k.partials['Q_A']*self._chain_scales
        j[6,self._chain_columns]+=k.partials['Q_return']*self._chain_scales
        tau=self.params.renal.pressure_time_constant
        j[16,1],j[16,16]=self._chain_scales[0]/tau,-1/tau
        j[17,6],j[17,17]=self._chain_scales[1]/tau,-1/tau
        return j

    def ledger_jacobian(self,state,boundary):
        x=self.validate_state(state); k=self._kidney(x,boundary)
        j=np.zeros((6,18)); j[:4,:16]=self.vascular.ledger_jacobian(x[:16],boundary)
        j[5,self._chain_columns]=k.partials['Q_urine']*self._chain_scales
        return j

    def equilibrium(self,boundary):
        """只求状态平衡，不改变任何生理参数。"""
        self._boundary(boundary)
        seed=self.validate_state(self.params.metadata['reference_state']).copy()
        if np.max(np.abs(self.evaluate(seed,boundary).derivative))<1e-8:
            return seed
        solution=root(lambda x:self.evaluate(x,boundary).derivative,seed,
                      jac=lambda x:self.jacobian(x,boundary),method='hybr',options={'xtol':1e-10})
        state=self.validate_state(solution.x).copy()
        residual=np.max(np.abs(self.evaluate(state,boundary).derivative))
        if not solution.success or residual>=1e-8:
            raise RuntimeError(f'肾循环平衡状态求解失败：{solution.message}；残差{residual:g}')
        return state
