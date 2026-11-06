"""24状态候选体液反馈；纯当前状态及解析链式Jacobian。"""
from dataclasses import replace
import numpy as np
from ..circulation import BoundaryValues,VascularCircuit
from ..circulation.model import Evaluation
from ..renal.kernel import evaluate_renal
from .boundary import FluidBoundary
from .kernel import evaluate_chemistry,evaluate_bed,evaluate_feedback,lymph_return
from ..circulation.parameters import finite_real


class FluidCircuit:
    state_size=24
    boundary_flux_size=9
    balance_names=('total_water','sodium','blood_volume','systemic_interstitial','pulmonary_interstitial','intracellular')
    balance_units=('mL','mmol','mL','mL','mL','mL')

    def __init__(self,params):
        self.params=params; self.vascular=VascularCircuit(params.vascular); self.index=self.vascular.index
        self._resistors={e.name:e.resistance for e in params.vascular.resistors}
        self.balance_matrix=np.array([[1,-1,0,0,0,0,0,0,0],[0,0,1,-1,0,0,0,0,0],
            [1,-1,0,0,-1,-1,1,1,0],[0,0,0,0,1,0,-1,0,-1],
            [0,0,0,0,0,1,0,-1,0],[0,0,0,0,0,0,0,0,1]],dtype=float)

    def validate_state(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(24,) or not np.all(np.isfinite(x)) or x[16]<=x[17]:
            raise ValueError('体液循环须为24有限状态，感知灌注差须为正')
        evaluate_chemistry(self.params,x)
        return x

    @staticmethod
    def _boundary(boundary):
        if not isinstance(boundary,FluidBoundary): raise ValueError('第3步需FluidBoundary和守恒心端转接')

    def balance_quantities(self,state):
        x=self.validate_state(state); v=evaluate_chemistry(self.params,x).values
        return np.array([v['W'],x[23],v['V_b'],x[20],x[21],x[22]])

    def _compute(self,state,boundary,*,pleural_pressure=None,atrial_transmission=0.,cardiac_volume=None):
        x=self.validate_state(state) if cardiac_volume is None else np.asarray(state,dtype=float)
        if x.shape!=(24,) or not np.all(np.isfinite(x)) or x[16]<=x[17]:raise ValueError('体液状态非法')
        self._boundary(boundary)
        finite_real(atrial_transmission,'房压外压传递',nonnegative=True)
        if atrial_transmission>1:raise ValueError('房压外压传递须在[0,1]')
        n=25 if cardiac_volume is None else 28
        p=self.params; f=p.fluid; eye=np.eye(n)
        ppl=p.vascular.pleural_pressure if pleural_pressure is None else pleural_pressure
        ch=evaluate_chemistry(p,x,pleural_pressure=ppl,cardiac_volume=cardiac_volume); cv=ch.values
        # 第25列是独立胸膜压偏导，供第4步链式耦合；公开第3步Jacobian仍为24列。
        cg={}
        for key,g in ch.partials.items():
            gradient=np.zeros(n);gradient[:24]=g[:24];gradient[24]=float(key=='P_Ip')
            if cardiac_volume is not None:gradient[25]=g[24]
            cg[key]=gradient
        shift=atrial_transmission*(ppl-p.vascular.pleural_pressure)
        bb=BoundaryValues(0.,0.,boundary.right_atrial_pressure+shift,boundary.left_atrial_pressure+shift)
        base=self.vascular.evaluate(x[:16],bb,pleural_pressure=ppl)
        pressure,slopes=self.vascular._pressures(x[:16],ppl)
        gp=np.zeros((16,n));gp[:,:16]=np.diag(slopes);gp[:,24]=self.vascular.pleural_mask
        dx=np.r_[base.derivative,np.zeros(8)]
        jac=np.zeros((24,n));jac[:16,:16]=self.vascular.jacobian(x[:16],bb,pleural_pressure=ppl)
        heart=np.zeros((4,n));heart[:,:16]=self.vascular.ledger_jacobian(x[:16],bb,pleural_pressure=ppl)
        jac[:16,24],heart[:,24]=self.vascular.pressure_response(x[:16],bb,pleural_pressure=ppl,atrial_transmission=atrial_transmission)
        # 无储量的跨循环转接；两入房流出仍按各自房压计算。
        if cardiac_volume is None:
            dx[0]+=base.flows['PV_LA'];dx[12]+=base.flows['TV_RA']
            jac[0]+=heart[3];jac[12]+=heart[1]
        else:
            rt=self.vascular.thoracic_resistance(x[11])[0]
            jac[11,26]=1/rt if pressure[11]>bb.right_atrial_pressure else 0.
            jac[14,27]=1/p.vascular.pulmonary_outflow_resistance if pressure[14]>bb.left_atrial_pressure else 0.
        flows={**base.flows,'SA_in':base.flows['PV_LA'] if cardiac_volume is None else 0.,
               'PA_in':base.flows['TV_RA'] if cardiac_volume is None else 0.}
        diag={**cv};js=qls=0.;gjs=np.zeros(n);gqls=np.zeros(n)
        for i,bed in enumerate(p.beds):
            up,down=i+1,i+6;name=bed.name
            bed_eval=evaluate_bed(bed,pressure[up],pressure[down],cv['P_Is'],cv['pi_b'],cv['pi_Is'],f.systemic_reflection)
            bv,bg=bed_eval.values,bed_eval.partials
            chain=np.stack((gp[up],gp[down],cg['P_Is'],cg['pi_b'],cg['pi_Is']))
            grad={key:gradient@chain for key,gradient in bg.items()}
            lymph,dl=lymph_return(cv['P_Is'],pressure[down],bed.lymph_resistance,bed.lymph_pump_pressure,bed.lymph_maxflow)
            gl=dl[0]*cg['P_Is']+dl[1]*gp[down]
            old_name=name+'P_'+name+'V';old_q=base.flows[old_name]
            gold=(gp[up]-gp[down])/self._resistors[old_name]
            dx[up]+=old_q-bv['Q_in'];dx[down]+=-old_q+bv['Q_out']+lymph
            jac[up]+=gold-grad['Q_in'];jac[down]+=-gold+grad['Q_out']+gl
            del flows[old_name]
            flows.update({name+'_cap_in':bv['Q_in'],name+'_cap_out':bv['Q_out'],name+'_lymph':lymph})
            diag.update({'Pc_'+name:bv['P_c'],'J_'+name:bv['J'],'QL_'+name:lymph,
                'node_'+name:bv['Q_in']-bv['Q_out']-bv['J']})
            js+=bv['J'];gjs+=grad['J'];qls+=lymph;gqls+=gl
        jp=f.pulmonary_filtration*(pressure[13]-cv['P_Ip']-f.pulmonary_reflection*(cv['pi_b']-cv['pi_Ip']))
        gjp=f.pulmonary_filtration*(gp[13]-cg['P_Ip']-f.pulmonary_reflection*(cg['pi_b']-cg['pi_Ip']))
        qlp,dlp=lymph_return(cv['P_Ip'],pressure[6],f.pulmonary_lymph_resistance,f.pulmonary_lymph_pump,f.pulmonary_lymph_maxflow)
        gqlp=dlp[0]*cg['P_Ip']+dlp[1]*gp[6]
        feedback=evaluate_feedback(p.feedback,p.renal,x[16],x[17],cv['C_Na'],x[18],x[19],
            boundary.water_excretion_shift,boundary.sodium_excretion_shift)
        fv=feedback.values;fc=np.stack((eye[16],eye[17],cg['C_Na'],eye[18],eye[19]))
        fg={name:gradient@fc for name,gradient in feedback.partials.items()}
        # 只借用肾核的灌注/滤过；排水由本阶段动态fW计算。局部替换不改变冻结参数。
        renal=replace(p.renal,oncotic_pressure=cv['pi_b'],excretion_fraction=0.)
        kidney=evaluate_renal(renal,pressure[1],pressure[6],x[16],x[17],boundary.filtration_scale)
        kv=kidney.values;kc=np.stack((gp[1],gp[6],eye[16],eye[17]))
        kg={name:gradient@kc for name,gradient in kidney.partials.items()}
        ga,ge=1/kv['R_A'],1/renal.efferent_resistance
        filt=renal.filtration_coefficient*boundary.filtration_scale if kv['GFR']>0 else 0.
        dpi=renal.oncotic_factor*filt/(ga+ge+filt)
        kg['P_G']+=dpi*cg['pi_b'];kg['Q_A']-=ga*dpi*cg['pi_b'];kg['Q_E']+=ge*dpi*cg['pi_b']
        kg['GFR']+=filt*(dpi-renal.oncotic_factor)*cg['pi_b']
        urine=fv['f_W']*kv['GFR']; gu=fv['f_W']*kg['GFR']+kv['GFR']*fg['f_W']
        una=fv['f_Na']*kv['GFR']*cv['C_Na']
        guna=fg['f_Na']*kv['GFR']*cv['C_Na']+fv['f_Na']*kg['GFR']*cv['C_Na']+fv['f_Na']*kv['GFR']*cg['C_Na']
        qreturn=kv['Q_A']-urine
        dx[1]-=kv['Q_A'];jac[1]-=kg['Q_A']
        dx[6]+=qreturn+boundary.water_input+qlp;jac[6]+=kg['Q_A']-gu+gqlp
        dx[13]-=jp;jac[13]-=gjp
        tau=p.renal.pressure_time_constant
        dx[16:18]=(pressure[[1,6]]-x[16:18])/tau
        jac[16]=(gp[1]-eye[16])/tau;jac[17]=(gp[6]-eye[17])/tau
        dx[18]=(fv['a_inf']-x[18])/p.feedback.sodium_time_constant
        dx[19]=(fv['h_inf']-x[19])/p.feedback.water_time_constant
        jac[18]=(fg['a_inf']-eye[18])/p.feedback.sodium_time_constant
        jac[19]=(fg['h_inf']-eye[19])/p.feedback.water_time_constant
        dx[20:]=[js-qls-cv['J_i'],jp-qlp,cv['J_i'],boundary.sodium_input-una]
        jac[20]=gjs-gqls-cg['J_i'];jac[21]=gjp-gqlp;jac[22]=cg['J_i'];jac[23]=-guna
        flux=np.array([boundary.water_input,urine,boundary.sodium_input,una,js,jp,qls,qlp,cv['J_i']])
        fj=np.stack((np.zeros(n),gu,np.zeros(n),guna,gjs,gjp,gqls,gqlp,cg['J_i']))
        diag.update(kv);diag.update(fv)
        diag.update(Q_urine=urine,Q_reabs=kv['GFR']-urine,Q_return=qreturn,U_Na=una,
            U_Na_reabs=(1-fv['f_Na'])*kv['GFR']*cv['C_Na'],J_s=js,J_p=jp,Q_Ls=qls,Q_Lp=qlp,
            GFR_ml_min=60*kv['GFR'],RBF_ml_min=60*kv['Q_A'],C_Na_mmol_L=1000*cv['C_Na'],
            P_EP=float(pressure[1]),P_EV=float(pressure[6]),a=float(x[18]),h=float(x[19]))
        flows.update(Q_A=kv['Q_A'],Q_E=kv['Q_E'],GFR=kv['GFR'],Q_urine=urine,Q_return=qreturn)
        if not all(np.all(np.isfinite(v)) for v in (dx,jac,fj,flux,list(diag.values()),list(flows.values()))):
            raise ValueError('体液RHS、代数量或Jacobian超出有限数值工作域')
        return Evaluation(pressure,flows,dx,flux,diag),jac,fj

    def evaluate(self,state,boundary): return self._compute(state,boundary)[0]
    def jacobian(self,state,boundary): return self._compute(state,boundary)[1][:,:24]
    def ledger_jacobian(self,state,boundary): return self._compute(state,boundary)[2][:,:24]

    def evaluate_with_pleural(self,state,boundary,pleural_pressure,*,atrial_transmission=1.):
        """返回Evaluation、24×25和9×25偏导；最后一列为独立ppl(mmHg)。"""
        return self._compute(state,boundary,pleural_pressure=pleural_pressure,atrial_transmission=atrial_transmission)

    def evaluate_with_heart(self,state,boundary,pleural_pressure,cardiac_volume):
        """24×28/9×28；末四输入ppl、心血量、P_RA、P_LA；禁用临时心端。"""
        return self._compute(state,boundary,pleural_pressure=pleural_pressure,atrial_transmission=0.,cardiac_volume=cardiac_volume)

    def equilibrium(self,boundary):
        self._boundary(boundary)
        seed=self.validate_state(self.params.metadata['reference_state']).copy()
        if np.max(np.abs(self.evaluate(seed,boundary).derivative))>=1e-8:
            raise RuntimeError('当前自动初态仅适用于冻结参考平衡；其他初态请显式传入initial_state')
        return seed
