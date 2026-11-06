"""53态主动心脏—呼吸—循环—肾—体液；无历史缓存。"""
import numpy as np
from .heart import ActiveHeart
from .kernel import fiber_rates,valve
from .boundary import ActiveBoundary
from ..fluid.model import FluidCircuit
from ..fluid.kernel import evaluate_chemistry
from ..respiratory.model import RespiratoryCircuit,respiratory_breakpoints
from ..respiratory.boundary import RespiratoryProtocol
from ..respiratory.kernel import evaluate_respiration,reference_gas_state
from ..circulation.model import Evaluation

class ActiveCircuit:
    state_size=53
    boundary_flux_size=12
    balance_names=RespiratoryCircuit.balance_names+('heart_phase',)
    balance_units=RespiratoryCircuit.balance_units+('cycle',)

    def __init__(self,p):
        self.params=p;self.fluid=FluidCircuit(p.respiratory.fluid);self.heart=ActiveHeart(p);self.index=self.fluid.index
        self.balance_matrix=np.zeros((10,12));self.balance_matrix[:6,:9]=self.fluid.balance_matrix
        self.balance_matrix[6,9]=1;self.balance_matrix[8,10]=1;self.balance_matrix[9,11]=1
        from .fast import pack
        self._numeric_parameters=pack(self)

    def rates(self,state,boundary):
        from .fast import rates_numeric,record,validate_working
        if not isinstance(boundary,ActiveBoundary):raise ValueError('需要ActiveBoundary')
        result=rates_numeric(validate_working(state),record(boundary),self._numeric_parameters)
        if not all(np.all(np.isfinite(v)) for v in result):raise ValueError('快核右端或通量非有限')
        return result

    def boundary_flux(self,state,boundary):
        from .fast import flux_numeric,record,validate_working
        if not isinstance(boundary,ActiveBoundary):raise ValueError('需要ActiveBoundary')
        result=flux_numeric(validate_working(state),record(boundary),self._numeric_parameters)
        if not np.all(np.isfinite(result)):raise ValueError('快核通量非有限')
        return result

    def _working(self,state):
        x=np.asarray(state,dtype=float)
        if x.shape!=(53,) or not np.all(np.isfinite(x)) or np.any(x[24:28]<=0) or np.any(x[30:39]<=0) or x[16]<=x[17]:raise ValueError('53态存在非法容积或非有限状态')
        evaluate_chemistry(self.params.respiratory.fluid,x[:24],cardiac_volume=float(x[30:34].sum()))
        return x

    def validate_state(self,state):
        x=self._working(state)
        if np.any(x[39:44]<-1e-7) or np.any(x[48:52]<-1e-7) or np.any(x[48:52]>1+1e-7) or x[29]<0 or x[52]<0:
            raise ValueError('已接受的激活/开度/累计相位越域，不裁剪修复')
        return x

    def reference_initial_state(self,boundary):
        if not isinstance(boundary,ActiveBoundary):raise ValueError('需要ActiveBoundary')
        p=self.params
        fluid=np.array(p.respiratory.fluid.metadata['reference_state'],dtype=float)
        volumes=np.array([50.,130.,50.,130.]);fluid[6]-=volumes.sum()
        start=self.heart.evaluate(volumes,np.full(5,2.),np.zeros(5))
        return self.validate_state(np.r_[fluid,reference_gas_state(p.respiratory.mechanics),volumes,
             start.lengths-p.series_length_um,np.zeros(5),np.zeros(8),0.]).copy()

    def equilibrium(self,boundary):raise ValueError('主动心脏须显式提供initial_state，启动种子不是静态平衡')

    def balance_quantities(self,state):
        x=self.validate_state(state)
        ch=evaluate_chemistry(self.params.respiratory.fluid,x[:24],cardiac_volume=float(x[30:34].sum())).values
        return np.r_[ch['W'],x[23],ch['V_b'],x[20:23],x[24:28].sum(),x[28]-x[25:28].sum(),x[29],x[52]]

    def evaluate_with_jacobian(self,state,boundary):
        x=self._working(state);p=self.params
        if not isinstance(boundary,ActiveBoundary):raise ValueError('需要ActiveBoundary')
        resp=evaluate_respiration(p.respiratory.mechanics,x[24:30],boundary.respiratory())
        heart=self.heart.evaluate(x[30:34],x[34:39],x[39:44],pleural_pressure_mmhg=resp.values['ppl_mmhg'])
        hp=np.zeros((4,53));hp[:,30:44]=heart.pressure_partials
        hp[:,24:30]+=resp.partials['ppl_mmhg']
        bb=boundary.respiratory(heart.pressures[2],heart.pressures[0]).fluid
        fluid,jf,lf=self.fluid.evaluate_with_heart(x[:24],bb,resp.values['ppl_mmhg'],float(x[30:34].sum()))
        chain=np.zeros((28,53));chain[:24,:24]=np.eye(24);chain[24,24:30]=resp.partials['ppl_mmhg']
        chain[25,30:34]=1;chain[26]=hp[2];chain[27]=hp[0]
        dx=np.zeros(53);jac=np.zeros((53,53));ledger=np.zeros((12,53))
        dx[:24]=fluid.derivative;jac[:24]=jf@chain
        dx[24:30]=resp.derivative;jac[24:30,24:30]=resp.jacobian
        ledger[:9]=lf@chain;ledger[9,24:30]=resp.partials['q_mouth']
        vascular=self.fluid.vascular
        vp,slopes=vascular._pressures(x[:16],resp.values['ppl_mmhg'])
        gp=np.zeros((16,53));gp[:,:16]=np.diag(slopes);gp[:,24:30]=vascular.pleural_mask[:,None]*resp.partials['ppl_mmhg']
        rt,drt=vascular.thoracic_resistance(x[11]);gt=np.zeros(53);gl=np.zeros(53)
        if vp[11]>heart.pressures[2]:
            gt=(gp[11]-hp[2])/rt;gt[11]-=(vp[11]-heart.pressures[2])*drt/rt**2
        if vp[14]>heart.pressures[0]:gl=(gp[14]-hp[0])/p.respiratory.fluid.vascular.pulmonary_outflow_resistance
        qt,ql=fluid.flows['TV_RA'],fluid.flows['PV_LA'];q=x[44:48]
        dx[0]+=q[1];dx[12]+=q[3];jac[0,45]+=1;jac[12,47]+=1
        dx[30:34]=[ql-q[0],q[0]-q[1],qt-q[2],q[2]-q[3]]
        jac[30]=gl;jac[30,44]-=1;jac[31,44]+=1;jac[31,45]-=1
        jac[32]=gt;jac[32,46]-=1;jac[33,46]+=1;jac[33,47]-=1
        lsgrad=np.zeros((5,53));lsgrad[:,30:44]=heart.length_partials
        fibers=fiber_rates(p,heart.lengths,x[34:39],x[39:44],x[52])
        fc=np.zeros((16,53));fc[:5]=lsgrad;fc[5:10,34:39]=np.eye(5);fc[10:15,39:44]=np.eye(5);fc[15,52]=1
        dx[34:44]=fibers.derivative;jac[34:44]=fibers.jacobian@fc
        delta=np.array([heart.pressures[0]-heart.pressures[1],heart.pressures[1]-vp[0],
                        heart.pressures[2]-heart.pressures[3],heart.pressures[3]-vp[12]])
        gd=np.stack((hp[0]-hp[1],hp[1]-gp[0],hp[2]-hp[3],hp[3]-gp[12]))
        flow={**fluid.flows,'SA_in':float(q[1]),'PA_in':float(q[3])}
        for i,v in enumerate(p.valves):
            ve=valve(v,delta[i],q[i],x[48+i]);vc=np.zeros((3,53));vc[0]=gd[i];vc[1,44+i]=1;vc[2,48+i]=1
            dx[44+i]=ve.values['q_dot'];dx[48+i]=ve.values['xi_dot']
            jac[44+i]=ve.partials['q_dot']@vc;jac[48+i]=ve.partials['xi_dot']@vc
            flow['q_'+v.name]=float(q[i])
        dx[52]=1/p.cycle
        flux=np.r_[fluid.boundary_flux,resp.values['q_mouth'],boundary.respiratory_rate/60,1/p.cycle]
        diagnostics={**fluid.diagnostics,**resp.values,'P_LA':heart.pressures[0],'P_LV':heart.pressures[1],
            'P_RA':heart.pressures[2],'P_RV':heart.pressures[3],'P_peri':heart.pericardial_pressure,
            'heart_balance':float(np.max(np.abs(heart.residual))),'heart_condition':heart.condition,
            'max_abs_z':max(abs(w['z']) for w in heart.walls),'min_raw_stress':min(w['sigma_raw_Pa'] for w in heart.walls),
            'max_extension_Pa':max(w['sigma_effective_Pa']-w['sigma_raw_Pa'] for w in heart.walls),
            'y_m':.03*np.exp(heart.coordinates[0]),'septal_ml':heart.coordinates[1]*100}
        for i,w in enumerate(heart.walls):
            for key in ('ls_um','sigma_active_Pa','sigma_raw_Pa','sigma_effective_Pa','tension_N_m','z'):
                diagnostics[w['name']+'_'+key]=w[key]
        return Evaluation(np.r_[vp,heart.pressures],flow,dx,flux,diagnostics),jac,ledger

    def evaluate(self,state,boundary):return self.evaluate_with_jacobian(state,boundary)[0]
    def jacobian(self,state,boundary):return self.evaluate_with_jacobian(state,boundary)[1]
    def ledger_jacobian(self,state,boundary):return self.evaluate_with_jacobian(state,boundary)[2]

    def integration_breakpoints(self,protocol,initial_state,end_time):
        resp_protocol=RespiratoryProtocol(protocol.switch_times,tuple(b.respiratory() for b in protocol.values))
        initial_state=self.validate_state(initial_state)
        cuts=respiratory_breakpoints(resp_protocol,float(initial_state[29]),end_time,self.params.respiratory.mechanics)
        phase=float(initial_state[52]);p=self.params
        for k in range(int(np.floor(phase))-1,int(np.ceil(phase+end_time/p.cycle))+1):
            for delay in (0.,p.av_delay_s):
                t=(k-phase)*p.cycle+delay
                if 1e-10<t<end_time-1e-10:cuts.append(t)
        return sorted(set(cuts))
