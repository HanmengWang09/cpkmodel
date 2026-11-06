"""确定性代数核；Numba仅编译同一解析公式，无状态缓存。"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def wall(p,a,c,li,activation,scale,se,lzero,soft):
    val=np.full(9,np.nan);grad=np.zeros((9,4))
    z=1.5*p[0]*c/a;u=3*p[0]*c**3/(8*math.pi)
    if a<=0 or abs(z)>=.95 or abs(u)>=.95:return val,grad
    gz=np.array([-z/a,1.5*p[0]/a,0.,0.])
    eps=.5*math.log(a/p[1])-z*z/12-.019*z**4
    ge=np.array([.5/a,0.,0.,0.])-(z/6+.076*z**3)*gz
    ls=p[4]*math.exp(eps);gls=ls*ge
    d=math.log(ls/p[5]);pas=p[2]*math.expm1(p[6]*d)+p[8]*p[3]*math.expm1(p[7]*d)
    tangent=p[6]*p[2]*math.exp(p[6]*d)+p[7]*p[8]*p[3]*math.exp(p[7]*d)
    gpas=tangent*ge
    k=1.51*(li/lzero-1);amplitude=p[3]*scale
    active=amplitude*activation*k*(ls-li)/se
    ga=amplitude*activation*k/se*gls
    ga[2]+=amplitude*activation*(1.51/lzero*(ls-li)-k)/se
    ga[3]=amplitude*k*(ls-li)/se
    raw=pas+active;gr=gpas+ga
    if soft>0:
        hyp=math.hypot(raw,soft)
        if raw>=0:eff=.5*(raw+hyp);de=.5*(1+raw/hyp)
        else:eff=soft*soft/(2*(hyp-raw));de=soft*soft/(2*hyp*(hyp-raw))
    else:eff=raw;de=1.
    factor=1+z*z/3+z**4/5;gf=(2*z/3+4*z**3/5)*gz
    pref=p[0]/(2*a);tension=pref*eff*factor
    gt=pref*(de*gr*factor+eff*gf);gt[0]-=tension/a
    val[:]=np.array([ls,pas,active,raw,eff,tension,z,u,eps])
    grad[0]=gls;grad[1]=gpas;grad[2]=ga;grad[3]=gr;grad[4]=de*gr;grad[5]=gt;grad[6]=gz;grad[8]=ge
    return val,grad

@njit(cache=True)
def geometry(par,v,li,cc,scale,se,lzero,soft,q):
    residual=np.zeros(2);jac=np.zeros((2,16));pressure=np.zeros(2);gp=np.zeros((2,16))
    lsgrad=np.zeros((3,16));values=np.zeros((3,9));geom=np.zeros((3,5))
    if abs(q[0])>5 or abs(q[1])>10:
        residual[:]=np.nan;return residual,jac,pressure,gp,lsgrad,values,geom
    y=.03*math.exp(q[0]);vs=q[1]*1e-4
    vols=np.array([vs-(v[1]*1e-6+.5*(par[2,0]+par[3,0])),vs,vs+(v[3]*1e-6+.5*(par[4,0]+par[3,0]))])
    if vols[0]>=0 or vols[2]<=0:
        residual[:]=np.nan;return residual,jac,pressure,gp,lsgrad,values,geom
    gy=np.zeros(16);gy[0]=y
    for n in range(3):
        vm=vols[n];gvm=np.zeros(16);gvm[1]=1e-4
        if n==0:gvm[3]=-1e-6
        if n==2:gvm[5]=1e-6
        x=2*y*math.sinh(math.asinh(3*vm/(math.pi*y**3))/3);den=x*x+y*y
        gx=2/(math.pi*den)*gvm-2*x*y/den*gy
        gd=2*x*gx+2*y*gy;a=math.pi*den;c=2*x/den
        garea=math.pi*gd;gc=2*gx/den-2*x/den**2*gd
        w,g=wall(par[n+2],a,c,li[n+2],cc[n+2],scale[n+2],se,lzero,soft)
        chain=np.zeros((4,16));chain[0]=garea;chain[1]=gc;chain[2,8+n]=1;chain[3,13+n]=1
        allg=g@chain;t=w[5];gt=allg[5]
        tx=t*c*y;ty=t*(1-x*c)
        gtx=gt*c*y+t*gc*y+t*c*gy
        gty=gt*(1-x*c)-t*gx*c-t*x*gc
        residual[0]+=tx;residual[1]+=ty;jac[0]+=gtx;jac[1]+=gty
        if n==0:pressure[0]=-2*tx/y;gp[0]=-2*gtx/y+2*tx/y**2*gy
        if n==2:pressure[1]=2*tx/y;gp[1]=2*gtx/y-2*tx/y**2*gy
        lsgrad[n]=allg[0];values[n]=w;geom[n]=np.array([x,y,vm,a,c])
    return residual,jac,pressure,gp,lsgrad,values,geom

@njit(cache=True)
def newton(par,v,li,cc,scale,se,lzero,soft,q0):
    q=q0.copy()
    for iteration in range(50):
        e=geometry(par,v,li,cc,scale,se,lzero,soft,q)
        r,j=e[0],e[1];norm=np.max(np.abs(r))
        if not np.isfinite(norm):return q,False,iteration
        if norm<1e-9:return q,True,iteration
        det=j[0,0]*j[1,1]-j[0,1]*j[1,0]
        if abs(det)<1e-16:return q,False,iteration
        step=np.array([(j[1,1]*r[0]-j[0,1]*r[1])/det,(-j[1,0]*r[0]+j[0,0]*r[1])/det])
        step/=max(1.,np.max(np.abs(step)))
        accepted=False
        for power in range(24):
            alpha=2.**(-power);candidate=q-alpha*step
            trial=geometry(par,v,li,cc,scale,se,lzero,soft,candidate)[0]
            nn=np.max(np.abs(trial))
            if np.isfinite(nn) and nn<norm*(1-1e-4*alpha):q=candidate;accepted=True;break
        if not accepted:return q,False,iteration
    return q,False,50
