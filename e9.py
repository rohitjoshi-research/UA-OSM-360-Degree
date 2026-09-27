# review round 2 harness: parameter overrides, L5 logging, perception faults, rear corner + ego motion, covariance miscalibration
import sim2 as sm, numpy as np, json, sys, time, math
import e5, e2, e6
e5.OCCL=False
from e3 import STAT
from scen import PED_B, CHILD_B
def make_scene(rng,k,region):
    objs=e5.make_scene(rng,k)
    if region=='RL':
        for o in objs: o['y']=-o['y']; o['vy']=-o['vy']
    return objs
def step(objs,dt,region,vego):
    ylo,yhi=(-5.8,-2.35) if region=='FL' else (2.35,5.8)
    for o in objs:
        o['x']+=o['vx']*dt; o['y']+=(o['vy']-vego)*dt
        if not -4.8<o['x']<-1.0: o['vx']*=-1; o['x']=min(max(o['x'],-4.8),-1.0)
        if vego==0 and not ylo<o['y']<yhi: o['vy']*=-1; o['y']=min(max(o['y'],ylo),yhi)
def faults(est,rng,t,dt,state,pmis,rfp,region):
    out=[]
    for e in est:
        if pmis>0:
            if e['id'] not in state['mis']: state['mis'][e['id']]=rng.random()<pmis
            if state['mis'][e['id']] and e['kind'] in ('Child','Pedestrian'):
                e=dict(e); e['kind']='Pedestrian' if e['kind']=='Child' else 'Child'; e['h']=1.75 if e['kind']=='Pedestrian' else 1.1
        out.append(e)
    if rfp>0 and rng.random()<rfp*dt:
        y=rng.uniform(-5.5,-2.35); y=y if region=='FL' else -y
        state['fp'].append(dict(id=f'fp{len(state["fp"])}',kind='Pedestrian',x=rng.uniform(-4.5,-1.05),y=y,h=1.75,r=0.25,t_end=t+0.5,coasting=False,sig_extra=0.0))
    out+=[f for f in state['fp'] if f['t_end']>t]
    return out
def run(methods,sc0,nsc,kset,sigma,lat,pdrop,sig_true=None,region='FL',vego=0.0,pmis=0.0,rfp=0.0,params=None,T=4.0,dt=0.1,seed=1,every=2):
    if params:
        for k,v in params.items(): setattr(sm,k,v)
    st_true=sigma if sig_true is None else sig_true
    out={m:dict(px=[],pl=[],l5=0,l5c=0,l5r=0,l5r_ok=0,fr=0,lostfr=0,lost_warned=0,lost_k=[0]*6,k=[]) for m in methods}
    for sc in range(sc0,sc0+nsc):
        k=kset[sc%len(kset)]; rng=np.random.default_rng(seed*1000+sc); objs=make_scene(rng,k,region)
        ms={mn:e5.mk(mn.replace('+vis',''),sigma) for mn in methods}
        pers={mn:e2.Percept(np.random.default_rng(seed*1000+sc+7),st_true,lat,pdrop) for mn in methods}
        fst={mn:dict(mis={},fp=[]) for mn in methods}; frng={mn:np.random.default_rng(seed*1000+sc+99) for mn in methods}
        acc={mn:[0,0,0] for mn in methods}; t=0; fi=0; occ=set(); hyst={}
        while t<T-1e-9:
            step(objs,dt,region,vego)
            if fi%every==0:
                raw=sm.cam_visibility(e6.STATLO,objs,minpx=4)
                for o in objs:
                    for c in sm.CAM_IDS:
                        key=(o['id'],c); hs=hyst.get(key,[False,0]); obs=key in raw
                        hs[1]=hs[1]+1 if obs!=hs[0] else 0
                        if hs[1]>=2: hs[0]=obs; hs[1]=0
                        hyst[key]=hs
                occ={k2 for k2,v in hyst.items() if v[0]}
            for mn in methods:
                m,up=ms[mn]; m['occl']=occ if mn.endswith('+vis') else set()
                est=faults(pers[mn](objs,t,dt),frng[mn],t,dt,fst[mn],pmis,rfp,region); up(m,est,dt)
            if fi%every==0:
                M=sm.pixel_masks(STAT,objs)
                for mn in methods:
                    m,_=ms[mn]; W,_=sm.weight_grid(m); pr=sm.pixel_eval(M,W,objs); vis=[r for r in pr if not r['hidden']]
                    R=out[mn]; R['fr']+=1
                    lost=any(sm.status(r)=='LOST' for r in vis) if vis else False
                    if vis: A=acc[mn]; A[0]+=min(r['score'] for r in vis); A[1]+=lost; A[2]+=1
                    if lost: R['lostfr']+=1; R['lost_warned']+=(m.get('level')==5); R['lost_k'][k]+=1
                    if m.get('level')==5:
                        R['l5']+=1; coast=any(e.get('coasting') for e in pers[mn].prev.values())
                        if coast: R['l5c']+=1
                        else:
                            R['l5r']+=1; R['l5r_ok']+=(not lost)
            t+=dt; fi+=1
        for mn in methods:
            A=acc[mn]
            if A[2]: out[mn]['px'].append(A[0]/A[2]); out[mn]['pl'].append(A[1]/A[2]); out[mn]['k'].append(k)
    return out
if __name__=='__main__':
    tag=sys.argv[1]; C=json.loads(sys.argv[2]); t0=time.time(); r=run(**C); json.dump(r,open(f'r2_{tag}.json','w'))
    print(tag,round(time.time()-t0),{m:dict(px=round(np.mean(v['px']),3),lost=round(100*np.mean(v['pl']),1),L5=round(100*v['l5']/max(v['fr'],1),1),L5coast=round(100*v['l5c']/max(v['fr'],1),1),L5res=round(100*v['l5r']/max(v['fr'],1),1),L5res_nolost=v['l5r_ok'],lost_frames=v['lostfr'],lost_warned=v['lost_warned']) for m,v in r.items()})
