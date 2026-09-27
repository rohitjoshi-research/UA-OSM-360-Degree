import sim2 as sm, numpy as np, json, sys, time
import e5, e2
from e3 import STAT
VIS=True
VISMETH={'uaosm','a5_noassign_dynbase'}
import sim2 as _sm
def _lo(stat,step=3):
    out={}
    for c,(st,valid) in stat.items():
        g=np.zeros((_sm.FE_N,_sm.FE_N),bool); g[::step,::step]=True; ix=np.nonzero(g.ravel())[0]
        out[c]=(dict(D=st['D'][ix],t=st['t'][ix],o=st['o']),valid.ravel()[ix])
    return out
STATLO=_lo(STAT)
def run_px(methods,sc0,nsc,kset,sigma,lat,pdrop,T=4.0,dt=0.1,seed=1,every=2,pred=True):
    out={m:dict(sc_an=[],sc_px=[],sc_an_lost=[],sc_px_lost=[],sc_k=[],pairs=[]) for m in methods}
    for sc in range(sc0,sc0+nsc):
        k=kset[sc%len(kset)]; rng=np.random.default_rng(seed*1000+sc); objs=e5.make_scene(rng,k)
        ms={}; pers={}
        for mn in methods: ms[mn]=e5.mk(mn.replace('+vis',''),sigma); pers[mn]=e2.Percept(np.random.default_rng(seed*1000+sc+7),sigma,lat,pdrop,pred)
        acc={mn:[0,0,0,0,0,0] for mn in methods}; t=0; fi=0; occ=set(); hyst={}
        while t<T-1e-9:
            e2.step(objs,dt)
            if VIS and fi%every==0:
                raw=sm.cam_visibility(STATLO,objs,minpx=4)
                for o in objs:
                    for c in sm.CAM_IDS:
                        key=(o['id'],c); hs=hyst.get(key,[False,0]); obs=key in raw
                        if obs!=hs[0]: hs[1]+=1
                        else: hs[1]=0
                        if hs[1]>=2: hs[0]=obs; hs[1]=0
                        hyst[key]=hs
                occ={k2 for k2,v in hyst.items() if v[0]}
            elif not VIS: occ=set()
            for mn in methods:
                m,up=ms[mn]; m['occl']=occ if mn.endswith('+vis') else set(); up(m,pers[mn](objs,t,dt),dt)
            if fi%every==0:
                M=sm.pixel_masks(STAT,objs)
                for mn in methods:
                    m,_=ms[mn]; rs=[sm.eval_obj(m,o) for o in objs]; W,_=sm.weight_grid(m); pr=sm.pixel_eval(M,W,objs)
                    vis=[i for i,r in enumerate(pr) if not r['hidden']]
                    if not vis: continue
                    a=min(rs[i]['score'] for i in vis); p=min(pr[i]['score'] for i in vis); A=acc[mn]
                    A[0]+=a; A[1]+=p; A[2]+=any(sm.status(rs[i])=='LOST' for i in vis); A[3]+=any(sm.status(pr[i])=='LOST' for i in vis); A[4]+=1
                    A[5]+=len(pr)-len(vis)
                    out[mn]['pairs'].append((a,p))
            t+=dt; fi+=1
        for mn in methods:
            A=acc[mn]; R=out[mn]; R['sc_an'].append(A[0]/A[4]); R['sc_px'].append(A[1]/A[4]); R['sc_an_lost'].append(A[2]/A[4]); R['sc_px_lost'].append(A[3]/A[4]); R['sc_k'].append(k); R.setdefault('hidden',0); R['hidden']+=A[5]
    return out
if __name__=='__main__':
    import e5 as _e5
    _e5.OCCL=False
    if len(sys.argv)>3 and sys.argv[3]=='novis': VIS=False
    tag=sys.argv[1]; C=json.loads(sys.argv[2]); t0=time.time(); res=run_px(**C); json.dump(res,open(f'r1_{tag}.json','w'))
    print(tag,round(time.time()-t0),{m:(round(np.mean(v['sc_an']),3),round(np.mean(v['sc_px']),3),round(100*np.mean(v['sc_px_lost']),1)) for m,v in res.items()})
