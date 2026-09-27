import sim2 as sm, numpy as np, math, json, sys, time
sm.NS=12
from scen import PED_B, CHILD_B
import e2
KINDS=[('Child',1.1,0.2),('Pedestrian',1.75,0.25),('Pole',2.2,0.09)]
def make_scene(rng,k):
    objs=[]
    while len(objs)<k:
        x=rng.uniform(-4.5,-1.05); y=rng.uniform(-5.5,-2.35)
        if any(math.hypot(x-o['x'],y-o['y'])<0.6 for o in objs): continue
        kind,h,r=KINDS[rng.choice(3,p=[0.35,0.5,0.15])]
        sp=0 if kind=='Pole' else rng.uniform(0.2,1.3); th=rng.uniform(0,2*math.pi)
        o=dict(id=f'o{len(objs)}',kind=kind,x=x,y=y,h=h,r=r,vx=sp*math.cos(th),vy=sp*math.sin(th))
        if kind=='Pole': o['pole']=True
        else: o['bands']=CHILD_B if kind=='Child' else PED_B
        objs.append(o)
    return objs
def mk(name,sg):
    m,up=_mk(name,sg)
    import sim2 as _s
    def upw(mm,est,dt,_u=up):
        if OCCL: mm['occl']=_s.occlusions(est)
        _u(mm,est,dt)
    return m,upw
OCCL=True
def _mk(name,sg):
    return {'fixed':lambda:(sm.make_method('fixed',6,sg),sm.update_method),
     'dynamic':lambda:(sm.make_method('dynamic',6,sg),sm.update_method),
     'osm_naive':lambda:(sm.make_method('shape',6,0.0),sm.update_method),
     'curve':lambda:(sm.make_curve(6,sg),sm.update_curve),
     'a1_noassign_fixedbase':lambda:(sm.make_method('shape',6,sg),sm.update_method),
     'a5_noassign_dynbase':lambda:(sm.make_prop(6,sg),sm.update_prop_noassign),
     'uaosm_nodilate':lambda:(sm.make_prop(6,0.0),sm.update_prop),
     'uaosm':lambda:(sm.make_prop(6,sg),sm.update_prop),
     'uaosm_novis':lambda:(sm.make_prop(6,sg),sm.update_prop)}[name]()
def run(methods,sc0,nsc,kset,sigma,lat,pdrop,pixel=False,T=4.0,dt=0.1,seed=1,every=2):
    if pixel: from e3 import STAT
    out={m:dict(sc_ovi=[],sc_lost=[],sc_k=[],px_ovi=[],px_lost=[],an_pairs=[],l5=0,frames=0) for m in methods}
    for sc in range(sc0,sc0+nsc):
        k=kset[sc%len(kset)]
        for mname in methods:
            rng=np.random.default_rng(seed*1000+sc); objs=make_scene(rng,k)
            per=e2.Percept(np.random.default_rng(seed*1000+sc+7),sigma,lat,pdrop)
            m,up=mk(mname,sigma); t=0; so=sl=sf=0; po=pl=pf=0; fi=0
            while t<T-1e-9:
                e2.step(objs,dt); est=per(objs,t,dt); up(m,est,dt)
                rs=[sm.eval_obj(m,o) for o in objs]; w=min(r['score'] for r in rs)
                so+=w; sf+=1; sl+=any(sm.status(r)=='LOST' for r in rs)
                out[mname]['l5']+=(m.get('level')==5); out[mname]['frames']+=1
                if pixel and fi%every==0:
                    M=sm.pixel_masks(STAT,objs); W,_=sm.weight_grid(m); pr=sm.pixel_eval(M,W,objs)
                    pw=min(r['score'] for r in pr); po+=pw; pf+=1; pl+=any(sm.status(r)=='LOST' for r in pr)
                    out[mname]['an_pairs'].append((w,pw))
                t+=dt; fi+=1
            R=out[mname]; R['sc_ovi'].append(so/sf); R['sc_lost'].append(sl/sf); R['sc_k'].append(k)
            if pixel: R['px_ovi'].append(po/pf); R['px_lost'].append(pl/pf)
    return out
if __name__=='__main__':
    tag=sys.argv[1]; C=json.loads(sys.argv[2]); t0=time.time()
    res=run(**C); res['_cfg']=C; json.dump(res,open(f'r1_{tag}.json','w'))
    print(tag,round(time.time()-t0),{m:(round(np.mean(v['sc_ovi']),3),round(100*np.mean(v['sc_lost']),1)) for m,v in res.items() if not m.startswith('_')},
          {m:(round(np.mean(v['px_ovi']),3),round(100*np.mean(v['px_lost']),1)) for m,v in res.items() if not m.startswith('_') and v['px_ovi']})
