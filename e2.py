# E2: Monte Carlo benchmark on random corner scenes (analytic footprint model)
import sim2 as sm, numpy as np, math, json, sys, time
sm.NS=12
KINDS=[('Child',1.1,0.2),('Pedestrian',1.75,0.25),('Pole',2.2,0.09)]
def make_scene(rng,k):
    objs=[]
    while len(objs)<k:
        x=rng.uniform(-4.5,-1.05); y=rng.uniform(-5.5,-2.35)
        if any(math.hypot(x-o['x'],y-o['y'])<0.6 for o in objs): continue
        kind,h,r=KINDS[rng.choice(3,p=[0.35,0.5,0.15])]
        sp=0 if kind=='Pole' else rng.uniform(0.2,1.3); th=rng.uniform(0,2*math.pi)
        objs.append(dict(id=f'o{len(objs)}',kind=kind,x=x,y=y,h=h,r=r,vx=sp*math.cos(th),vy=sp*math.sin(th)))
    return objs
def step(objs,dt):
    for o in objs:
        o['x']+=o['vx']*dt; o['y']+=o['vy']*dt
        if not -4.8<o['x']<-1.0: o['vx']*=-1; o['x']=min(max(o['x'],-4.8),-1.0)
        if not -5.8<o['y']<-2.35: o['vy']*=-1; o['y']=min(max(o['y'],-5.8),-2.35)
class Percept:
    def __init__(s,rng,sigma,lat,pdrop,pred=True):
        s.pred=pred; s.rng=rng; s.sig=sigma; s.lat=lat; s.p=pdrop; s.hist={}; s.n={}; s.lost={}; s.prev={}
    def __call__(s,objs,t,dt):
        est=[]
        for o in objs:
            h=s.hist.setdefault(o['id'],[]); h.append((t,o['x'],o['y']))
            nx,ny=s.n.get(o['id'],(0.0,0.0)); tau=0.6
            nx+=-nx/tau*dt+s.sig*math.sqrt(2*dt/tau)*s.rng.normal(); ny+=-ny/tau*dt+s.sig*math.sqrt(2*dt/tau)*s.rng.normal(); s.n[o['id']]=(nx,ny)
            L=s.lost.get(o['id'],0)
            if L<=t and s.rng.random()<s.p*dt: s.lost[o['id']]=t+1.0; L=t+1.0
            coast=L>t and o['id'] in s.prev
            if coast:
                pv=s.prev[o['id']]; e=dict(o,x=pv['x']+pv['evx']*dt,y=pv['y']+pv['evy']*dt,coasting=True,sig_extra=min(0.5,pv.get('sig_extra',0)+0.03)); e['evx'],e['evy']=pv['evx'],pv['evy']
            else:
                td=t-s.lat; past=[p for p in h if p[0]<=td+1e-9] or [h[0]]; _,px,py=past[-1]
                p2=[p for p in h if p[0]<=td-0.2+1e-9] or [h[0]]; _,qx,qy=p2[-1]
                vx,vy=(px-qx)/0.2,(py-qy)/0.2
                k=s.lat if s.pred else 0.0
                e=dict(o,x=px+nx+vx*k,y=py+ny+vy*k,coasting=False,sig_extra=0.0); e['evx'],e['evy']=vx,vy
            s.prev[o['id']]=e; est.append(e)
        return est
METHODS={
 'fixed':lambda sg:sm.make_method('fixed',6,sg),
 'dynamic':lambda sg:sm.make_method('dynamic',6,sg),
 'osm_naive':lambda sg:sm.make_method('shape',6,0.0),
 'uaosm_noassign':lambda sg:sm.make_method('shape',6,sg),
 'uaosm_nodilate':lambda sg:sm.make_prop(6,0.0),
 'uaosm':lambda sg:sm.make_prop(6,sg)}
def run(cond,methods,nscenes,kset,sigma,lat,pdrop,T=4.0,dt=0.1,seed=0,pred=True):
    out={m:dict(ovi=0,lost=0,cut=0,frames=0,switch=0,objsec=0,motion=0,l5=0,sc_ovi=[],sc_lost=[]) for m in methods}
    for sc in range(nscenes):
        k=kset[sc%len(kset)]
        for mname in methods:
            rng=np.random.default_rng(seed*1000+sc); objs=make_scene(rng,k); per=Percept(np.random.default_rng(seed*1000+sc+7),sigma,lat,pdrop,pred)
            m=METHODS[mname](sigma); t=0; prevown={}; so=0; sl=0; sf=0
            while t<T-1e-9:
                step(objs,dt); est=per(objs,t,dt)
                if m['kind']=='prop': sm.update_prop(m,est,dt)
                else: sm.update_method(m,est,dt)
                rs=[sm.eval_obj(m,o) for o in objs]; R=out[mname]
                w=min(r['score'] for r in rs); R['ovi']+=w; R['frames']+=1; so+=w; sf+=1; sl+=any(sm.status(r)=='LOST' for r in rs)
                R['lost']+=any(sm.status(r)=='LOST' for r in rs); R['cut']+=any(sm.status(r) in ('CUT','LOST') for r in rs)
                R['motion']+=m.get('motion',0) if mname=='dynamic' else 0
                R['l5']+=(m.get('level')==5)
                own=dict(m.get('owners',{}))
                R['switch']+=sum(1 for i,c in own.items() if i in prevown and prevown[i]!=c); prevown=own
                R['objsec']+=len(objs)*dt
                t+=dt
            out[mname]['sc_ovi'].append(so/sf); out[mname]['sc_lost'].append(sl/sf)
    for mname,R in out.items():
        f=R['frames']; R.update(ovi=R['ovi']/f,lost=R['lost']/f,cut=R['cut']/f,motion=R['motion']/f,l5=R['l5']/f,switch_per_objs=R['switch']/max(R['objsec'],1e-9))
    return out
if __name__=='__main__':
    cond=sys.argv[1]; t0=time.time()
    C=json.loads(sys.argv[2])
    res=run(cond,**C); res['_cfg']=C; res['_time']=round(time.time()-t0,1)
    json.dump(res,open(f'e2_{cond}.json','w'),indent=1)
    print(cond,res['_time'],{m:(round(v['ovi'],3),round(v['lost'],3)) for m,v in res.items() if not m.startswith('_')})
