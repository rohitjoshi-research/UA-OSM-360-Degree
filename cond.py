# Scenario catalog: time of day x weather -> assumed perception degradation (documented in the paper)
import e9, json, sys, time, numpy as np
TOD={'morning':dict(sigma=0.08,pdrop=0.03,pmis=0.05,rfp=0.10),   # low sun, long shadows, glare
     'afternoon':dict(sigma=0.06,pdrop=0.01,pmis=0.03,rfp=0.05), # high sun, best case
     'evening':dict(sigma=0.10,pdrop=0.04,pmis=0.08,rfp=0.15),   # blue hour, mixed artificial light
     'night':dict(sigma=0.14,pdrop=0.08,pmis=0.12,rfp=0.30)}     # artificial light only, noise
WX={'clear':dict(sm=1.0,dp=0.0,dm=0.0,df=0.0),
    'rain':dict(sm=1.4,dp=0.04,dm=0.05,df=0.30),                 # droplets, spray, wet reflections
    'fog':dict(sm=1.6,dp=0.06,dm=0.08,df=0.10)}                  # reduced range and contrast
def params(t,w):
    a=TOD[t]; b=WX[w]
    return dict(sigma=round(a['sigma']*b['sm'],3),pdrop=round(a['pdrop']+b['dp'],3),pmis=round(a['pmis']+b['dm'],3),rfp=round(a['rfp']+b['df'],3))
if __name__=='__main__':
    for key in sys.argv[1:]:
        t,w=key.split('-'); P=params(t,w); t0=time.time()
        r=e9.run(methods=['dynamic','uaosm+vis'],sc0=500,nsc=10,kset=[1,2,3,4,5],lat=0.1,**P)
        json.dump(dict(params=P,res=r),open(f'cond_{key}.json','w'))
        print(key,P,round(time.time()-t0),{m:round(100*np.mean(v['pl']),1) for m,v in r.items()},'L5',round(100*r['uaosm+vis']['l5']/max(r['uaosm+vis']['fr'],1),1),flush=True)
