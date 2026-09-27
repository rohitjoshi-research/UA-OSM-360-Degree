import numpy as np, cv2, math
HX, HY = 0.95, 2.3
M0=0.12; LAM_U=0.03; LAM_S=0.25; L2_THR=0.95; L5_THR=0.9
BR=7.2; BN=440
CURVE_R=np.arange(0.05,9.0,0.1); CURVE_A=np.arange(0.0,90.01,2.5)
DEG = 180/math.pi
CAM_IDS = ['F','B','L','R']
SEAM_MIN, SEAM_MAX = 15, 75
CAR_H = 1.45

def norm(v): return v/np.linalg.norm(v)

class Cam:
    def __init__(s, pos, fwd):
        s.pos = np.array(pos, float); f = norm(np.array(fwd, float)); up = np.array([0,0,1.0])
        d = norm(-up - np.dot(-up, f)*f); r = np.cross(f, d)
        s.R = np.stack([r, d, f])        # rows: cam axes in world
        s.h = s.pos[2]

def pitched(yaw_vec, pitch_deg):
    y = norm(np.array([yaw_vec[0], yaw_vec[1], 0.0])); p = math.radians(pitch_deg)
    return np.array([y[0]*math.cos(p), y[1]*math.cos(p), -math.sin(p)])

CAMS = {'F': Cam((0,-2.35,0.65), pitched((0,-1),30)), 'B': Cam((0,2.35,0.95), pitched((0,1),35)),
        'L': Cam((-1.0,-0.9,1.0), pitched((-1,0),45)), 'R': Cam((1.0,-0.9,1.0), pitched((1,0),45))}

# ---------------- analytic model (same as web demo) ----------------
def sees(c,x,y):
    return (y<=-HY) if c=='F' else (y>=HY) if c=='B' else (x<=-HX) if c=='L' else (x>=HX)
CORNERS=[dict(a='F',b='L',cx=-HX,cy=-HY,sx=-1,sy=-1),dict(a='F',b='R',cx=HX,cy=-HY,sx=1,sy=-1),
         dict(a='B',b='L',cx=-HX,cy=HY,sx=-1,sy=1),dict(a='B',b='R',cx=HX,cy=HY,sx=1,sy=1)]
def corner_of(x,y):
    ya=-1 if y<=-HY else (1 if y>=HY else 0); xa=-1 if x<=-HX else (1 if x>=HX else 0)
    if not ya or not xa: return -1
    return (0 if xa<0 else 1) if ya<0 else (2 if xa<0 else 3)
def sstep(e0,e1,x):
    t=min(1.0,max(0.0,(x-e0)/(e1-e0))); return t*t*(3-2*t)
def cor_inside(c,x,y):
    px,py=x-c['x1'],y-c['y1']; s=px*c['dx']+py*c['dy']; p=abs(px*c['dy']-py*c['dx'])
    if s<-c['rad']: return False
    if s<0: return math.hypot(px,py)<=c['rad']
    if s>c['len']: return False
    return p<=c['rad']+c['slope']*s
CAM_CODE={'F':0,'B':1,'L':2,'R':3}
def weights(m,x,y):
    out={'F':0.0,'B':0.0,'L':0.0,'R':0.0}
    bg=m.get('blobgrid')
    if bg is not None:
        i=int((x+BR)/(2*BR)*BN); j=int((y+BR)/(2*BR)*BN)
        if 0<=i<BN and 0<=j<BN and bg[j,i]>=0:
            c=CAM_IDS[bg[j,i]]
            if sees(c,x,y): out[c]=1.0; return out
    for c in m['corridors']:
        if sees(c['owner'],x,y) and cor_inside(c,x,y): out[c['owner']]=1.0; return out
    k=corner_of(x,y)
    if k<0:
        if y<=-HY: out['F']=1
        elif y>=HY: out['B']=1
        elif x<=-HX: out['L']=1
        elif x>=HX: out['R']=1
        return out
    C=CORNERS[k]; u=(x-C['cx'])*C['sx']; v=(y-C['cy'])*C['sy']; a=math.atan2(u,v)*DEG
    phi=m['seams'][k]
    cv=m.get('curve')
    if cv is not None and cv[k] is not None:
        phi=float(np.interp(math.hypot(u,v),CURVE_R,cv[k]))
    b=m['band']; wa=1-sstep(phi-b,phi+b,a); out[C['a']]=wa; out[C['b']]=1-wa; return out
STREAK_CAP=7.0
LMAX=10.5   # beyond the display corner (display half-width 7.2 m)
def streak(c,o):
    C=CAMS[c]; dx=o['x']-C.pos[0]; dy=o['y']-C.pos[1]; d=math.hypot(dx,dy); dx/=d; dy/=d
    # exact flat-ground projection of the object top; objects at/above camera height reach the display edge
    L=LMAX if o['h']>=C.h*0.999 else min(d*o['h']/(C.h-o['h']),LMAX)
    return dict(x1=o['x'],y1=o['y'],x2=o['x']+dx*L,y2=o['y']+dy*L,dx=dx,dy=dy,len=L,d=d)
NS=18
def eval_obj(m,o):
    lst=[]
    for c in CAM_IDS:
        if not sees(c,o['x'],o['y']): continue
        s=streak(c,o); acc=0; cnt=0
        for i in range(NS):
            t=(i+0.5)/NS; px=s['x1']+(s['x2']-s['x1'])*t; py=s['y1']+(s['y2']-s['y1'])*t
            if abs(px)>BR or abs(py)>BR: continue   # only what is on screen counts
            cnt+=1
            if sees(c,px,py): acc+=weights(m,px,py)[c]
        lst.append((acc/max(cnt,1),c))
    lst.sort(reverse=True)
    best=lst[0][0] if lst else 0; second=lst[1][0] if len(lst)>1 else 0
    return dict(best=best,second=second,score=max(0,min(1,best-0.5*second)),owner=lst[0][1] if lst else None)
def status(r):
    if r['best']<0.55: return 'LOST'
    if r['second']>0.3: return 'DOUBLED'
    if r['best']<0.9: return 'CUT'
    return 'VISIBLE'
def relevant(o):
    s=set()
    for c in CAM_IDS:
        if not sees(c,o['x'],o['y']): continue
        st=streak(c,o)
        for i in range(7):
            t=i/6; k=corner_of(st['x1']+(st['x2']-st['x1'])*t, st['y1']+(st['y2']-st['y1'])*t)
            if k>=0: s.add(k)
    return s
def make_method(kind,band=6,sigma=0.0):
    return dict(kind=kind,band=band,sigma=sigma,seams=[45.0]*4,corridors=[],owners={},motion=0.0)
def update_method(m,est,dt):
    if m['kind']=='dynamic':
        for k in range(4):
            rel=[o for o in est if k in relevant(o)]; prev=m['seams'][k]; target=45.0
            if rel:
                bc=1e9; phi=SEAM_MIN
                while phi<=SEAM_MAX+1e-6:
                    m['seams'][k]=phi; cost=sum(1-eval_obj(m,o)['score'] for o in rel)+0.002*abs(phi-prev)+0.001*abs(phi-45)
                    if cost<bc: bc=cost; target=phi
                    phi+=2.5
            step=max(-90*dt,min(90*dt,target-prev)); m['seams'][k]=prev+step
    elif m['kind']=='shape':
        cor=[]
        for o in est:
            cams=[c for c in CAM_IDS if sees(c,o['x'],o['y'])]
            if not cams: continue
            st={c:streak(c,o) for c in cams}; owner=min(cams,key=lambda c:st[c]['d'])
            prev=m['owners'].get(o['id'])
            if prev in cams and prev!=owner and st[prev]['d']<st[owner]['d']*1.15: owner=prev
            m['owners'][o['id']]=owner
            for c in cams:
                s=st[c]; cor.append(dict(owner=owner,x1=s['x1'],y1=s['y1'],dx=s['dx'],dy=s['dy'],len=s['len'],
                    rad=o['r']+0.12+2*m['sigma'],slope=o['r']/s['d']+0.02))
        m['corridors']=cor

# ---------------- ground texture ----------------
TR=12.0; TN=1200
def make_ground():
    rng=np.random.default_rng(3)
    base=np.full((TN,TN,3),(92,96,99),np.float32)
    n=rng.normal(0,1,(TN//4,TN//4)).astype(np.float32); n=cv2.resize(n,(TN,TN),interpolation=cv2.INTER_CUBIC)
    base+=n[...,None]*5+rng.normal(0,4,(TN,TN,1)).astype(np.float32)
    P=lambda x,y:(int((x+TR)/(2*TR)*TN),int((y+TR)/(2*TR)*TN))
    img=np.clip(base,0,255).astype(np.uint8)
    W=(222,222,215); lw=max(2,int(0.12/(2*TR)*TN))
    for x in (-1.45,1.45,4.2,-4.2):   # parking stall side lines
        cv2.line(img,P(x,-2.9),P(x,3.2),W,lw)
    for xs in ((1.45,4.2),(-4.2,-1.45)): cv2.line(img,P(xs[0],3.2),P(xs[1],3.2),W,lw)
    cv2.line(img,P(-1.45,3.2),P(1.45,3.2),W,lw)
    for y0 in np.arange(-11,11,2.0):  # dashed aisle line
        cv2.line(img,P(-7.5,y0),P(-7.5,y0+1.0),W,lw); cv2.line(img,P(7.5,y0),P(7.5,y0+1.0),W,lw)
    for i in range(7):  # zebra in front
        x=-3.0+i*1.0; cv2.rectangle(img,P(x,-6.6),P(x+0.5,-5.4),(210,210,200),-1)
    cv2.circle(img,P(3.2,-4.6),int(0.35/(2*TR)*TN),(70,72,74),-1)
    return img
GROUND=make_ground()
def ground_color(gx,gy):
    gx=gx.ravel(); gy=gy.ravel(); n=gx.size; W=2048; rows=(n+W-1)//W; pad=rows*W-n
    u=np.pad(((gx+TR)/(2*TR)*TN).astype(np.float32),(0,pad)).reshape(rows,W)
    v=np.pad(((gy+TR)/(2*TR)*TN).astype(np.float32),(0,pad)).reshape(rows,W)
    out=cv2.remap(GROUND,u,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(92,96,99))
    return out.reshape(-1,3)[:n].astype(np.float32)
LIGHT=norm(np.array([0.45,-0.55,0.7]))
SKY=np.array([205,190,160],np.float32)  # BGR-ish warm sky
def shade_box(nax,sign):
    n=np.zeros(3); n[nax]=sign; return 0.55+0.45*max(0,float(np.dot(n,LIGHT)))
CAR_COL=np.array([150,120,90],np.float32)  # BGR: steel blue

def trace_static(origin, D):
    D=D.reshape(-1,3); o=origin; n=D.shape[0]
    col=np.empty((n,3),np.float32); col[:]=SKY; tbest=np.full(n,np.inf,np.float32)
    dz=D[:,2]; hitg=dz<-1e-6; tg=np.where(hitg,-o[2]/np.where(hitg,dz,-1),np.inf).astype(np.float32)
    gx=(o[0]+tg*D[:,0]).astype(np.float32); gy=(o[1]+tg*D[:,1]).astype(np.float32)
    idx=np.where(hitg)[0]
    gc=ground_color(gx[idx].reshape(-1,1),gy[idx].reshape(-1,1)).reshape(-1,3)
    fog=np.clip((tg[idx]-12)/25,0,0.6)[:,None]; col[idx]=gc*(1-fog)+SKY*fog; tbest[idx]=tg[idx]
    lo=np.array([-HX,-HY,0.0]); hi=np.array([HX,HY,CAR_H])
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(lo-o)/D; t2=(hi-o)/D
    tmin=np.minimum(t1,t2); tmax=np.maximum(t1,t2)
    tn=np.nanmax(tmin,axis=1); tf=np.nanmin(tmax,axis=1); ax=np.nanargmax(tmin,axis=1)
    hb=(tn<tf)&(tn>1e-4)&(tn<tbest)
    if hb.any():
        k=hb.sum(); axs=ax[hb]; sgn=np.sign(-D[hb,axs])
        nrm=np.zeros((k,3),np.float32); nrm[np.arange(k),axs]=sgn; f=(0.55+0.45*np.clip(nrm@LIGHT,0,None))[:,None]
        zz=o[2]+tn[hb]*D[hb,2]; win=(zz>0.95)&(zz<1.35)&(axs!=2)
        base=np.where(win[:,None],np.array([70,60,55],np.float32),CAR_COL)
        base[axs==2]=np.array([120,95,72],np.float32)
        col[hb]=base*f; tbest[hb]=tn[hb]
    ground=hitg&~hb
    return dict(col=col,t=tbest,gx=gx,gy=gy,ground=ground,D=D,o=o)
def body_parts(ob):
    if ob.get('kind')=='Trolley':
        x,y=ob['x'],ob['y']; g=(150,150,155)
        return [(x,y,0.3,0.12,1.0,(60,150,215)),(x-0.2,y-0.2,0.05,0,0.12,g),(x+0.2,y-0.2,0.05,0,0.12,g),(x-0.2,y+0.2,0.05,0,0.12,g),(x+0.2,y+0.2,0.05,0,0.12,g)]
    if 'bands' not in ob or ob.get('pole'): return [(ob['x'],ob['y'],ob['r'],0,ob['h'],(60,200,230))]
    k=ob['h']/1.75; x,y=ob['x'],ob['y']; B=ob['bands']
    leg,shirt,skin=B[1][2],B[2][2],B[3][2]
    return [(x-0.1*k,y,0.08*k,0,0.86*k,leg),(x+0.1*k,y,0.08*k,0,0.86*k,leg),
            (x,y,0.19*k,0.84*k,1.46*k,shirt),(x-0.25*k,y,0.055*k,0.9*k,1.44*k,shirt),(x+0.25*k,y,0.055*k,0.9*k,1.44*k,shirt),
            (x,y,0.05*k,1.44*k,1.52*k,skin),(x,y,0.1*k,1.52*k,1.75*k,skin)]
def trace_dynamic(st, objs, shape):
    col=st['col'].copy(); tb=st['t'].copy(); D=st['D']; o=st['o']
    g=st['ground']; gx=st['gx']; gy=st['gy']
    L2=np.array([LIGHT[0],LIGHT[1]]); L2=L2/np.linalg.norm(L2)
    for ob in objs:
        px=gx-ob['x']; py=gy-ob['y']; s=-(px*L2[0]+py*L2[1]); p=np.abs(px*L2[1]-py*L2[0])
        m=g&(s>-ob['r'])&(s<ob['h']*0.7)&(p<ob['r']*1.1); col[m]*=0.62
    for ob in objs:
        for pt in body_parts(ob):
            ox=o[0]-pt[0]; oy=o[1]-pt[1]; r=pt[2]; a=D[:,0]**2+D[:,1]**2; b=2*(ox*D[:,0]+oy*D[:,1]); c=ox*ox+oy*oy-r*r
            disc=b*b-4*a*c; ok=disc>0
            t=np.where(ok,(-b-np.sqrt(np.where(ok,disc,0)))/(2*np.where(a>1e-9,a,1)),np.inf)
            z=o[2]+t*D[:,2]; hit=ok&(t>1e-4)&(z>=pt[3])&(z<=pt[4])&(t<tb)
            if hit.any():
                nx=(ox+t[hit]*D[hit,0])/r; ny=(oy+t[hit]*D[hit,1])/r
                f=(0.45+0.55*np.clip(nx*LIGHT[0]+ny*LIGHT[1]+0.3,0,1))[:,None]
                col[hit]=np.array(pt[5],np.float32)*f; tb[hit]=t[hit]
    return np.clip(col,0,255).astype(np.uint8).reshape(*shape,3)

# fisheye
FE_N=400; FE_F=(FE_N/2)/math.radians(96)
def fisheye_dirs(cam):
    u,v=np.meshgrid(np.arange(FE_N)+0.5,np.arange(FE_N)+0.5)
    x=u-FE_N/2; y=v-FE_N/2; r=np.hypot(x,y); th=r/FE_F; ph=np.arctan2(y,x)
    dc=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)],-1)
    return (dc@cam.R).astype(np.float32), th<math.radians(96)
def fisheye_proj(cam,P):
    d=P-cam.pos; dc=d@cam.R.T; th=np.arctan2(np.hypot(dc[...,0],dc[...,1]),dc[...,2]); ph=np.arctan2(dc[...,1],dc[...,0])
    return (FE_N/2+FE_F*th*np.cos(ph)).astype(np.float32),(FE_N/2+FE_F*th*np.sin(ph)).astype(np.float32),th

# ---------------- BEV ----------------
_c=(np.arange(BN)+0.5)/BN*2*BR-BR
BX,BY=np.meshgrid(_c,_c)
BEV_MAP={}; BEV_OK={}
for _k,_cam in CAMS.items():
    P=np.stack([BX,BY,np.zeros_like(BX)],-1)
    mu,mv,th=fisheye_proj(_cam,P); BEV_MAP[_k]=(mu,mv); BEV_OK[_k]=th<math.radians(95)
SEES={'F':BY<=-HY,'B':BY>=HY,'L':BX<=-HX,'R':BX>=HX}
for _k in CAM_IDS: SEES[_k]=SEES[_k]&BEV_OK[_k]
KGRID=np.full(BX.shape,-1); AGRID=np.zeros(BX.shape,np.float32)
for _i,C in enumerate(CORNERS):
    m=(SEES[C['a']])&(SEES[C['b']])&((BX-C['cx'])*C['sx']>=0)&((BY-C['cy'])*C['sy']>=0)
    KGRID[m]=_i; u=(BX-C['cx'])*C['sx']; v=(BY-C['cy'])*C['sy']; AGRID[m]=(np.arctan2(u,v)*DEG)[m]
CARMASK=(np.abs(BX)<HX+0.02)&(np.abs(BY)<HY+0.02)
def sstep_np(e0,e1,x):
    t=np.clip((x-e0)/(e1-e0),0,1); return t*t*(3-2*t)
def cor_mask(c):
    px=BX-c['x1']; py=BY-c['y1']; s=px*c['dx']+py*c['dy']; p=np.abs(px*c['dy']-py*c['dx'])
    return ((s>=0)&(s<=c['len'])&(p<=c['rad']+c['slope']*s))|(np.hypot(px,py)<=c['rad'])
def weight_grid(m):
    W={c:np.zeros(BX.shape,np.float32) for c in CAM_IDS}
    rest=KGRID<0; taken=np.zeros(BX.shape,bool)
    for c in CAM_IDS:
        mm=rest&SEES[c]&~taken; W[c][mm]=1; taken|=mm
    for i,C in enumerate(CORNERS):
        mm=KGRID==i; wa=1-sstep_np(m['seams'][i]-m['band'],m['seams'][i]+m['band'],AGRID[mm])
        W[C['a']][mm]=wa; W[C['b']][mm]=1-wa
    assigned=np.zeros(BX.shape,bool)
    bg=m.get('blobgrid')
    if bg is not None:
        for ci,c in enumerate(CAM_IDS):
            mm=(bg==ci)&SEES[c]
            for k in CAM_IDS: W[k][mm]=0
            W[c][mm]=1; assigned|=mm
    for c in m['corridors']:
        mm=cor_mask(c)&SEES[c['owner']]&~assigned
        for k in CAM_IDS: W[k][mm]=0
        W[c['owner']][mm]=1; assigned|=mm
    return W, assigned
def warp_all(camimgs):
    return {c:cv2.remap(camimgs[c],*BEV_MAP[c],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for c in CAM_IDS}
def stitch(warps,m):
    W,owned=weight_grid(m); out=np.zeros(BX.shape+(3,),np.float32)
    for c in CAM_IDS:
        out+=W[c][...,None]*warps[c]
    out[CARMASK]=0
    return np.clip(out,0,255).astype(np.uint8), owned
def w2p(x,y): return (int((x+BR)/(2*BR)*BN), int((y+BR)/(2*BR)*BN))

# ================= v2: proposed method with ladder =================
def risk_weight(o):
    base={'Child':1.6,'Pedestrian':1.0,'Pole':0.5,'Trolley':0.8}.get(o.get('kind'),1.0)
    d=math.hypot(max(0,abs(o['x'])-HX),max(0,abs(o['y'])-HY))
    return base*(1.0+1.0/(0.5+d))
def corridors_for(o,owner,m):
    out=[]
    for c in CAM_IDS:
        if not sees(c,o['x'],o['y']): continue
        s=streak(c,o)
        out.append(dict(owner=owner,x1=s['x1'],y1=s['y1'],dx=s['dx'],dy=s['dy'],len=s['len'],
                        rad=o['r']+m.get('m0',M0)+2*m['sigma']+2*o.get('sig_extra',0),slope=o['r']/s['d']+0.02))
    return out
def explain_mask(est,sigma):
    """region explained by tracked objects (their projections from every camera)"""
    mk=np.zeros(BX.shape,bool); dm=dict(sigma=sigma)
    for o in est:
        for c in corridors_for(o,'F',dm): mk|=cor_mask(c)
    return mk
def make_prop(band=6,sigma=0.0):
    m=make_method('dynamic',band,sigma); m['kind']='prop'; m['level']=1; m['owners']={}; m['blobgrid']=None; m['nblobs']=0; m['worst_id']=None
    return m
def update_prop(m,est,dt):
    # step A: behave like a turning straight seam (cheapest option)
    saved=m['corridors']; m['corridors']=[]
    m['kind']='dynamic'; update_method(m,est,dt); m['kind']='prop'
    rel=[o for o in est if relevant(o)]
    if not rel:
        m['level']=1; m['worst_id']=None; return
    NSsave=globals()['NS']
    sc=[eval_obj(m,o)['score'] for o in rel]
    if min(sc)>=m.get('l2thr',L2_THR) and not any(o.get('coasting') for o in rel):
        m['level']=2; m['worst_id']=None; return
    # step B: ownership labeling over objects that have two candidate cameras
    cand={}; occ=m.get('occl') or set()
    for o in rel:
        cams=[c for c in CAM_IDS if sees(c,o['x'],o['y'])]
        vis=[c for c in cams if (o['id'],c) not in occ]
        cand[o['id']]=vis if vis else cams
    multi=[o for o in rel if len(cand[o['id']])>1]
    order=sorted(rel,key=risk_weight,reverse=True)
    best=None
    globals()['NS']=10
    import itertools
    for combo in itertools.product(*[cand[o['id']] for o in multi]):
        own={o['id']:c for o,c in zip(multi,combo)}
        for o in rel:
            if o['id'] not in own: own[o['id']]=cand[o['id']][0]
        cors=[]
        for o in order: cors+=corridors_for(o,own[o['id']],m)
        m['corridors']=cors
        cost=0; worst=(2,None)
        for o in rel:
            r=eval_obj(m,o); w=risk_weight(o); cost+=w*(1-r['score'])
            if r['score']<worst[0]: worst=(r['score'],o['id'])
            st={c:streak(c,o)['d'] for c in cand[o['id']]}
            cost+=m.get('lam_u',LAM_U)*st[own[o['id']]]/min(st.values())
            if m['owners'].get(o['id']) not in (None,own[o['id']]): cost+=m.get('lam_s',LAM_S)
        if best is None or cost<best[0]: best=(cost,own,cors,worst)
    globals()['NS']=NSsave
    _,own,cors,worst=best
    m['corridors']=cors; m['owners'].update(own)
    n_in=len([o for o in rel if len(cand[o['id']])>1])
    resid=min(eval_obj(m,o)['score'] for o in rel)
    coast=[o for o in rel if o.get('coasting')]
    if coast: m['level']=5; m['worst_id']=coast[0]['id']
    elif resid<L5_THR: m['level']=5; m['worst_id']=worst[1]
    else: m['level']=4 if n_in>=2 else 3; m['worst_id']=None

NEAR=np.hypot(np.maximum(0,np.abs(BX)-HX),np.maximum(0,np.abs(BY)-HY))<4.5
def disagreement_blobs(warps,est,sigma,min_area=120,thr_seed=60,thr_grow=30):
    """camera-disagreement check inside overlaps: 3D content that no tracked object explains.
    seeds: strong disagreement near the car; region grows along weaker disagreement (hysteresis)."""
    grid=np.full(BX.shape,-1,np.int8); expl=explain_mask(est,sigma); n=0
    for i,C in enumerate(CORNERS):
        ovf=(KGRID==i)
        if not ovf.any(): continue
        a=cv2.GaussianBlur(warps[C['a']],(15,15),0).astype(np.float32); b=cv2.GaussianBlur(warps[C['b']],(15,15),0).astype(np.float32)
        dif=np.abs(a-b).mean(-1)
        seed=ovf&NEAR&(dif>thr_seed)&~expl
        seed=cv2.morphologyEx(seed.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8)).astype(bool)
        if seed.sum()<min_area: continue
        grow=(ovf&(dif>min(thr_grow,thr_seed-5))&~expl).astype(np.uint8)
        k,lab,stats,_=cv2.connectedComponentsWithStats(grow)
        for li in range(1,k):
            comp=lab==li
            if (comp&seed).sum()<min_area: continue
            reg=cv2.dilate(comp.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)&ovf
            ys,xs=np.nonzero(comp&seed); px=BX[ys,xs]; py=BY[ys,xs]
            dcar=np.hypot(np.maximum(0,np.abs(px)-HX),np.maximum(0,np.abs(py)-HY)); kmin=np.argmin(dcar)
            bx,by=px[kmin],py[kmin]
            owner=min((C['a'],C['b']),key=lambda c:math.hypot(bx-CAMS[c].pos[0],by-CAMS[c].pos[1]))
            grid[reg]=CAM_IDS.index(owner); n+=1
    return grid,n

# ======== review round 1 additions ========
RGRID=np.zeros(BX.shape,np.float32)
for _i,C in enumerate(CORNERS):
    _m=KGRID==_i; RGRID[_m]=np.hypot((BX-C['cx'])*C['sx'],(BY-C['cy'])*C['sy'])[_m]
_wg_orig=weight_grid
def weight_grid(m):
    cv=m.get('curve')
    if cv is None: return _wg_orig(m)
    saved=m['seams']; W,assigned=None,None
    # temporarily evaluate per-corner curved seams
    Wd={c:np.zeros(BX.shape,np.float32) for c in CAM_IDS}
    rest=KGRID<0; taken=np.zeros(BX.shape,bool)
    for c in CAM_IDS:
        mm=rest&SEES[c]&~taken; Wd[c][mm]=1; taken|=mm
    for i,C in enumerate(CORNERS):
        mm=KGRID==i
        phi=np.interp(RGRID[mm],CURVE_R,cv[i]) if cv[i] is not None else np.full(mm.sum(),m['seams'][i])
        wa=1-sstep_np(phi-m['band'],phi+m['band'],AGRID[mm]); Wd[C['a']][mm]=wa; Wd[C['b']][mm]=1-wa
    return Wd,np.zeros(BX.shape,bool)
def cor_inside_np(c,X,Y):
    px=X-c['x1']; py=Y-c['y1']; s=px*c['dx']+py*c['dy']; p=np.abs(px*c['dy']-py*c['dx'])
    return ((s>=0)&(s<=c['len'])&(p<=c['rad']+c['slope']*s))|(np.hypot(px,py)<=c['rad'])
def make_curve(band=6,sigma=0.0):
    m=make_method('dynamic',band,sigma); m['kind']='curve'; m['curve']=[None]*4; return m
CURVE_STEPS=3
def update_curve(m,est,dt,lam_s=0.004,lam_c=0.0005):
    """B4: mask-aware curved seam. Seam angle is a function of distance from the corner,
    found by dynamic programming that minimises risk-weighted crossings of the projected
    (dilated) object masks of both cameras (the union of footprints), as in object-avoiding
    seam disclosures that allow arbitrary seam shapes."""
    RR,AA=np.meshgrid(CURVE_R,CURVE_A,indexing='ij')
    for k,C in enumerate(CORNERS):
        rel=[o for o in est if k in relevant(o)]
        prev=m['curve'][k] if m['curve'][k] is not None else np.full(len(CURVE_R),45.0)
        if not rel: target=np.full(len(CURVE_R),45.0)
        else:
            X=C['cx']+C['sx']*RR*np.sin(np.radians(AA)); Y=C['cy']+C['sy']*RR*np.cos(np.radians(AA))
            cost=np.zeros(RR.shape)
            for o in rel:
                w=risk_weight(o); ins=np.zeros(RR.shape,bool)
                for cc in corridors_for(o,'F',m): ins|=cor_inside_np(cc,X,Y)
                cost+=w*ins
            nA=len(CURVE_A); nR=len(CURVE_R); V=np.full((nR,nA),1e9); P=np.zeros((nR,nA),int)
            V[0]=cost[0]+lam_c*np.abs(CURVE_A-45)
            for i in range(1,nR):
                best=np.full(nA,1e9); arg=np.zeros(nA,int)
                for d in range(-CURVE_STEPS,CURVE_STEPS+1):
                    src=np.roll(V[i-1],d); idx=(np.arange(nA)-d)
                    valid=(idx>=0)&(idx<nA); cand=np.where(valid,src+lam_s*abs(d)*2.5,1e9)
                    upd=cand<best; best[upd]=cand[upd]; arg[upd]=idx[upd]
                V[i]=best+cost[i]+lam_c*np.abs(CURVE_A-45); P[i]=arg
            j=int(np.argmin(V[-1])); path=np.zeros(nR)
            for i in range(nR-1,-1,-1): path[i]=CURVE_A[j]; j=P[i,j]
            target=path
        step=np.clip(target-prev,-90*dt,90*dt); m['curve'][k]=prev+step
def update_prop_noassign(m,est,dt):
    """A5: dynamic base seam + dilated ownership regions, independent nearest owners (no joint assignment)."""
    m['corridors']=[]; m['kind']='dynamic'; update_method(m,est,dt); m['kind']='prop_na'
    rel=[o for o in est if relevant(o)]
    if not rel: m['level']=1; return
    if min(eval_obj(m,o)['score'] for o in rel)>=0.95 and not any(o.get('coasting') for o in rel): m['level']=2; return
    cor=[]
    for o in sorted(rel,key=risk_weight,reverse=True):
        cams=[c for c in CAM_IDS if sees(c,o['x'],o['y'])]
        vis=[c for c in cams if (o['id'],c) not in (m.get('occl') or set())]; cams=vis if vis else cams
        st={c:streak(c,o)['d'] for c in cams}; own=min(cams,key=lambda c:st[c])
        prev=m['owners'].get(o['id'])
        if prev in cams and prev!=own and st[prev]<st[own]*1.15: own=prev
        m['owners'][o['id']]=own; cor+=corridors_for(o,own,m)
    m['corridors']=cor; m['level']=3
# ---------- pixel-level (rendered) evaluation ----------
def trace_ids(st,objs):
    tb=st['t'].copy(); D=st['D']; o=st['o']; ids=np.full(tb.shape,-1,np.int16)
    for oi,ob in enumerate(objs):
        for pt in body_parts(ob):
            ox=o[0]-pt[0]; oy=o[1]-pt[1]; r=pt[2]; a=D[:,0]**2+D[:,1]**2; b=2*(ox*D[:,0]+oy*D[:,1]); c=ox*ox+oy*oy-r*r
            disc=b*b-4*a*c; ok=disc>0
            t=np.where(ok,(-b-np.sqrt(np.where(ok,disc,0)))/(2*np.where(a>1e-9,a,1)),np.inf)
            z=o[2]+t*D[:,2]; hit=ok&(t>1e-4)&(z>=pt[3])&(z<=pt[4])&(t<tb)
            tb[hit]=t[hit]; ids[hit]=oi
    return ids
def pixel_masks(stat,objs):
    """per camera: top-view pixels at which that camera's projected image shows object o"""
    M={}
    for c in CAM_IDS:
        idimg=trace_ids(stat[c][0],objs).reshape(FE_N,FE_N).astype(np.float32); idimg[~stat[c][1]]=-1
        w=cv2.remap(idimg,*BEV_MAP[c],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=-1)
        w[~SEES[c]]=-1; w[CARMASK]=-1; M[c]=w
    return M
def pixel_eval(M,W,objs):
    res=[]
    for oi,o in enumerate(objs):
        vals=[]
        for c in CAM_IDS:
            mk=M[c]==oi; n=mk.sum()
            if n<15: continue
            vals.append(float(W[c][mk].sum()/n))
        vals.sort(reverse=True); b=vals[0] if vals else 0; s2=vals[1] if len(vals)>1 else 0
        res.append(dict(best=b,second=s2,score=max(0,min(1,b-0.5*s2)),hidden=not vals))
    return res

# ---------- occlusion awareness (added after pixel-level evaluation, review round 1) ----------
def occlusions(est):
    """(object id, camera) pairs where another estimated object blocks the camera's line of sight"""
    occ=set()
    for o in est:
        for c in CAM_IDS:
            if not sees(c,o['x'],o['y']): continue
            p=CAMS[c].pos[:2]; v=np.array([o['x'],o['y']])-p; d=np.linalg.norm(v); u=v/d
            for j in est:
                if j is o or j['r']<0.5*o['r'] or j['h']<0.5*o['h']: continue
                w=np.array([j['x'],j['y']])-p; t=float(w@u); lat=abs(w[0]*u[1]-w[1]*u[0])
                if 0<t<d-0.1 and lat<j['r']+0.6*o['r']: occ.add((o['id'],c)); break
    return occ
_eval_orig=eval_obj
def eval_obj(m,o):
    occ=m.get('occl')
    if not occ: return _eval_orig(m,o)
    lst=[]
    for c in CAM_IDS:
        if not sees(c,o['x'],o['y']) or (o['id'],c) in occ: continue
        s=streak(c,o); acc=0; cnt=0
        for i in range(NS):
            t=(i+0.5)/NS; px=s['x1']+(s['x2']-s['x1'])*t; py=s['y1']+(s['y2']-s['y1'])*t
            if abs(px)>BR or abs(py)>BR: continue
            cnt+=1
            if sees(c,px,py): acc+=weights(m,px,py)[c]
        lst.append((acc/max(cnt,1),c))
    lst.sort(reverse=True)
    best=lst[0][0] if lst else 0; second=lst[1][0] if len(lst)>1 else 0
    return dict(best=best,second=second,score=max(0,min(1,best-0.5*second)),owner=lst[0][1] if lst else None)

def cam_visibility(stat,objs,frac=0.4,minpx=30):
    """simulated per-camera detection: object o is detected in camera c if at least `frac` of its
    unoccluded fisheye pixels are visible (occlusion by other objects) and >= minpx pixels"""
    occ=set()
    for c in CAM_IDS:
        st=stat[c][0]; valid=stat[c][1].ravel()
        D=st['D']; o0=st['o']; tb=st['t'].copy(); ids=np.full(tb.shape,-1,np.int16); alone=np.zeros(len(objs))
        for oi,ob in enumerate(objs):
            hit_any=np.zeros(tb.shape,bool)
            for pt in body_parts(ob):
                ox=o0[0]-pt[0]; oy=o0[1]-pt[1]; r=pt[2]; a=D[:,0]**2+D[:,1]**2; b=2*(ox*D[:,0]+oy*D[:,1]); cc=ox*ox+oy*oy-r*r
                disc=b*b-4*a*cc; ok=disc>0
                t=np.where(ok,(-b-np.sqrt(np.where(ok,disc,0)))/(2*np.where(a>1e-9,a,1)),np.inf)
                z=o0[2]+t*D[:,2]; h=ok&(t>1e-4)&(z>=pt[3])&(z<=pt[4])&(t<st['t'])&valid
                hit_any|=h; closer=h&(t<tb); tb[closer]=t[closer]; ids[closer]=oi
            alone[oi]=hit_any.sum()
        for oi,ob in enumerate(objs):
            if not sees(c,ob['x'],ob['y']): continue
            v=(ids==oi).sum()
            if alone[oi]<minpx or v<max(minpx,frac*alone[oi]): occ.add((ob['id'],c))
    return occ
