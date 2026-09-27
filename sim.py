import numpy as np, cv2, math
HX, HY = 0.95, 2.3
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
def weights(m,x,y):
    out={'F':0.0,'B':0.0,'L':0.0,'R':0.0}
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
    phi=m['seams'][k]; b=m['band']; wa=1-sstep(phi-b,phi+b,a); out[C['a']]=wa; out[C['b']]=1-wa; return out
STREAK_CAP=7.0
def streak(c,o):
    C=CAMS[c]; dx=o['x']-C.pos[0]; dy=o['y']-C.pos[1]; d=math.hypot(dx,dy); dx/=d; dy/=d
    z=min(o['h'],0.8*C.h); L=min(d*z/(C.h-z),STREAK_CAP)
    return dict(x1=o['x'],y1=o['y'],x2=o['x']+dx*L,y2=o['y']+dy*L,dx=dx,dy=dy,len=L,d=d)
NS=18
def eval_obj(m,o):
    lst=[]
    for c in CAM_IDS:
        if not sees(c,o['x'],o['y']): continue
        s=streak(c,o); acc=0
        for i in range(NS):
            t=(i+0.5)/NS; px=s['x1']+(s['x2']-s['x1'])*t; py=s['y1']+(s['y2']-s['y1'])*t
            if sees(c,px,py): acc+=weights(m,px,py)[c]
        lst.append((acc/NS,c))
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
BR=7.2; BN=440
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
    assigned=np.zeros(BX.shape,bool); owned=np.zeros(BX.shape,bool)
    for c in m['corridors']:
        mm=cor_mask(c)&SEES[c['owner']]&~assigned
        for k in CAM_IDS: W[k][mm]=0
        W[c['owner']][mm]=1; assigned|=mm
    return W, assigned
def stitch(camimgs,m):
    W,owned=weight_grid(m); out=np.zeros(BX.shape+(3,),np.float32)
    for c in CAM_IDS:
        warped=cv2.remap(camimgs[c],*BEV_MAP[c],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        out+=W[c][...,None]*warped
    out[CARMASK]=0
    return np.clip(out,0,255).astype(np.uint8), owned
def w2p(x,y): return (int((x+BR)/(2*BR)*BN), int((y+BR)/(2*BR)*BN))
