import sys, numpy as np, math, json, subprocess
import sim2 as sm
import scen2
import render3base as R
from PIL import Image, ImageDraw
import e6
rng=np.random.default_rng(3)
COND={'k':None}
LOOK={'morning':((0.85,0.93,1.03),0.95),'afternoon':((1.05,1.05,1.05),1.05),'evening':((1.05,0.75,0.62),0.62),'night':((0.62,0.82,1.0),0.5)}
def cond_filter(img,depth):
    tod,wx=COND['k']; mul,g=LOOK[tod]; im=img.astype(np.float32)/255.0
    im=im*np.array(mul)*g
    if wx=='fog':
        f=0.85*(1-np.exp(-np.minimum(depth,60)/12.0)); fc=np.array([0.77,0.76,0.74])*g; im=im*(1-f[...,None])+fc*f[...,None]
    if wx=='rain':
        im=0.85*im+0.04*g; H,W=im.shape[:2]; st=np.zeros((H,W),np.float32)
        for _ in range(int(H*W/600)):
            x=rng.integers(0,W); y=rng.integers(0,H-20); st[y:y+rng.integers(6,18),x]=0.3
        im=im+st[...,None]*max(g,0.4)
    if tod=='night': im=im+rng.normal(0,0.02,im.shape)
    return (np.clip(im,0,1)*255).astype(np.uint8)
_td=sm.trace_dynamic
def trace_dynamic(st,objs,shape):
    img=_td(st,objs,shape)
    if COND['k'] is None: return img
    return cond_filter(img,st['t'].reshape(shape))
sm.trace_dynamic=trace_dynamic
sm.disagreement_blobs=lambda w,est,sig,**k:(np.full(sm.BX.shape,-1,np.int8),0)   # proof-of-concept rule off in the demo
from scen import PED_B, CHILD_B
def ped(i,x,y): return dict(id=i,name='Adult',kind='Pedestrian',x=x,y=y,h=1.75,r=0.25,bands=PED_B,nx=0.0,ny=0.0,det=True)
def child(i,x,y): return dict(id=i,name='Child',kind='Child',x=x,y=y,h=1.1,r=0.2,bands=CHILD_B,nx=0.0,ny=0.0,det=True)
def pole(i,x,y): return dict(id=i,name='Pole',kind='Pole',x=x,y=y,h=2.2,r=0.09,pole=True,nx=0.0,ny=0.0,det=True)
lerp=scen2.lerp; ease=scen2.ease
def walk(o,t,a,b,T): u=ease(t/T); o['x']=lerp(a[0],b[0],u); o['y']=lerp(a[1],b[1],u)
SC={
 'walk1':(lambda:[ped('p1',-4.2,-2.0)],lambda o,t:walk(o[0],t,(-4.2,-2.0),(-0.5,-3.9),4.5),'One adult walks past the corner'),
 'child1':(lambda:[child('c1',-3.0,-4.0)],lambda o,t:walk(o[0],t,(-3.0,-4.0),(-1.14,-2.48),3.0),'A child walks up to the bumper'),
 'two':(lambda:[child('c1',-1.4,-2.55),ped('p2',-1.8,-6.5)],lambda o,t:walk(o[1],t,(-1.8,-6.5),(-1.45,-4.0),3.5),'A child at the corner, an adult approaching'),
 'three':(lambda:[child('c1',-1.3,-2.6),ped('p2',-4.5,-2.8),pole('q3',-2.6,-4.3)],lambda o,t:walk(o[1],t,(-4.5,-2.8),(-2.6,-3.1),4.0),'A child, an adult and a pole'),
 'run':(lambda:[child('c1',-5.0,-1.4)],lambda o,t:walk(o[0],t,(-5.0,-1.4),(0.2,-4.6),2.8),'A child runs across the corner'),
 'crowd':(lambda:[child('c1',-1.2,-2.55),ped('p2',-1.6,-6.8),ped('p3',-5.2,-2.7),pole('q4',-2.7,-4.5),child('c5',-3.8,-5.6)],
          lambda o,t:(walk(o[1],t,(-1.6,-6.8),(-1.5,-3.9),4.0),walk(o[2],t,(-5.2,-2.7),(-2.4,-3.0),4.0),walk(o[4],t,(-3.8,-5.6),(-2.2,-4.1),4.0)),'Five objects crowd the corner'),
}
RES=json.load(open('cond_summary.json'))
TODS=['morning','afternoon','evening','night']; WXS=['clear','rain','fog']
PLAN=[('morning','clear','walk1'),('afternoon','clear','child1'),('evening','clear','two'),('night','clear','crowd'),
      ('morning','rain','three'),('afternoon','rain','run'),('evening','rain','crowd'),('night','rain','two'),
      ('morning','fog','child1'),('afternoon','fog','three'),('evening','fog','walk1'),('night','fog','crowd')]
import cond as CD
def seg(i,fps=10,dur=4.5,out=None):
    tod,wx,sk=PLAN[i]; COND['k']=(tod,wx); P=CD.params(tod,wx); R.SIGMA=P['sigma']
    make,step,desc=SC[sk]; objs=make()
    b=RES['B'][WXS.index(wx)][TODS.index(tod)]; u=RES['U'][WXS.index(wx)][TODS.index(tod)]
    S=dict(title=f'{tod.capitalize()} · {wx}: {desc.lower()}',dur=dur,
      text=[f'{tod.capitalize()} · {wx}: ±{P["sigma"]*100:.0f} cm detection error, {P["pdrop"]:.2f} track losses/s',
            f'Scene: {desc} ({len(objs)} object{"s" if len(objs)>1 else ""}).',
            'Red circle = merge line erases a person; green = shown whole.',
            f'Catalog result: straight seam loses an object in {b:.1f} %',f'of frames in this condition, UA-OSM in {u:.1f} %.'])
    lost={}
    if sk=='crowd' and tod in ('night',) : lost={'c5':(2.0,3.4)}
    scen2.SCENES=[S]*12
    ms={'fixed':sm.make_method('fixed',6,R.SIGMA),'dynamic':sm.make_method('dynamic',6,R.SIGMA),'prop':sm.make_prop(6,R.SIGMA)}
    hist={k:[] for k in ms}; views={}; trk=R.Tracker(40+i); dt=1/fps; t=0.0; frames=[]
    p=None
    if out: p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-','-c:v','libx264','-preset','medium','-crf','21','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
    for fi in range(int(dur*fps)):
        step(objs,t); est=trk.step(objs,t,dt,lost)
        ms['prop']['occl']=sm.cam_visibility(e6.STATLO,objs,minpx=4)
        img,ev=R.frame(i,S,t,objs,est,ms,hist,views)
        for k in ms: hist[k].append((t,min(r['best'] for r in ev[k])))
        if p: p.stdin.write(np.asarray(img).tobytes())
        else: img.save(f'test_seg{i}.png'); break
        t+=dt
    if p:
        for _ in range(8): p.stdin.write(np.asarray(img).tobytes())
        p.stdin.close(); p.wait()
if __name__=='__main__':
    if sys.argv[1]=='test':
        import time; t0=time.time(); seg(int(sys.argv[2])); print('frame s',round(time.time()-t0,1))
    else:
        for a in sys.argv[1:]: seg(int(a),out=f'd3_{int(a):02d}.mp4'); print('done',a,flush=True)
