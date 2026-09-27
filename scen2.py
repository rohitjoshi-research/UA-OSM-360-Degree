import math
from scen import PED_B, CHILD_B, lerp, ease
def ped(id,x,y,det=True): return dict(id=id,name='Pedestrian',kind='Pedestrian',x=x,y=y,h=1.75,r=0.25,bands=PED_B,nx=0.0,ny=0.0,det=det)
def child(id,x,y,det=True): return dict(id=id,name='Child',kind='Child',x=x,y=y,h=1.1,r=0.2,bands=CHILD_B,nx=0.0,ny=0.0,det=det)
def pole(id,x,y): return dict(id=id,name='Pole',kind='Pole',x=x,y=y,h=2.2,r=0.09,pole=True,nx=0.0,ny=0.0,det=True)
def trolley(id,x,y): return dict(id=id,name='Trolley',kind='Trolley',x=x,y=y,h=1.0,r=0.3,nx=0.0,ny=0.0,det=False)
def pp(t,T):
    u=(t%(2*T))/T; return u if u<1 else 2-u
S=[]
def A_make(): return [ped('p1',-4.2,-2.0)]
def A_step(o,t): u=min(1,t/7.5); o[0]['x']=lerp(-4.2,-0.4,u); o[0]['y']=lerp(-2.0,-3.9,u)
S.append(dict(title='Pedestrian walking past the corner',dur=8.0,make=A_make,step=A_step,
  text=['One person, open space. A turned straight line','is enough, so the proposed system stays on','ladder level 2: the calmest image that works.']))
def B_make(): return [child('c1',-3.2,-4.2)]
def B_step(o,t):
    u=ease(t/4.0); o[0]['x']=lerp(-3.2,-1.14,u)+0.05*math.sin(t*0.9)*(t>4); o[0]['y']=lerp(-4.2,-2.48,u)
S.append(dict(title='Child standing right at the corner',dur=8.0,make=B_make,step=B_step,
  text=['The child stops at the bumper. No straight','line can avoid her, so the system climbs to','level 3 and bends the merge line around her.']))
def C_make(): return [child('c1',-1.2,-2.55),ped('p2',-1.6,-6.8),ped('p3',-5.2,-2.7),pole('q4',-2.7,-4.5),child('c5',-3.8,-5.6)]
def C_step(o,t):
    o[0]['x']=-1.2+0.04*math.sin(t)
    u=ease(t/5.0); o[1]['x']=lerp(-1.6,-1.5,u); o[1]['y']=lerp(-6.8,-3.9,u)
    u=ease((t-1)/5.0); o[2]['x']=lerp(-5.2,-2.4,u); o[2]['y']=lerp(-2.7,-3.0,u)
    u=ease((t-2)/6.0); o[4]['x']=lerp(-3.8,-2.2,u); o[4]['y']=lerp(-5.6,-4.1,u)
S.append(dict(title='Crowded corner: five objects in one merge zone',dur=11.0,lost={'c5':(5.6,7.6)},make=C_make,step=C_step,
  text=['Two children, two adults and a pole compete','for the same pixels. The system tests every','camera assignment and keeps groups together.','When the small child walks behind the adult,','tracking is lost for 2 s: level 5, the screen','says so instead of silently guessing.']))
def D_make(): return [trolley('t1',-1.9,-3.1),ped('p2',3.0,-5.6)]
def D_step(o,t):
    u=min(1,t/8.0); o[1]['x']=lerp(3.0,-0.2,u); o[1]['y']=lerp(-5.6,-5.2,u)
S.append(dict(title='Undetected object: a shopping trolley',dur=9.0,make=D_make,step=D_step,
  text=['The detector does not know trolleys. Where the','two cameras disagree and no tracked object','explains it, the proposed system treats it as','unknown 3D content and never hides it.']))
def E_make(): return [child('c1',-5.0,-1.4)]
def E_step(o,t):
    u=pp(max(0,t-0.3),2.2); u=ease(u); o[0]['x']=lerp(-5.0,0.3,u); o[0]['y']=lerp(-1.4,-4.6,u)
S.append(dict(title='Running child with detection delay',dur=7.0,make=E_make,step=E_step,
  text=['A child runs across the corner. Detections','arrive 100 ms late; the system predicts ahead','and widens the region along the motion.']))
SCENES=S
