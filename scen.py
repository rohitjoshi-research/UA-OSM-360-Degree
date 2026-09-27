import math
PED_B=[(0,0.1,(40,40,45)),(0.1,0.4,(110,60,50)),(0.4,1.5,(30,110,235)),(1.5,1.75,(130,160,200))]
CHILD_B=[(0,0.08,(40,40,45)),(0.08,0.3,(120,120,125)),(0.3,0.9,(60,200,120)),(0.9,1.1,(130,160,200))]
def ped(id,name,x,y): return dict(id=id,name=name,kind='Pedestrian',x=x,y=y,h=1.75,r=0.25,bands=PED_B,nx=0.0,ny=0.0)
def child(id,name,x,y): return dict(id=id,name=name,kind='Child',x=x,y=y,h=1.1,r=0.2,bands=CHILD_B,nx=0.0,ny=0.0)
def lerp(a,b,u): return a+(b-a)*u
def ease(u): u=max(0,min(1,u)); return u*u*(3-2*u)
def sA_make(): return [ped('p1','Pedestrian',-4.2,-2.0)]
def sA_step(o,t):
    u=min(1,t/9.0); o[0]['x']=lerp(-4.2,-0.4,u); o[0]['y']=lerp(-2.0,-3.9,u)
def sB_make(): return [child('c1','Child',-3.2,-4.2)]
def sB_step(o,t):
    u=ease(t/4.0); o[0]['x']=lerp(-3.2,-1.14,u)+0.05*math.sin(t*0.9)*(t>4); o[0]['y']=lerp(-4.2,-2.48,u)
def sC_make(): return [child('c1','Child',-1.45,-2.55),ped('p2','Pedestrian',-1.7,-6.8)]
def sC_step(o,t):
    o[0]['x']=-1.45+0.04*math.sin(t); u=ease(t/5.0); o[1]['x']=lerp(-1.7,-1.45,u); o[1]['y']=lerp(-6.8,-4.0,u)
SCENES=[
 dict(title='Pedestrian walking past the front-left corner',dur=9.5,make=sA_make,step=sA_step,
      text=['A pedestrian crosses the zone that both the front','and the left camera see.',
            'The fixed line crops him as he passes the corner.','The turning line copes; the proposed merge keeps','him whole the entire time.']),
 dict(title='Child standing right at the corner',dur=9.0,make=sB_make,step=sB_step,
      text=['A child walks up and stops close to the bumper.','No straight line can avoid her here: the fixed','line loses her completely and the turning line','hits its angle limit.','The proposed merge bends around her.']),
 dict(title='Two people in the same corner zone',dur=10.0,make=sC_make,step=sC_step,
      text=['A child at the corner and an adult approaching.','Each needs the straight line turned a different','way, so the turning line sacrifices one of them.','The proposed merge shapes around both.']),
]
