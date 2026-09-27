import sim, scen, numpy as np, cv2, math, subprocess, sys, random
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1920,1080,15
FD='/usr/share/fonts/truetype/dejavu/'
def F(sz,b=False): return ImageFont.truetype(FD+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),sz)
f12,f14,f16,f18,f20,f22,f26,f30,f40,f56=F(12),F(14),F(16),F(18),F(20),F(22),F(26),F(30,True),F(40,True),F(56,True)
fb16,fb18,fb20,fb22,fb26=F(16,True),F(18,True),F(20,True),F(22,True),F(26,True)
BG=(16,20,25);PANEL=(26,31,38);LINE=(48,56,66);TXT=(236,240,243);MUT=(150,160,170)
RED=(232,76,61);AMB=(240,168,48);GRN=(46,196,120);YEL=(255,212,64)
METH=[('fixed','Problem: fixed merge line','Used in most systems today',RED),
      ('dynamic','Other fix: turning straight line','Line rotates 15-75° to dodge objects',AMB),
      ('shape','Proposed: object-shaped merge','Line bends around each object',GRN)]
STCOL={'LOST':RED,'CUT':AMB,'DOUBLED':AMB,'VISIBLE':GRN}
SIGMA=0.08

# static tracing
STAT={}
for c,cam in sim.CAMS.items():
    D,valid=sim.fisheye_dirs(cam); STAT[c]=(sim.trace_static(cam.pos,D),valid)

def rr(d,box,r,fill=None,outline=None,width=1): d.rounded_rectangle(box,r,fill=fill,outline=outline,width=width)
def dashed(d,p0,p1,col,w=2,dash=10,gap=7):
    L=math.hypot(p1[0]-p0[0],p1[1]-p0[1]);
    if L<1: return
    ux,uy=(p1[0]-p0[0])/L,(p1[1]-p0[1])/L; s=0
    while s<L:
        e=min(L,s+dash); d.line([(p0[0]+ux*s,p0[1]+uy*s),(p0[0]+ux*e,p0[1]+uy*e)],fill=col,width=w); s=e+gap
def tw(d,t,f): return d.textbbox((0,0),t,font=f)[2]
def to_disp(x,y,S): return ((x+sim.BR)/(2*sim.BR)*S,(y+sim.BR)/(2*sim.BR)*S)

def car_icon(d,ox,oy,S):
    x0,y0=to_disp(-sim.HX,-sim.HY,S); x1,y1=to_disp(sim.HX,sim.HY,S); x0+=ox;x1+=ox;y0+=oy;y1+=oy
    rr(d,(x0,y0,x1,y1),int(0.35*S/14.4),fill=(58,68,82),outline=(120,132,146),width=2)
    w=x1-x0;h=y1-y0
    d.polygon([(x0+w*.14,y0+h*.27),(x1-w*.14,y0+h*.27),(x1-w*.2,y0+h*.4),(x0+w*.2,y0+h*.4)],fill=(120,150,175))
    d.polygon([(x0+w*.2,y0+h*.76),(x1-w*.2,y0+h*.76),(x1-w*.16,y0+h*.84),(x0+w*.16,y0+h*.84)],fill=(120,150,175))
    d.rectangle((x0+w*.22,y0+h*.42,x1-w*.22,y0+h*.74),fill=(70,82,98))

def seam_lines(d,m,ox,oy,S,col,faint=False):
    for i,C in enumerate(sim.CORNERS):
        ph=math.radians(m['seams'][i]); p0=to_disp(C['cx'],C['cy'],S); p1=to_disp(C['cx']+C['sx']*math.sin(ph)*12,C['cy']+C['sy']*math.cos(ph)*12,S)
        # clip p1 to box
        t=1.0
        for k,lim in ((0,S),(1,S)):
            dv=p1[k]-p0[k]
            if dv>0 and p1[k]>lim: t=min(t,(lim-p0[k])/dv)
            if dv<0 and p1[k]<0: t=min(t,(0-p0[k])/dv)
        p1=(p0[0]+(p1[0]-p0[0])*t,p0[1]+(p1[1]-p0[1])*t)
        dashed(d,(p0[0]+ox,p0[1]+oy),(p1[0]+ox,p1[1]+oy),col,2 if not faint else 1)

def dist_to_car(o): return max(0.0,math.hypot(max(0,abs(o['x'])-sim.HX),max(0,abs(o['y'])-sim.HY))-o['r'])

def pinhole(owner,target,objs,Wp=700,Hp=394,hfov=100):
    cam=sim.CAMS[owner]; tx,ty=target
    dx,dy=tx-cam.pos[0],ty-cam.pos[1]; dist=math.hypot(dx,dy)
    pitch=math.degrees(math.atan2(cam.h-0.55,dist))+4
    c=sim.Cam(tuple(cam.pos),sim.pitched((dx,dy),pitch))
    f=(Wp/2)/math.tan(math.radians(hfov/2))
    u,v=np.meshgrid(np.arange(Wp)+0.5,np.arange(Hp)+0.5)
    dc=np.stack([(u-Wp/2)/f,(v-Hp/2)/f,np.ones_like(u)],-1); dc/=np.linalg.norm(dc,axis=-1,keepdims=True)
    D=(dc@c.R).astype(np.float32)
    st=sim.trace_static(c.pos,D); img=sim.trace_dynamic(st,objs,(Hp,Wp))
    def proj(P):
        q=(np.array(P)-c.pos)@c.R.T
        if q[2]<=0.05: return None
        return (Wp/2+f*q[0]/q[2],Hp/2+f*q[1]/q[2])
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),proj,c

class Runner:
    def __init__(s): s.rng=np.random.default_rng(5); s.totals={k:[0,0,0] for k,*_ in METH}
    def noise(s,objs,dt):
        tau=0.6
        for o in objs:
            o['nx']+=-o['nx']/tau*dt+SIGMA*math.sqrt(2*dt/tau)*s.rng.normal(); o['ny']+=-o['ny']/tau*dt+SIGMA*math.sqrt(2*dt/tau)*s.rng.normal()
    def est(s,objs): return [dict(o,x=o['x']+o['nx'],y=o['y']+o['ny']) for o in objs]

def frame_main(si,S,t,objs,ms,hist,views):
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    # title bar
    d.text((24,16),'Object-preserving merge for 360° surround view',font=f30,fill=TXT)
    lab=f'Scene {si+1} of {len(scen.SCENES)}:  {S["title"]}'; d.text((W-24-tw(d,lab,f20),22),lab,font=f20,fill=MUT)
    d.line([(16,66),(W-16,66)],fill=LINE,width=1)
    # camera images
    cams={}
    for c in sim.CAM_IDS:
        im=sim.trace_dynamic(STAT[c][0],objs,(sim.FE_N,sim.FE_N)); im[~STAT[c][1]]=0; cams[c]=im
    evals={}
    for wi,(k,title,sub,col) in enumerate(METH):
        m=ms[k]; bev,owned=sim.stitch(cams,m)
        S_=384; bev=cv2.resize(cv2.cvtColor(bev,cv2.COLOR_BGR2RGB),(S_,S_),interpolation=cv2.INTER_AREA)
        if k=='shape' and owned.any():
            mk=cv2.resize(owned.astype(np.uint8)*255,(S_,S_),interpolation=cv2.INTER_NEAREST)
            cnts,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bev,cnts,-1,GRN,2,cv2.LINE_AA)
        wx=16+wi*634; wy=76; ww=620
        rr(d,(wx,wy,wx+ww,wy+486),10,fill=PANEL)
        d.rectangle((wx,wy+10,wx+6,wy+40),fill=col)
        d.text((wx+18,wy+8),title,font=fb20,fill=TXT); d.text((wx+18,wy+34),sub,font=f14,fill=MUT)
        bx,by=wx+26,wy+60
        img.paste(Image.fromarray(bev),(bx,by)); car_icon(d,bx,by,S_)
        if k!='shape': seam_lines(d,m,bx,by,S_,(245,245,245))
        else: seam_lines(d,m,bx,by,S_,(200,205,210),faint=True)
        rs=[sim.eval_obj(m,o) for o in objs]; evals[k]=rs
        worst=min(r['best'] for r in rs); anylost=any(sim.status(r)=='LOST' for r in rs)
        for o,r in zip(objs,rs):
            stt=sim.status(r); sc=STCOL[stt]; px,py=to_disp(o['x'],o['y'],S_); px+=bx; py+=by
            R=19; d.ellipse((px-R,py-R,px+R,py+R),outline=sc,width=3)
            tag=f'{stt}  {int(round(r["best"]*100))}%'; tx=tw(d,tag,fb16)
            if o['x']<0: lx=px-R-tx-24
            else: lx=px+R+8
            lx=min(max(lx,bx+4),bx+S_-tx-20); ly=py-13
            rr(d,(lx,ly,lx+tx+16,ly+26),6,fill=sc); d.text((lx+8,ly+3),tag,font=fb16,fill=(15,15,15))
        # gauge
        gx=bx+S_+40; gy=by; gh=S_-24
        d.text((gx-6,gy-2),'Worst',font=f12,fill=MUT); d.text((gx-6,gy+12),'object',font=f12,fill=MUT)
        rr(d,(gx,gy+36,gx+36,gy+gh),5,fill=(38,45,54))
        fh=(gh-36)*worst; gc=GRN if worst>=0.9 else (AMB if worst>=0.55 else RED)
        if fh>2: rr(d,(gx,gy+gh-fh,gx+36,gy+gh),5,fill=gc)
        pct=f'{int(round(worst*100))}%'; d.text((gx+18-tw(d,pct,fb18)/2,gy+gh+6),pct,font=fb18,fill=TXT)
        # status line
        if anylost: msg,mc=('Person disappears at the merge line' if len(objs)==1 else 'A person disappears at the merge line'),RED
        elif any(sim.status(r)=='CUT' for r in rs): msg,mc='Person partly cropped by the merge line',AMB
        else: msg,mc='Every person fully visible',GRN
        d.ellipse((wx+18,wy+457,wx+30,wy+469),fill=mc); d.text((wx+38,wy+452),msg,font=fb18,fill=mc)
        if k=='shape': views['bev']=bev; views['owned']=owned
    # ---------- HMI ----------
    hx,hy,hw,hh=16,578,1220,486
    rr(d,(hx,hy,hx+hw,hy+hh),24,fill=(8,9,11),outline=(70,78,88),width=3)
    sx0,sy0=hx+20,hy+18; sw,sh=hw-40,hh-36
    rr(d,(sx0,sy0,sx0+sw,sy0+sh),10,fill=(12,15,19))
    d.text((sx0+16,sy0+8),'10:42',font=fb16,fill=TXT)
    ct='Park assist   360° view'; d.text((sx0+sw/2-tw(d,ct,f16)/2,sy0+9),ct,font=f16,fill=MUT)
    rr(d,(sx0+sw-44,sy0+6,sx0+sw-14,sy0+30),5,outline=TXT,width=2); d.text((sx0+sw-35,sy0+8),'R',font=fb16,fill=TXT)
    d.text((hx+hw-210,hy+hh-2+4),'',font=f12,fill=MUT)
    HB=384; hbx,hby=sx0+16,sy0+40
    bevh=cv2.resize(views['bev'],(HB,HB),interpolation=cv2.INTER_AREA)
    img.paste(Image.fromarray(bevh),(hbx,hby)); car_icon(d,hbx,hby,HB)
    est=[dict(o,x=o['x']+o['nx'],y=o['y']+o['ny']) for o in objs]
    near=sorted(est,key=dist_to_car)
    for o in est:
        dd=dist_to_car(o); cx,cy=to_disp(o['x'],o['y'],HB); cx+=hbx; cy+=hby; b=26; L=9
        cc=RED if dd<0.6 else (YEL if dd<1.5 else (140,210,255))
        for sxn,syn in ((-1,-1),(1,-1),(-1,1),(1,1)):
            ex,ey=cx+sxn*b,cy+syn*b; d.line([(ex,ey),(ex-sxn*L,ey)],fill=cc,width=3); d.line([(ex,ey),(ex,ey-syn*L)],fill=cc,width=3)
        lbl=f'{dd:.1f} m'; d.text((cx+b+4,cy-10),lbl,font=fb14 if False else fb16,fill=cc)
        # distance arc at nearest car corner
        ccx=-sim.HX if o['x']<0 else sim.HX; ccy=-sim.HY if o['y']<0 else sim.HY
        if abs(o['x'])<sim.HX: ccx=o['x']
        if abs(o['y'])<sim.HY: ccy=o['y']
        pcx,pcy=to_disp(ccx,ccy,HB); pcx+=hbx; pcy+=hby
        ang=math.degrees(math.atan2(o['y']-ccy,o['x']-ccx))
        for rad,colr,act in ((0.45,RED,dd<0.6),(0.85,YEL,0.6<=dd<1.5),(1.25,GRN,1.5<=dd<2.5)):
            rp=rad/(2*sim.BR)*HB
            d.arc((pcx-rp,pcy-rp,pcx+rp,pcy+rp),ang-28,ang+28,fill=colr if act else (70,76,84),width=5 if act else 3)
    rr(d,(hbx+6,hby+HB-28,hbx+150,hby+HB-6),5,fill=(0,0,0)); d.text((hbx+12,hby+HB-26),'Top view',font=f14,fill=TXT)
    # corner view
    tgt=near[0]; own=ms['shape']['owners'].get(tgt['id'],'F')
    views.setdefault('aim',(tgt['x'],tgt['y']))
    ax,ay=views['aim']; ax+=0.25*(tgt['x']-ax); ay+=0.25*(tgt['y']-ay); views['aim']=(ax,ay)
    cvw,cvh=740,HB; pim,proj,pc=pinhole(own,(ax,ay),objs,cvw,cvh,118)
    cxo,cyo=hbx+HB+16,hby
    img.paste(Image.fromarray(pim),(cxo,cyo))
    for o in est:
        vd=np.array([o['x'],o['y']])-pc.pos[:2]; vd/=np.linalg.norm(vd); lat=np.array([-vd[1],vd[0]])*o['r']*1.25
        pts=[proj((o['x']+sg*lat[0],o['y']+sg*lat[1],z)) for sg in (-1,1) for z in (0,o['h']+0.05)]
        if None in pts: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
        if x1<0 or x0>cvw: continue
        x0=max(2,x0);x1=min(cvw-2,x1);y0=max(2,y0);y1=min(cvh-2,y1)
        dd=dist_to_car(o); cc=RED if dd<0.6 else (YEL if dd<1.5 else (140,210,255))
        d.rectangle((cxo+x0,cyo+y0,cxo+x1,cyo+y1),outline=cc,width=3)
        lb=f'{o["kind"]}  {dd:.1f} m'; lw_=tw(d,lb,fb16)
        ly=cyo+y0-26 if y0>28 else cyo+y1+2
        rr(d,(cxo+x0,ly,cxo+x0+lw_+12,ly+24),4,fill=cc); d.text((cxo+x0+6,ly+2),lb,font=fb16,fill=(15,15,15))
    camname={'F':'front','B':'rear','L':'left','R':'right'}[own]
    chip=f'Corner view (auto): {camname} camera'
    rr(d,(cxo+8,cyo+8,cxo+16+tw(d,chip,f14),cyo+32),5,fill=(0,0,0)); d.text((cxo+12,cyo+10),chip,font=f14,fill=TXT)
    dd=dist_to_car(tgt); side=('front' if tgt['y']<-sim.HY else 'rear' if tgt['y']>sim.HY else '')+('-left' if tgt['x']<0 else '-right')
    side=side.strip('-')
    if dd<0.6: bt,bc=f'Stop: {tgt["kind"].lower()} very close at {side} corner ({dd:.1f} m)',RED
    elif dd<1.5: bt,bc=f'Caution: {tgt["kind"].lower()} at {side} corner ({dd:.1f} m)',YEL
    else: bt,bc=f'{tgt["kind"]} detected {side} ({dd:.1f} m)',(140,210,255)
    by0=cyo+cvh-44; ov=Image.new('RGBA',(cvw,44),(0,0,0,170)); img.paste(ov,(cxo,by0),ov)
    d.polygon([(cxo+18,by0+34),(cxo+34,by0+8),(cxo+50,by0+34)],fill=bc); d.text((cxo+31,by0+13),'!',font=fb18,fill=(0,0,0))
    d.text((cxo+62,by0+11),bt,font=fb18,fill=bc)
    d.text((sx0+16,sy0+sh-24),'In-car screen with the proposed merge: the person stays whole in the top view, and the corner view opens on them.',font=f14,fill=MUT)
    # ---------- right panel ----------
    px0,py0,pw=1252,578,652
    rr(d,(px0,py0,px0+pw,py0+486),10,fill=PANEL)
    d.text((px0+20,py0+14),'What is happening',font=fb22,fill=TXT)
    yy=py0+50
    for ln in S['text']: d.text((px0+20,yy),ln,font=f18,fill=(205,212,218)); yy+=26
    cx0,cy0,cw,chh=px0+62,py0+238,pw-90,120
    d.text((px0+20,cy0-30),'Worst-object visibility over time',font=fb16,fill=TXT)
    for v in (0,0.5,1):
        yv=cy0+(1-v)*chh; d.line([(cx0,yv),(cx0+cw,yv)],fill=LINE,width=1); lb=f'{int(v*100)}%'; d.text((cx0-8-tw(d,lb,f14),yv-9),lb,font=f14,fill=MUT)
    for k,_,_,col in METH:
        pts=[(cx0+tt/S['dur']*cw,cy0+(1-vv)*chh) for tt,vv in hist[k]]
        if len(pts)>1: d.line(pts,fill=col,width=4 if k=='shape' else 3)
    ly=cy0+chh+14; lx=px0+20
    for k,title,_,col in METH:
        nm=title.split(':')[0]; d.line([(lx,ly+10),(lx+26,ly+10)],fill=col,width=4); d.text((lx+32,ly),nm,font=f16,fill=MUT); lx+=150+tw(d,nm,f16)*0.3
    rr(d,(px0+20,py0+410,px0+pw-20,py0+470),8,fill=(22,52,40))
    d.text((px0+34,py0+418),'What the driver gains',font=fb16,fill=GRN)
    d.text((px0+34,py0+442),'People never vanish from the top view near the bumper.',font=f16,fill=(215,240,225))
    d.text((1252,H-14),'',font=f12,fill=MUT)
    return img,evals

def card_intro():
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    d.text((120,150),'Keeping people visible where',font=f56,fill=TXT)
    d.text((120,222),'two surround-view cameras meet',font=f56,fill=TXT)
    d.text((122,310),'Simulation of a 360° top view built from four fisheye cameras, with three ways of merging them.',font=f22,fill=MUT)
    y=410
    for k,title,sub,col in METH:
        d.rectangle((120,y,128,y+56),fill=col); d.text((148,y),title,font=fb26,fill=TXT); d.text((148,y+32),sub,font=f20,fill=MUT); y+=92
    d.rectangle((120,y,128,y+56),fill=YEL); d.text((148,y),'Bottom: the in-car screen',font=fb26,fill=TXT); d.text((148,y+32),'How the proposed view looks and helps the driver',font=f20,fill=MUT)
    d.text((120,H-70),'Custom Python ray-cast simulator with flat-ground projection and 8 cm detection error. Not road data.',font=f18,fill=MUT)
    return img
def card_outro(tot,lost):
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    d.text((120,120),'Result across all three scenes',font=f56,fill=TXT)
    d.text((122,200),'Average visibility of the worst-covered person, per frame',font=f22,fill=MUT)
    y=290; bw=1100
    for k,title,_,col in METH:
        v=tot[k]; d.text((120,y),title,font=fb22,fill=TXT)
        rr(d,(120,y+36,120+bw,y+76),8,fill=(38,45,54)); rr(d,(120,y+36,120+max(12,bw*v),y+76),8,fill=col)
        d.text((120+bw+24,y+38),f'{v*100:.0f}%',font=f40,fill=TXT)
        d.text((120,y+84),f'A person was lost in {lost[k]*100:.0f}% of frames',font=f18,fill=MUT); y+=170
    d.text((120,H-110),'Straight merge lines, even turning ones, cannot keep people whole near the car corner.',font=f22,fill=TXT)
    d.text((120,H-76),'A merge region shaped around each detected object can.',font=f22,fill=TXT)
    return img

def main(out):
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
        '-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',out],stdin=subprocess.PIPE)
    def emit(im,n=1):
        b=np.asarray(im,dtype=np.uint8).tobytes()
        for _ in range(n): p.stdin.write(b)
    intro=card_intro()
    for i in range(int(FPS*4)):
        a=min(1,i/8); emit(Image.blend(Image.new('RGB',(W,H),BG),intro,a))
    run=Runner(); tot={k:0.0 for k,*_ in METH}; lost={k:0 for k,*_ in METH}; nfr=0
    dt=1/FPS
    for si,S in enumerate(scen.SCENES):
        objs=S['make'](); ms={k:sim.make_method(k,6,SIGMA) for k,*_ in METH}; hist={k:[] for k,*_ in METH}; views={}; t=0.0; n=int(S['dur']*FPS)
        for fi in range(n):
            S['step'](objs,t); run.noise(objs,dt); est=run.est(objs)
            for m in ms.values(): sim.update_method(m,est,dt)
            img,ev=frame_main(si,S,t,objs,ms,hist,views)
            for k in ms:
                w=min(r['best'] for r in ev[k]); hist[k].append((t,w)); tot[k]+=w; lost[k]+=any(sim.status(r)=='LOST' for r in ev[k])
            nfr+=1
            emit(img); t+=dt
            if fi%30==0: print(si,fi,flush=True)
        emit(img,int(FPS*0.8))
    tot={k:v/nfr for k,v in tot.items()}; lost={k:v/nfr for k,v in lost.items()}
    print(tot,lost)
    oc=card_outro(tot,lost)
    for i in range(int(FPS*6)): emit(Image.blend(Image.new('RGB',(W,H),BG),oc,min(1,i/8)))
    p.stdin.close(); p.wait()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='still':
        S=scen.SCENES[int(sys.argv[2])]; objs=S['make'](); ms={k:sim.make_method(k,6,SIGMA) for k,*_ in METH}; hist={k:[] for k,*_ in METH}; views={}
        tt=float(sys.argv[3]); t=0
        while t<tt:
            S['step'](objs,t)
            for m in ms.values(): sim.update_method(m,objs,1/FPS)
            for k in ms: hist[k].append((t,min(sim.eval_obj(ms[k],o)['best'] for o in objs)))
            t+=1/FPS
        img,_=frame_main(int(sys.argv[2]),S,t,objs,ms,hist,views); img.save('still.png')
    else: main(sys.argv[1])
