import sim2 as sm, scen2, numpy as np, cv2, math, subprocess, sys, json
from PIL import Image, ImageDraw
from render import (F,f12,f14,f16,f18,f20,f22,f26,f30,f40,f56,fb16,fb18,fb20,fb22,fb26,BG,PANEL,LINE,TXT,MUT,RED,AMB,GRN,YEL,
                    rr,dashed,tw,to_disp,car_icon,seam_lines,dist_to_car)
W,H,FPS=1920,1080,15
CYAN=(80,200,255)
METH=[('fixed','Problem: fixed merge line','Used in most systems today',RED),
      ('dynamic','Other fix: turning straight line','Other fix: turns the straight line',AMB),
      ('prop','Proposed: UA-OSM','Proposed: bends the merge around each person',GRN)]
LADDER=['Default merge line','Turned straight line','Line bent around object','Multi-object camera assignment','Honest fallback: tell the driver']
SIGMA=0.08; LAT=0.10
STAT={}
for c,cam in sm.CAMS.items():
    D,valid=sm.fisheye_dirs(cam); STAT[c]=(sm.trace_static(cam.pos,D),valid)

def pinhole(owner,target,objs,Wp,Hp,hfov):
    cam=sm.CAMS[owner]; tx,ty=target; dx,dy=tx-cam.pos[0],ty-cam.pos[1]; dist=math.hypot(dx,dy)
    pitch=math.degrees(math.atan2(cam.h-0.55,dist))+4
    c=sm.Cam(tuple(cam.pos),sm.pitched((dx,dy),pitch)); f=(Wp/2)/math.tan(math.radians(hfov/2))
    u,v=np.meshgrid(np.arange(Wp)+0.5,np.arange(Hp)+0.5)
    dc=np.stack([(u-Wp/2)/f,(v-Hp/2)/f,np.ones_like(u)],-1); dc/=np.linalg.norm(dc,axis=-1,keepdims=True)
    st=sm.trace_static(c.pos,(dc@c.R).astype(np.float32)); img=sm.trace_dynamic(st,objs,(Hp,Wp))
    def proj(P):
        q=(np.array(P)-c.pos)@c.R.T
        return None if q[2]<=0.05 else (Wp/2+f*q[0]/q[2],Hp/2+f*q[1]/q[2])
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),proj,c

class Tracker:
    """detections arrive LAT late with noise; velocity-based prediction compensates the delay"""
    def __init__(s,seed): s.rng=np.random.default_rng(seed); s.hist={}; s.n={}; s.prev={}
    def step(s,objs,t,dt,lost):
        est=[]
        for o in objs:
            h=s.hist.setdefault(o['id'],[]); h.append((t,o['x'],o['y']))
            nx,ny=s.n.get(o['id'],(0.0,0.0)); tau=0.6
            nx+=-nx/tau*dt+SIGMA*math.sqrt(2*dt/tau)*s.rng.normal(); ny+=-ny/tau*dt+SIGMA*math.sqrt(2*dt/tau)*s.rng.normal(); s.n[o['id']]=(nx,ny)
            if not o['det']: continue
            L=lost.get(o['id']); coasting=L is not None and L[0]<=t<L[1]
            td=t-LAT; past=[p for p in h if p[0]<=td] or [h[0]]; _,px,py=past[-1]
            past2=[p for p in h if p[0]<=td-0.2] or [h[0]]; _,qx,qy=past2[-1]
            vx,vy=(px-qx)/0.2,(py-qy)/0.2
            if coasting:
                pv=s.prev[o['id']]; e=dict(o,x=pv['x']+pv['vx']*dt,y=pv['y']+pv['vy']*dt,coasting=True,sig_extra=min(0.5,pv.get('sig_extra',0)+0.02))
                e['vx'],e['vy']=pv['vx'],pv['vy']
            else:
                e=dict(o,x=px+nx+vx*LAT,y=py+ny+vy*LAT,coasting=False,sig_extra=0.0); e['vx'],e['vy']=vx,vy
            s.prev[o['id']]=e; est.append(e)
        return est

def draw_tag(d,px,py,left,text,col,bx,S_,taken=None):
    tx=tw(d,text,fb16); lx=px-19-tx-24 if left else px+27
    lx=min(max(lx,bx+4),bx+S_-tx-20); ly=py-13
    if taken is not None:
        for _ in range(12):
            if not any(not(lx+tx+16<a or lx>c or ly+26<b or ly>e) for a,b,c,e in taken): break
            ly+=30
        taken.append((lx,ly,lx+tx+16,ly+26))
    rr(d,(lx,ly,lx+tx+16,ly+26),6,fill=col); d.text((lx+8,ly+3),text,font=fb16,fill=(15,15,15))

def frame(si,S,t,objs,est,ms,hist,views):
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    d.text((24,16),'Keeping people visible in the 360° parking view',font=f30,fill=TXT)
    lab=f'Scene {si+1} of {len(scen2.SCENES)}:  {S["title"]}'; d.text((W-24-tw(d,lab,f20),22),lab,font=f20,fill=MUT)
    d.line([(16,66),(W-16,66)],fill=LINE,width=1)
    cams={}
    for c in sm.CAM_IDS:
        im=sm.trace_dynamic(STAT[c][0],objs,(sm.FE_N,sm.FE_N)); im[~STAT[c][1]]=0; cams[c]=im
    warps=sm.warp_all(cams)
    grid,nb=sm.disagreement_blobs(warps,est,SIGMA); ms['prop']['blobgrid']=grid if nb else None; ms['prop']['nblobs']=nb
    for k,m in ms.items():
        if k=='prop': sm.update_prop(m,est,1/FPS)
        else: sm.update_method(m,est,1/FPS)
    evals={}
    for wi,(k,title,sub,col) in enumerate(METH):
        m=ms[k]; bev,owned=sm.stitch(warps,m); S_=384
        bev=cv2.resize(cv2.cvtColor(bev,cv2.COLOR_BGR2RGB),(S_,S_),interpolation=cv2.INTER_AREA)
        if k=='prop':
            if m['corridors']:
                cm=np.zeros(sm.BX.shape,bool)
                for c in m['corridors']: cm|=sm.cor_mask(c)
                mk=cv2.resize(cm.astype(np.uint8)*255,(S_,S_),interpolation=cv2.INTER_NEAREST)
                cn,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(bev,cn,-1,GRN,2,cv2.LINE_AA)
            if m.get('blobgrid') is not None:
                mk=cv2.resize((m['blobgrid']>=0).astype(np.uint8)*255,(S_,S_),interpolation=cv2.INTER_NEAREST)
                cn,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(bev,cn,-1,CYAN,2,cv2.LINE_AA)
            views['bev']=bev
        wx=16+wi*634; wy=76; ww=620
        rr(d,(wx,wy,wx+ww,wy+486),10,fill=PANEL); d.rectangle((wx,wy+10,wx+6,wy+40),fill=col)
        d.text((wx+18,wy+8),title,font=fb20,fill=TXT); d.text((wx+18,wy+34),sub,font=f14,fill=MUT)
        bx,by=wx+26,wy+60
        img.paste(Image.fromarray(bev),(bx,by)); car_icon(d,bx,by,S_)
        if k!='prop' or m['level']<=2: seam_lines(d,m,bx,by,S_,(245,245,245))
        else: seam_lines(d,m,bx,by,S_,(200,205,210),faint=True)
        rs=[sm.eval_obj(m,o) for o in objs]; evals[k]=rs
        worst=min(r['best'] for r in rs); taken=[]
        for o,r in zip(objs,rs):
            stt=sm.status(r); sc={'LOST':RED,'CUT':AMB,'DOUBLED':AMB,'VISIBLE':GRN}[stt]
            px,py=to_disp(o['x'],o['y'],S_); px+=bx; py+=by; R=15 if len(objs)>2 else 19
            d.ellipse((px-R,py-R,px+R,py+R),outline=sc,width=3)
            if stt!='VISIBLE' or len(objs)<=2: draw_tag(d,px,py,o['x']<0,f'{stt} {int(round(r["best"]*100))}%',sc,bx,S_,taken)
        if k=='prop':
            lv=m['level']; lc=[MUT,GRN,GRN,GRN,RED][lv-1]; tag=f'Ladder level {lv}'
            rr(d,(bx+S_-tw(d,tag,fb16)-22,by+8,bx+S_-8,by+34),6,fill=(0,0,0)); d.text((bx+S_-tw(d,tag,fb16)-14,by+11),tag,font=fb16,fill=lc)
            if m['nblobs']: 
                t2='Unknown object kept visible'; rr(d,(bx+8,by+S_-34,bx+22+tw(d,t2,fb16),by+S_-8),6,fill=(0,0,0)); d.text((bx+15,by+S_-31),t2,font=fb16,fill=CYAN)
        gx=bx+S_+40; gy=by; gh=S_-24
        d.text((gx-6,gy-2),'Worst',font=f12,fill=MUT); d.text((gx-6,gy+12),'object',font=f12,fill=MUT)
        rr(d,(gx,gy+36,gx+36,gy+gh),5,fill=(38,45,54)); fh=(gh-36)*worst; gc=GRN if worst>=0.9 else (AMB if worst>=0.55 else RED)
        if fh>2: rr(d,(gx,gy+gh-fh,gx+36,gy+gh),5,fill=gc)
        pct=f'{int(round(worst*100))}%'; d.text((gx+18-tw(d,pct,fb18)/2,gy+gh+6),pct,font=fb18,fill=TXT)
        nl=sum(sm.status(r)=='LOST' for r in rs); nc=sum(sm.status(r)=='CUT' for r in rs)
        if nl: msg,mc=(f'{nl} object{"s disappear" if nl>1 else " disappears"} at the merge line'),RED
        elif nc: msg,mc='An object is partly cropped',AMB
        else: msg,mc='Every object fully visible',GRN
        d.ellipse((wx+18,wy+457,wx+30,wy+469),fill=mc); d.text((wx+38,wy+452),msg,font=fb18,fill=mc)
    # ---------- HMI ----------
    m=ms['prop']; hx,hy,hw,hh=16,578,1220,486
    rr(d,(hx,hy,hx+hw,hy+hh),24,fill=(8,9,11),outline=(70,78,88),width=3)
    sx0,sy0=hx+20,hy+18; sw,sh=hw-40,hh-36; rr(d,(sx0,sy0,sx0+sw,sy0+sh),10,fill=(12,15,19))
    d.text((sx0+16,sy0+8),'10:42',font=fb16,fill=TXT); ct='Park assist   360° view'; d.text((sx0+sw/2-tw(d,ct,f16)/2,sy0+9),ct,font=f16,fill=MUT)
    rr(d,(sx0+sw-44,sy0+6,sx0+sw-14,sy0+30),5,outline=TXT,width=2); d.text((sx0+sw-35,sy0+8),'R',font=fb16,fill=TXT)
    HB=384; hbx,hby=sx0+16,sy0+40
    img.paste(Image.fromarray(cv2.resize(views['bev'],(HB,HB),interpolation=cv2.INTER_AREA)),(hbx,hby)); car_icon(d,hbx,hby,HB)
    for o in est:
        dd=dist_to_car(o); cx,cy=to_disp(o['x'],o['y'],HB); cx+=hbx; cy+=hby; b=22; L=8
        cc=RED if dd<0.6 else (YEL if dd<1.5 else (140,210,255))
        if o.get('coasting'):
            for a in range(0,360,40): d.arc((cx-24,cy-24,cx+24,cy+24),a,a+22,fill=RED,width=3)
            d.text((cx+28,cy-10),'not tracked',font=fb16,fill=RED); continue
        for sxn,syn in ((-1,-1),(1,-1),(-1,1),(1,1)):
            ex,ey=cx+sxn*b,cy+syn*b; d.line([(ex,ey),(ex-sxn*L,ey)],fill=cc,width=3); d.line([(ex,ey),(ex,ey-syn*L)],fill=cc,width=3)
        if len(est)<=2: d.text((cx+b+4,cy-10),f'{dd:.1f} m',font=fb16,fill=cc)
    if m.get('blobgrid') is not None:
        ys,xs=np.nonzero(m['blobgrid']>=0)
        if len(xs):
            px=sm.BX[ys,xs]; py=sm.BY[ys,xs]; dc=np.hypot(np.maximum(0,np.abs(px)-sm.HX),np.maximum(0,np.abs(py)-sm.HY)); i=np.argmin(dc)
            ux,uy=to_disp(px[i],py[i],HB); ux+=hbx; uy+=hby
            dashed(d,(ux-20,uy-20),(ux+20,uy-20),CYAN,3,6,5); dashed(d,(ux+20,uy-20),(ux+20,uy+20),CYAN,3,6,5); dashed(d,(ux+20,uy+20),(ux-20,uy+20),CYAN,3,6,5); dashed(d,(ux-20,uy+20),(ux-20,uy-20),CYAN,3,6,5)
            d.text((ux+24,uy-10),'unknown object',font=fb16,fill=CYAN)
            views['unk']=(px[i],py[i])
    else: views.pop('unk',None)
    rr(d,(hbx+6,hby+HB-28,hbx+110,hby+HB-6),5,fill=(0,0,0)); d.text((hbx+12,hby+HB-26),'Top view',font=f14,fill=TXT)
    # corner view target: worst/coasting object at level 5, else nearest object (incl. unknown)
    cands=[(dist_to_car(o),o['x'],o['y'],o) for o in est]
    if 'unk' in views: cands.append((math.hypot(max(0,abs(views['unk'][0])-sm.HX),max(0,abs(views['unk'][1])-sm.HY)),views['unk'][0],views['unk'][1],None))
    focus=None
    if m['level']==5 and m['worst_id']: focus=next((o for o in est if o['id']==m['worst_id']),None)
    if focus is not None: tx,ty,fo=focus['x'],focus['y'],focus
    elif cands: _,tx,ty,fo=min(cands,key=lambda c:c[0])
    else: tx,ty,fo=-2,-4,None
    own=(m['owners'].get(fo['id']) if fo else None) or ('F' if ty<-sm.HY else 'B' if ty>sm.HY else ('L' if tx<0 else 'R'))
    if not sm.sees(own,tx,ty): own='F' if ty<-sm.HY else ('L' if tx<0 else 'R')
    views.setdefault('aim',(tx,ty)); ax,ay=views['aim']; ax+=0.25*(tx-ax); ay+=0.25*(ty-ay); views['aim']=(ax,ay)
    cvw,cvh=740,HB; pim,proj,pc=pinhole(own,(ax,ay),objs,cvw,cvh,118); cxo,cyo=hbx+HB+16,hby
    img.paste(Image.fromarray(pim),(cxo,cyo))
    for o in est:
        if o.get('coasting'): continue
        vd=np.array([o['x'],o['y']])-pc.pos[:2]; vd/=np.linalg.norm(vd); lat=np.array([-vd[1],vd[0]])*o['r']*1.25
        pts=[proj((o['x']+sg*lat[0],o['y']+sg*lat[1],z)) for sg in (-1,1) for z in (0,o['h']+0.05)]
        if None in pts: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; x0,x1,y0,y1=max(2,min(xs)),min(cvw-2,max(xs)),max(2,min(ys)),min(cvh-2,max(ys))
        if x1<=x0: continue
        dd=dist_to_car(o); cc=RED if dd<0.6 else (YEL if dd<1.5 else (140,210,255))
        d.rectangle((cxo+x0,cyo+y0,cxo+x1,cyo+y1),outline=cc,width=3)
        lb=f'{o["kind"]} {dd:.1f} m'; lw_=tw(d,lb,fb16); ly=cyo+y0-26 if y0>28 else cyo+y1+2
        rr(d,(cxo+x0,ly,cxo+x0+lw_+12,ly+24),4,fill=cc); d.text((cxo+x0+6,ly+2),lb,font=fb16,fill=(15,15,15))
    camname={'F':'front','B':'rear','L':'left','R':'right'}[own]; chip=f'Corner view (auto): {camname} camera'
    rr(d,(cxo+8,cyo+8,cxo+16+tw(d,chip,f14),cyo+32),5,fill=(0,0,0)); d.text((cxo+12,cyo+10),chip,font=f14,fill=TXT)
    if m['level']==5 and fo is not None:
        bt,bc=(f'{fo["kind"]} not tracked right now. Check the corner view before moving.' if fo.get('coasting') else f'{fo["kind"]} cannot be shown whole in top view. Check corner view.'),RED
    elif fo is None and 'unk' in views: bt,bc='Unknown object near the front-left corner',CYAN
    elif fo is not None:
        dd=dist_to_car(fo)
        bt,bc=(f'Stop: {fo["kind"].lower()} very close ({dd:.1f} m)',RED) if dd<0.6 else ((f'Caution: {fo["kind"].lower()} near the corner ({dd:.1f} m)',YEL) if dd<1.5 else (f'{fo["kind"]} detected ({dd:.1f} m)',(140,210,255)))
    else: bt,bc='No objects near the vehicle',GRN
    by0=cyo+cvh-44; ov=Image.new('RGBA',(cvw,44),(0,0,0,175)); img.paste(ov,(cxo,by0),ov)
    d.polygon([(cxo+18,by0+34),(cxo+34,by0+8),(cxo+50,by0+34)],fill=bc); d.text((cxo+31,by0+13),'!',font=fb18,fill=(0,0,0))
    d.text((cxo+62,by0+11),bt,font=fb18,fill=bc)
    d.text((sx0+16,sy0+sh-24),'In-car screen with UA-OSM: people stay whole, unknown objects are never hidden, and the screen admits uncertainty.',font=f14,fill=MUT)
    # ---------- right panel ----------
    px0,py0,pw=1252,578,652; rr(d,(px0,py0,px0+pw,py0+486),10,fill=PANEL)
    d.text((px0+20,py0+12),'What is happening',font=fb22,fill=TXT); yy=py0+44
    for ln in S['text']: d.text((px0+20,yy),ln,font=f16,fill=(205,212,218)); yy+=22
    ly0=py0+178; d.text((px0+20,ly0),'Escalation ladder (proposed)',font=fb16,fill=TXT)
    for i,nm in enumerate(LADDER):
        yv=ly0+28+i*30; act=(i+1)==m['level']; col=(RED if i==4 else GRN) if act else (60,68,78)
        rr(d,(px0+20,yv,px0+48,yv+24),5,fill=col); d.text((px0+29,yv+2),str(i+1),font=fb16,fill=(15,15,15) if act else MUT)
        d.text((px0+60,yv+2),nm,font=fb16 if act else f16,fill=TXT if act else MUT)
    cx0,cy0,cw,chh=px0+62,py0+380,pw-90,72
    d.text((px0+330,ly0),'Worst-object visibility',font=fb16,fill=TXT)
    for v in (0,1):
        yv=cy0+(1-v)*chh; d.line([(cx0,yv),(cx0+cw,yv)],fill=LINE,width=1); lb=f'{int(v*100)}%'; d.text((cx0-8-tw(d,lb,f14),yv-9),lb,font=f14,fill=MUT)
    for k,_,_,col in METH:
        pts=[(cx0+tt/S['dur']*cw,cy0+(1-vv)*chh) for tt,vv in hist[k]]
        if len(pts)>1: d.line(pts,fill=col,width=4 if k=='prop' else 3)
    lx=px0+62
    for k,title,_,col in METH:
        nm=title.split(':')[0]; d.line([(lx,cy0+chh+18),(lx+22,cy0+chh+18)],fill=col,width=4); d.text((lx+28,cy0+chh+8),nm,font=f14,fill=MUT); lx+=150
    return img,evals

def card(lines):
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    for (x,y,txt,font,col) in lines: d.text((x,y),txt,font=font,fill=col)
    return img
def intro():
    L=[(120,120,'UA-OSM v2: keeping people visible',f56,TXT),(120,192,'where surround-view cameras meet',f56,TXT),
       (122,280,'Now with crowds, undetected objects, detection delay and an honest fallback on the in-car screen.',f22,MUT)]
    y=370
    for k,t,s,c in METH: L+=[(148,y,t,fb26,TXT),(148,y+32,s,f20,MUT)]; y+=86
    L+=[(148,y,'Five-step escalation ladder',fb26,TXT),(148,y+32,'Use the calmest method that works; tell the driver when nothing does',f20,MUT)]
    L+=[(120,H-70,'Custom Python ray-cast simulator, flat-ground top view, 8 cm detection error, 100 ms detection delay. Not road data.',f18,MUT)]
    img=card(L); d=ImageDraw.Draw(img); y=370
    for k,t,s,c in METH: d.rectangle((120,y,128,y+56),fill=c); y+=86
    d.rectangle((120,y,128,y+56),fill=YEL); return img
def outro(res):
    img=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(img)
    d.text((120,90),'Results across all five scenes',font=f56,fill=TXT)
    d.text((122,168),'Worst-covered object per frame, including the undetected trolley',font=f22,fill=MUT)
    y=240; bw=1000
    for k,title,_,col in METH:
        v=res['tot'][k]; d.text((120,y),title,font=fb22,fill=TXT)
        rr(d,(120,y+34,120+bw,y+72),8,fill=(38,45,54)); rr(d,(120,y+34,120+max(12,bw*v),y+72),8,fill=col)
        d.text((120+bw+24,y+34),f'{v*100:.0f}%',font=f40,fill=TXT)
        d.text((120,y+80),f'An object was lost in {res["lost"][k]*100:.0f}% of frames',font=f18,fill=MUT); y+=150
    lv=res['levels']; tot=sum(lv.values()) or 1
    d.text((120,y+10),'Proposed ladder usage:  '+'   '.join(f'L{i}: {lv.get(str(i),0)*100/tot:.0f}%' for i in range(1,6)),font=f22,fill=TXT)
    d.text((120,H-120),'In this model, camera assignment resolved every crowd layout tested. Level 5 was triggered by',font=f22,fill=TXT)
    d.text((120,H-86),'perception (a lost track), not geometry. Real-world validation is the next step.',font=f22,fill=TXT)
    return img
