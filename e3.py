# E3: unknown-object rule (rendered images) and false alarms under extrinsic calibration error
import sim2 as sm, numpy as np, math, json, cv2, sys
STAT={}
for c,cam in sm.CAMS.items():
    D,valid=sm.fisheye_dirs(cam); STAT[c]=(sm.trace_static(cam.pos,D),valid)
def render(objs,stat=STAT):
    cams={}
    for c in sm.CAM_IDS:
        im=sm.trace_dynamic(stat[c][0],objs,(sm.FE_N,sm.FE_N)); im[~stat[c][1]]=0; cams[c]=im
    return sm.warp_all(cams)
mode=sys.argv[1] if len(sys.argv)>1 and __name__=='__main__' else 'none'; rng=np.random.default_rng(11); out={}
if mode=='unknown':
    rows=[]
    for i in range(40):
        while True:
            x=rng.uniform(-3.5,-1.1); y=rng.uniform(-4.5,-2.4)
            if math.hypot(x+0.95,y+2.3)>0.35: break
        kind=['Trolley','Pedestrian','Child'][i%3]
        h,r={'Trolley':(1.0,0.3),'Pedestrian':(1.75,0.25),'Child':(1.1,0.2)}[kind]
        from scen import PED_B, CHILD_B
        o=dict(id='u',kind=kind,x=x,y=y,h=h,r=r,bands=PED_B if kind=='Pedestrian' else CHILD_B)
        w=render([o]); res={}
        for name in ('fixed','dynamic','uaosm_norule','uaosm'):
            m=sm.make_method('fixed',6,0.08) if name=='fixed' else (sm.make_method('dynamic',6,0.08) if name=='dynamic' else sm.make_prop(6,0.08))
            if name=='uaosm':
                g,n=sm.disagreement_blobs(w,[],0.08); m['blobgrid']=g if n else None
            for _ in range(20):   # undetected object: detections list is empty
                (sm.update_prop if m['kind']=='prop' else sm.update_method)(m,[],0.1)
            res[name]=sm.eval_obj(m,o)['best']
        rows.append(dict(kind=kind,x=x,y=y,**res))
    out['rows']=rows
    for name in ('fixed','dynamic','uaosm_norule','uaosm'):
        v=np.array([r[name] for r in rows]); out[name]=dict(mean=float(v.mean()),lost=float((v<0.55).mean()))
    print({k:v for k,v in out.items() if k!='rows'})
    json.dump(out,open('e3_unknown.json','w'),indent=1)
elif mode=='calib':
    lv=[0.0,0.25,0.5,1.0,2.0]
    for err in lv:
        fa=[]; area=[]
        for trial in range(8 if err>0 else 1):
            stat=dict(STAT)
            for c in sm.CAM_IDS:
                cam=sm.CAMS[c]
                if err==0: continue
                yaw=math.radians(rng.normal(0,err)); pit=math.radians(rng.normal(0,err))
                f=cam.R[2].copy()
                Rz=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
                f=Rz@f; ax=np.cross(f,[0,0,1]); ax/=np.linalg.norm(ax)
                K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]]); Rp=np.eye(3)+math.sin(pit)*K+(1-math.cos(pit))*K@K
                pc=sm.Cam(tuple(cam.pos),Rp@f)
                D,valid=sm.fisheye_dirs(pc); stat[c]=(sm.trace_static(pc.pos,D),valid)
            w=render([],stat); THR=int(sys.argv[2]) if len(sys.argv)>2 else 60; g,n=sm.disagreement_blobs(w,[],0.08,thr_seed=THR,thr_grow=max(15,THR//2))
            fa.append(n); area.append(float((g>=0).sum())*(2*sm.BR/sm.BN)**2)
        out[str(err)]=dict(false_blob_rate=float(np.mean(np.array(fa)>0)),mean_blobs=float(np.mean(fa)),mean_area_m2=float(np.mean(area)))
        print(err,out[str(err)],flush=True)
    json.dump(out,open(f'e3_calib_{sys.argv[2] if len(sys.argv)>2 else 60}.json','w'),indent=1)
