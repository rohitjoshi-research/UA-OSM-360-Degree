import numpy as np, subprocess, json
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1920,1080,10
FD='/usr/share/fonts/truetype/dejavu/'
F=lambda s,b=False:ImageFont.truetype(FD+('DejaVuSans-Bold.ttf' if b else 'DejaVuSans.ttf'),s)
BG=(16,20,25);TXT=(236,240,243);MUT=(150,160,170);RED=(232,76,61);AMB=(240,168,48);GRN=(46,196,120);BLU=(90,160,230);PANEL=(26,31,38)
def canvas(): im=Image.new('RGB',(W,H),BG); return im,ImageDraw.Draw(im)
def write(im,name,sec):
    p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','21','-pix_fmt','yuv420p',name],stdin=subprocess.PIPE)
    bg=Image.new('RGB',(W,H),BG)
    for i in range(int(sec*FPS)): p.stdin.write(np.asarray(Image.blend(bg,im,min(1,(i+1)/5))).tobytes())
    p.stdin.close(); p.wait()
# 1 intro
im,d=canvas()
d.text((120,110),'Keeping people visible in the 360° parking view',font=F(54,True),fill=TXT)
d.text((122,190),'UA-OSM demo: 12 times of day and weather conditions, 1 to 5 people',font=F(28),fill=MUT)
y=300
for n,(h,t) in enumerate([('The problem','Four cameras are joined into one top view. Near the car corners, a merge line decides which camera\'s picture is shown.'),
                          ('What goes wrong','A standing person looks stretched, differently in each camera. The merge line can cut away both copies,\nso a child can vanish from the screen.'),
                          ('Our idea (UA-OSM)','Give each person\'s whole shape to one camera that really sees them, and warn the driver when that is not possible.')]):
    d.rounded_rectangle((120,y,1800,y+170),16,fill=PANEL); d.text((150,y+22),h,font=F(30,True),fill=[RED,AMB,GRN][n]); d.multiline_text((150,y+70),t,font=F(24),fill=TXT,spacing=8); y+=200
d.text((120,H-70),'Simulation with a ray-traced 4-camera car model. Condition effects are modelled through detection quality. Not road data.',font=F(20),fill=MUT)
write(im,'c_intro.mp4',8)
# 2 how to read
im,d=canvas(); d.text((120,90),'How to read each scene',font=F(50,True),fill=TXT)
items=[(RED,'Left window: fixed merge line','Used in most systems today.'),(AMB,'Middle window: turning straight line','Other fix: the line rotates to dodge people.'),
       (GRN,'Right window: proposed UA-OSM','The merge bends around each person (green outlines).'),
       (TXT,'Circles on people','Green = shown whole.  Orange = partly cut.  Red = erased by the merge line.'),
       (BLU,'Bottom left: the car\'s screen','Top view with UA-OSM, auto corner camera, distance warning.'),
       (MUT,'Bottom right: escalation ladder','1 simplest merge ... 5 warn the driver when a person cannot be shown whole.')]
y=200
for c,h,t in items:
    d.ellipse((130,y+10,160,y+40),fill=c); d.text((190,y),h,font=F(30,True),fill=TXT); d.text((190,y+44),t,font=F(24),fill=MUT); y+=130
write(im,'c_read.mp4',8)
# 3 group cards
for w,sub in (('Clear weather','Morning, afternoon, evening and night'),('Rain','Droplets and spray: detection gets noisier'),('Fog','Reduced range and contrast: detection gets harder')):
    im,d=canvas(); d.text((120,420),w,font=F(80,True),fill=TXT); d.text((124,540),sub,font=F(32),fill=MUT)
    write(im,f'c_{w.split()[0].lower()}.mp4',2.5)
# 4 results: main benchmark bars
im,d=canvas(); d.text((120,80),'Result 1: how often is a person lost?',font=F(48,True),fill=TXT)
d.text((122,150),'Share of frames in which at least one person disappears (105 random scenes, rendered pixels)',font=F(24),fill=MUT)
rows=[('Fixed merge line',63.0,RED),('One camera per object (naive)',27.6,(127,140,141)),('Curved merge line',24.0,(142,68,173)),('Turning straight line',19.9,AMB),('UA-OSM (proposed)',7.0,GRN)]
y=240
for n,v,c in rows:
    d.text((120,y),n,font=F(28,True if 'UA' in n else False),fill=TXT); d.rounded_rectangle((620,y-4,620+int(1100*v/70),y+40),8,fill=c); d.text((640+int(1100*v/70),y),f'{v:.1f} %',font=F(30,True),fill=TXT); y+=110
d.text((120,H-110),'UA-OSM loses a person about 3× less often than the turning straight line (Holm-adjusted p < 10⁻¹²).',font=F(26),fill=TXT)
write(im,'c_res1.mp4',8)
# 5 results: conditions heatmap figure
im,d=canvas(); d.text((120,60),'Result 2: all 12 times of day and weather conditions',font=F(46,True),fill=TXT)
fig=Image.open('figs/f8_conditions.png').convert('RGB'); s=1680/fig.width; fig=fig.resize((1680,int(fig.height*s))); im.paste(fig,(120,150))
yb=150+fig.height+30
d.text((120,yb),'UA-OSM is better in every condition. Straight line: 22–27 % lost frames. UA-OSM: 4.5 % (good light) to 13 % (night fog).',font=F(26),fill=TXT)
d.text((120,yb+45),'Its advantage is largest in good conditions (5× fewer losses) and smallest at night in rain or fog (2×). Crowds matter most.',font=F(26),fill=MUT)
write(im,'c_res2.mp4',10)
# 6 honest limits + message
im,d=canvas(); d.text((120,100),'What this shows, and what it does not',font=F(50,True),fill=TXT)
pts=[(GRN,'Shown','Straight merge lines cannot keep people whole near the corner; UA-OSM can, in far more frames.'),
     (GRN,'Shown','The gain comes from assigning people to cameras jointly, and holds across all 12 conditions.'),
     (AMB,'Open','Warnings fire too often (up to 40 % of frames at night in fog) and need tuning with drivers.'),
     (AMB,'Open','All results are simulated. Real 4-camera recordings are the next validation step.')]
y=240
for c,h,t in pts:
    d.rounded_rectangle((120,y,1800,y+120),14,fill=PANEL); d.text((150,y+18),h,font=F(28,True),fill=c); d.text((300,y+22),t,font=F(26),fill=TXT); y+=150
d.text((120,H-110),'Message: a parking view should never silently lose a person.',font=F(36,True),fill=GRN)
write(im,'c_end.mp4',9)
