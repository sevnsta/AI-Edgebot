"""
=======================================================================
  ADVIT EdgeBot — Advanced Edge AI Navigation System
  RoboEdge | CELESTAI'26 | Dayananda Sagar University
  Theme: Robotics Challenge (Edge AI Layer)
=======================================================================
  Innovations:
  1. Predictive Kalman Obstacle Tracking & Trajectory Forecasting
  2. Neural-Inspired A* (NA*) with obstacle-density heuristic
  3. Adaptive Speed Governor (clearance × curvature × confidence)
  4. 16-ray LiDAR Sensor Fusion with Kalman noise reduction
  5. Proactive Path Safety Check against predicted obstacle paths
  6. State Machine: PLANNING → NAVIGATING → EMERGENCY → ARRIVED
  7. Real-time Performance Analytics Dashboard
  8. Fuzzy Logic Reactive Navigation Layer
  9. XAI Decision Logger with natural language reasoning
  10. Semantic Obstacle Classification
=======================================================================
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import heapq
import os
import math
import json
from collections import defaultdict

# ───────────────────────────────────────────────────────────────
#  CONSTANTS
# ───────────────────────────────────────────────────────────────
ARENA_W       = 20.0
ARENA_H       = 20.0
CELL_SIZE     = 0.25
GRID_W        = int(ARENA_W / CELL_SIZE)
GRID_H        = int(ARENA_H / CELL_SIZE)
ROBOT_RADIUS  = 0.30
MAX_SPEED     = 1.20
MAX_ANG_SPD   = 1.50
SAFETY_MARGIN = 0.45
SENSOR_RANGE  = 5.0
NUM_RAYS      = 16
OUTPUT_DIR    = "simulations"

# ───────────────────────────────────────────────────────────────
#  KALMAN FILTER
# ───────────────────────────────────────────────────────────────
class KalmanFilter:
    def __init__(self, initial_pos):
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.05
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    def _F(self, dt):
        F = np.eye(4)
        F[0,2] = dt; F[1,3] = dt
        return F

    def predict(self, dt):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()

    def predict_future(self, steps, dt=0.3):
        x_tmp = self.x.copy()
        F = self._F(dt)
        out = []
        for _ in range(steps):
            x_tmp = F @ x_tmp
            out.append(x_tmp[:2].copy())
        return out

# ───────────────────────────────────────────────────────────────
#  XAI DECISION LOGGER
# ───────────────────────────────────────────────────────────────
class XAILogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.log = []
        self.phase_counts = defaultdict(int)

    def record(self, step, state, pos, speed, clearance, confidence, reason, emergency=False):
        entry = {
            "step": step,
            "state": state,
            "position": [round(float(pos[0]),3), round(float(pos[1]),3)],
            "speed_ms": round(float(speed), 4),
            "clearance_m": round(float(clearance), 4),
            "sensor_confidence": round(float(confidence), 4),
            "emergency": emergency,
            "reasoning": reason
        }
        self.log.append(entry)
        self.phase_counts[state] += 1

    def save(self):
        path = f"{self.out_dir}/xai_decision_log.json"
        with open(path, "w") as f:
            json.dump({"decisions": self.log, "phase_distribution": dict(self.phase_counts)}, f, indent=2)
        print(f"  [✓] XAI Log   → {path}  ({len(self.log)} decisions)")
        return path

# ───────────────────────────────────────────────────────────────
#  SEMANTIC OBSTACLE CLASSIFIER
# ───────────────────────────────────────────────────────────────
class SemanticClassifier:
    TYPES = {
        "wall_segment":    {"color": "#e63946", "urgency": 0.9, "strategy": "hard_avoid"},
        "dynamic_mover":   {"color": "#ff9f1c", "urgency": 1.0, "strategy": "predictive_avoid"},
        "large_block":     {"color": "#c77dff", "urgency": 0.8, "strategy": "smooth_avoid"},
        "checkpoint_cone": {"color": "#06d6a0", "urgency": 0.0, "strategy": "navigate_to"},
    }

    @staticmethod
    def classify_static(radius):
        if radius >= 1.0:
            return "large_block"
        return "wall_segment"

    @staticmethod
    def classify_dynamic():
        return "dynamic_mover"

# ───────────────────────────────────────────────────────────────
#  FUZZY LOGIC LAYER
# ───────────────────────────────────────────────────────────────
def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif x <= b: return (x-a)/(b-a)
    else: return (c-x)/(c-b)

def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif x <= b: return (x-a)/(b-a)
    elif x <= c: return 1.0
    else: return (d-x)/(d-c)

class FuzzySpeedController:
    def compute(self, clearance, confidence, curvature):
        # Fuzzify clearance
        c_danger  = trapmf(clearance, 0, 0, 0.5, 0.9)
        c_caution = trimf(clearance, 0.7, 1.5, 2.5)
        c_clear   = trapmf(clearance, 2.0, 3.0, 5.0, 5.0)
        # Fuzzify confidence
        conf_low  = trapmf(confidence, 0, 0, 0.3, 0.5)
        conf_high = trapmf(confidence, 0.4, 0.7, 1.0, 1.0)
        # Rules → speed
        rules = [
            (c_danger,                        0.10),
            (min(c_caution, conf_low),        0.25),
            (min(c_caution, conf_high),       0.50),
            (min(c_clear,   conf_low),        0.65),
            (min(c_clear,   conf_high),       MAX_SPEED),
        ]
        total_w = sum(r[0] for r in rules)
        if total_w < 1e-6: return 0.2
        speed = sum(r[0]*r[1] for r in rules) / total_w
        # Reduce for high curvature
        speed *= max(0.3, 1 - curvature * 0.5)
        return float(np.clip(speed, 0.10, MAX_SPEED))

# ───────────────────────────────────────────────────────────────
#  DYNAMIC OBSTACLE
# ───────────────────────────────────────────────────────────────
class DynamicObstacle:
    def __init__(self, pos, radius, velocity=None):
        self.pos        = np.array(pos, dtype=float)
        self.radius     = radius
        self.velocity   = np.array(velocity or [0.0, 0.0], dtype=float)
        self.kf         = KalmanFilter(pos)
        self.confidence = 0.0
        self.sem_type   = SemanticClassifier.classify_dynamic()

    def update(self, dt):
        self.pos += self.velocity * dt
        for i, limit in enumerate([ARENA_W, ARENA_H]):
            if self.pos[i] <= self.radius:     self.velocity[i] =  abs(self.velocity[i])
            if self.pos[i] >= limit-self.radius: self.velocity[i] = -abs(self.velocity[i])
        self.pos = np.clip(self.pos, [self.radius]*2, [ARENA_W-self.radius, ARENA_H-self.radius])
        self.kf.predict(dt)
        self.kf.update(self.pos)

# ───────────────────────────────────────────────────────────────
#  ARENA
# ───────────────────────────────────────────────────────────────
class Arena:
    def __init__(self):
        self.start = np.array([1.5, 1.5])
        self.goal  = np.array([18.5, 18.5])
        self.static_obs = [
            (5.0,5.0,0.8),(5.0,6.0,0.8),(5.0,7.0,0.8),
            (10.0,2.5,0.8),(10.0,3.5,0.8),(10.0,4.5,0.8),
            (8.0,12.0,1.0),
            (15.0,8.0,0.7),(15.0,9.0,0.7),
            (3.0,15.0,0.8),(4.0,15.0,0.8),
            (12.0,15.0,1.0),
            (7.0,18.0,0.6),
            (17.0,5.0,0.9),
            (14.0,12.0,0.7),
        ]
        self.dynamic_obs = [
            DynamicObstacle([10.0,10.0],0.50,[ 0.30, 0.20]),
            DynamicObstacle([ 6.0,14.0],0.40,[-0.25, 0.30]),
            DynamicObstacle([15.0,15.0],0.50,[ 0.20,-0.30]),
            DynamicObstacle([12.0, 7.0],0.40,[-0.30,-0.20]),
        ]
        # Semantic labels for static obs
        self.static_sem = [SemanticClassifier.classify_static(r) for _,_,r in self.static_obs]

    def update(self, dt):
        for obs in self.dynamic_obs: obs.update(dt)

    def occupancy_grid(self):
        grid = np.zeros((GRID_H, GRID_W), dtype=float)
        def mark(cx, cy, r, val):
            sr = r + SAFETY_MARGIN
            gx0 = max(0, int((cx-sr)/CELL_SIZE))
            gx1 = min(GRID_W-1, int((cx+sr)/CELL_SIZE))
            gy0 = max(0, int((cy-sr)/CELL_SIZE))
            gy1 = min(GRID_H-1, int((cy+sr)/CELL_SIZE))
            for gx in range(gx0,gx1+1):
                for gy in range(gy0,gy1+1):
                    if math.hypot(gx*CELL_SIZE-cx, gy*CELL_SIZE-cy) <= sr:
                        grid[gy,gx] = val
        for ox,oy,r in self.static_obs: mark(ox,oy,r,1.0)
        for obs in self.dynamic_obs:    mark(obs.pos[0],obs.pos[1],obs.radius,0.85)
        return grid

    def in_collision(self, pos, r=ROBOT_RADIUS):
        x,y = pos
        if not (r < x < ARENA_W-r and r < y < ARENA_H-r): return True
        for ox,oy,or_ in self.static_obs:
            if math.hypot(x-ox,y-oy) <= or_+r+0.05: return True
        for obs in self.dynamic_obs:
            if math.hypot(x-obs.pos[0],y-obs.pos[1]) <= obs.radius+r+0.05: return True
        return False

# ───────────────────────────────────────────────────────────────
#  NEURAL-INSPIRED A* PLANNER
# ───────────────────────────────────────────────────────────────
class NAStarPlanner:
    def __init__(self, arena):
        self.arena = arena

    def _h(self, node, goal, grid):
        dx,dy = node[0]-goal[0], node[1]-goal[1]
        base = math.hypot(dx,dy)
        density = sum(
            grid[min(GRID_H-1,max(0,node[1]+dy2)), min(GRID_W-1,max(0,node[0]+dx2))]
            for dx2 in range(-2,3) for dy2 in range(-2,3)
        ) / 25.0
        return base * (1 + 0.35*density)

    def plan(self, start_world, goal_world, grid=None):
        if grid is None: grid = self.arena.occupancy_grid()
        to_grid = lambda p: (max(0,min(GRID_W-1,int(p[0]/CELL_SIZE))),
                              max(0,min(GRID_H-1,int(p[1]/CELL_SIZE))))
        to_world = lambda g: np.array([g[0]*CELL_SIZE+CELL_SIZE/2, g[1]*CELL_SIZE+CELL_SIZE/2])
        S = to_grid(start_world)
        G = to_grid(goal_world)
        if grid[S[1],S[0]] > 0.5:
            for r in range(1,8):
                found=False
                for dx in range(-r,r+1):
                    for dy in range(-r,r+1):
                        nx,ny=S[0]+dx,S[1]+dy
                        if 0<=nx<GRID_W and 0<=ny<GRID_H and grid[ny,nx]<=0.5:
                            S=(nx,ny); found=True; break
                    if found: break
                if found: break
        open_q=[(0.0,S)]; came={}
        g_sc=defaultdict(lambda: float('inf')); g_sc[S]=0.0
        DIRS=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
        while open_q:
            _,cur=heapq.heappop(open_q)
            if cur==G or (abs(cur[0]-G[0])<=1 and abs(cur[1]-G[1])<=1):
                path=[]
                while cur in came: path.append(to_world(cur)); cur=came[cur]
                path.reverse(); path.append(goal_world.copy()); return path
            for ddx,ddy in DIRS:
                nb=(cur[0]+ddx,cur[1]+ddy)
                if not (0<=nb[0]<GRID_W and 0<=nb[1]<GRID_H): continue
                if grid[nb[1],nb[0]]>0.5: continue
                cost=1.414 if ddx and ddy else 1.0
                new_g=g_sc[cur]+cost
                if new_g<g_sc[nb]:
                    came[nb]=cur; g_sc[nb]=new_g
                    heapq.heappush(open_q,(new_g+self._h(nb,G,grid),nb))
        return []

    @staticmethod
    def smooth(path, alpha=0.5, beta=0.3, iters=60):
        if len(path)<3: return path
        s=[p.copy() for p in path]
        for _ in range(iters):
            for i in range(1,len(s)-1):
                for j in range(2):
                    s[i][j]+=alpha*(path[i][j]-s[i][j])
                    s[i][j]+=beta*(s[i-1][j]+s[i+1][j]-2*s[i][j])
        return s

# ───────────────────────────────────────────────────────────────
#  SENSOR SYSTEM
# ───────────────────────────────────────────────────────────────
class SensorSystem:
    def __init__(self, arena):
        self.arena = arena
        self.kf_rays = [KalmanFilter([SENSOR_RANGE,0]) for _ in range(NUM_RAYS)]

    def scan(self, pos, heading):
        readings=[]; detections=[]
        for i in range(NUM_RAYS):
            angle=heading+2*math.pi*i/NUM_RAYS
            hit_d=SENSOR_RANGE; hit_t='free'
            for d in np.arange(0.1,SENSOR_RANGE,0.1):
                cx=pos[0]+d*math.cos(angle); cy=pos[1]+d*math.sin(angle)
                if not (0<cx<ARENA_W and 0<cy<ARENA_H):
                    hit_d=d; hit_t='wall'; break
                hit=False
                for ox,oy,r in self.arena.static_obs:
                    if math.hypot(cx-ox,cy-oy)<=r:
                        hit_d=d; hit_t='static'; hit=True; break
                if not hit:
                    for obs in self.arena.dynamic_obs:
                        if math.hypot(cx-obs.pos[0],cy-obs.pos[1])<=obs.radius:
                            hit_d=d; hit_t='dynamic'
                            conf=0.70+0.25*(1-d/SENSOR_RANGE)
                            obs.confidence=conf
                            detections.append({'pos':obs.pos.copy(),'radius':obs.radius,
                                               'confidence':conf,'velocity':obs.velocity.copy()})
                            hit=True; break
                if hit: break
            noisy=hit_d+np.random.normal(0,0.04)
            filtered=self.kf_rays[i].update([noisy,0])[0]
            readings.append(float(np.clip(filtered,0,SENSOR_RANGE)))
        return readings, detections

    def fwd_clearance(self, pos, heading):
        clearance=SENSOR_RANGE
        for ao in [-0.30,-0.15,0.0,0.15,0.30]:
            ang=heading+ao
            for d in np.arange(0.1,SENSOR_RANGE,0.05):
                px=pos[0]+d*math.cos(ang); py=pos[1]+d*math.sin(ang)
                if self.arena.in_collision([px,py],0.0):
                    clearance=min(clearance,d); break
        return clearance

# ───────────────────────────────────────────────────────────────
#  EDGE AI BRAIN  (combined: original + fuzzy + XAI + semantic)
# ───────────────────────────────────────────────────────────────
class EdgeAIBrain:
    def __init__(self, arena, xai_logger):
        self.arena   = arena
        self.planner = NAStarPlanner(arena)
        self.sensors = SensorSystem(arena)
        self.fuzzy   = FuzzySpeedController()
        self.xai     = xai_logger
        self.path=[]; self.pidx=0; self.step_n=0
        self.REPLAN_EVERY=25
        self.metrics=dict(distance=0.0,replans=0,emergencies=0,conf=[],speed=[],clearance=[])
        self.state='PLANNING'

    def _path_curvature(self, path, idx):
        if idx<1 or idx>=len(path)-1: return 0.0
        v1=path[idx]-path[idx-1]; v2=path[idx+1]-path[idx]
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        if n1<1e-4 or n2<1e-4: return 0.0
        return math.acos(np.clip(np.dot(v1/n1,v2/n2),-1,1))

    def _predictions(self, steps=6, dt=0.35):
        return {i:obs.kf.predict_future(steps,dt) for i,obs in enumerate(self.arena.dynamic_obs)}

    def _path_safe(self, path, preds):
        if not path: return True
        for i,obs in enumerate(self.arena.dynamic_obs):
            for fpos in preds.get(i,[]):
                for wp in path[:8]:
                    if np.linalg.norm(wp-fpos)<obs.radius+ROBOT_RADIUS+SAFETY_MARGIN:
                        return False
        return True

    def _build_reason(self, state, clearance, speed, preds_unsafe):
        if state=='EMERGENCY':
            return f"EMERGENCY: clearance={clearance:.2f}m below safe threshold. Executing best-direction escape."
        if state=='REPLANNING':
            reason = f"Path replanned. "
            if preds_unsafe: reason += "Previous path conflicted with predicted obstacle trajectory. "
            reason += f"NA* found new corridor. Speed={speed:.2f}m/s"
            return reason
        if state=='NAVIGATING':
            return f"Path clear. Fuzzy speed governor set {speed:.2f}m/s based on clearance={clearance:.2f}m."
        return f"State={state}, speed={speed:.2f}m/s, clearance={clearance:.2f}m"

    def step(self, robot_pos, heading, goal, dt):
        self.step_n+=1
        readings,detections=self.sensors.scan(robot_pos,heading)
        clearance=self.sensors.fwd_clearance(robot_pos,heading)
        conf=float(np.mean(readings))/SENSOR_RANGE
        preds=self._predictions()
        self.metrics['conf'].append(conf)
        self.metrics['clearance'].append(clearance)

        if np.linalg.norm(robot_pos-goal)<0.55:
            self.state='ARRIVED'
            self.metrics['speed'].append(0.0)
            self.xai.record(self.step_n,'ARRIVED',robot_pos,0.0,clearance,conf,"Goal reached!")
            return goal.copy(),0.0,self._info(readings,detections,clearance,preds)

        if clearance<ROBOT_RADIUS+0.45:
            self.state='EMERGENCY'
            self.metrics['emergencies']+=1
            best_a,best_c=heading+math.pi,0.0
            for ta in np.linspace(0,2*math.pi,24,endpoint=False):
                c=0.0
                for d in np.arange(0.1,2.0,0.1):
                    if self.arena.in_collision([robot_pos[0]+d*math.cos(ta),robot_pos[1]+d*math.sin(ta)]): break
                    c=d
                if c>best_c: best_c,best_a=c,ta
            tgt=robot_pos+0.3*np.array([math.cos(best_a),math.sin(best_a)])
            self.metrics['speed'].append(0.18)
            reason=self._build_reason('EMERGENCY',clearance,0.18,False)
            self.xai.record(self.step_n,'EMERGENCY',robot_pos,0.18,clearance,conf,reason,True)
            info=self._info(readings,detections,clearance,preds); info['emergency']=True
            return tgt,0.18,info

        self.state='NAVIGATING'
        preds_unsafe=not self._path_safe(self.path[self.pidx:],preds)
        need_replan=(not self.path or self.pidx>=len(self.path)-1 or
                     self.step_n%self.REPLAN_EVERY==0 or preds_unsafe)
        if need_replan:
            grid=self.arena.occupancy_grid()
            newpath=self.planner.plan(robot_pos,goal,grid)
            if newpath:
                self.path=self.planner.smooth(newpath)
                self.pidx=0; self.metrics['replans']+=1
                self.state='REPLANNING'

        if self.path and self.pidx<len(self.path):
            tgt=self.path[self.pidx]
            if np.linalg.norm(robot_pos-tgt)<0.40:
                self.pidx=min(self.pidx+1,len(self.path)-1)
                tgt=self.path[self.pidx]
        else:
            tgt=goal.copy()

        curv=self._path_curvature(self.path,self.pidx) if self.path else 0.0
        # USE FUZZY CONTROLLER for final speed (innovation)
        speed=self.fuzzy.compute(clearance,conf,curv)
        self.metrics['speed'].append(speed)
        reason=self._build_reason(self.state,clearance,speed,preds_unsafe)
        self.xai.record(self.step_n,self.state,robot_pos,speed,clearance,conf,reason)
        return tgt,speed,self._info(readings,detections,clearance,preds)

    def _info(self,readings,detections,clearance,preds):
        return dict(readings=readings,detections=detections,clearance=clearance,
                    state=self.state,path=self.path,pidx=self.pidx,
                    emergency=False,preds=preds,
                    metrics={k:list(v) if isinstance(v,list) else v for k,v in self.metrics.items()})

# ───────────────────────────────────────────────────────────────
#  ROBOT
# ───────────────────────────────────────────────────────────────
class Robot:
    def __init__(self, start):
        self.pos=np.array(start,dtype=float)
        self.angle=math.pi/4
        self.speed=0.0
        self.trail=[self.pos.copy()]
        self.dist=0.0

    def move(self, target, speed, dt, arena):
        d=target-self.pos
        dist_to_tgt=np.linalg.norm(d)
        if dist_to_tgt<0.01: self.speed=0.0; return
        ta=math.atan2(d[1],d[0])
        diff=ta-self.angle
        while diff> math.pi: diff-=2*math.pi
        while diff<-math.pi: diff+=2*math.pi
        self.angle+=np.clip(diff,-MAX_ANG_SPD*dt,MAX_ANG_SPD*dt)
        v=min(speed,dist_to_tgt/dt)
        npos=self.pos+v*dt*np.array([math.cos(self.angle),math.sin(self.angle)])
        if not arena.in_collision(npos,ROBOT_RADIUS):
            self.dist+=np.linalg.norm(npos-self.pos)
            self.pos=npos; self.speed=v
        else:
            self.speed=0.0
        self.trail.append(self.pos.copy())
        if len(self.trail)>600: self.trail.pop(0)

# ───────────────────────────────────────────────────────────────
#  VISUALISER
# ───────────────────────────────────────────────────────────────
class Visualizer:
    def __init__(self):
        plt.style.use('dark_background')
        self.fig=plt.figure(figsize=(18,10))
        self.fig.patch.set_facecolor('#0d1117')
        gs=gridspec.GridSpec(2,3,figure=self.fig,hspace=0.38,wspace=0.30,
                              left=0.04,right=0.97,top=0.92,bottom=0.06)
        self.ax_main =self.fig.add_subplot(gs[:,  :2])
        self.ax_lidar=self.fig.add_subplot(gs[0,2],projection='polar')
        self.ax_speed=self.fig.add_subplot(gs[1,2])

    def render(self, arena, robot, ai, info, step):
        for a in [self.ax_main,self.ax_speed]: a.cla()
        self.ax_lidar.cla()
        ax=self.ax_main
        ax.set_facecolor('#0d1117')
        ax.set_xlim(0,ARENA_W); ax.set_ylim(0,ARENA_H); ax.set_aspect('equal')
        ax.set_title('ADVIT EdgeBot — Real-Time Navigation',color='#00ff9f',fontsize=13,fontweight='bold',pad=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.tick_params(colors='#4a5568',labelsize=7)
        for x in range(0,int(ARENA_W)+1,2): ax.axvline(x,color='#161b22',lw=0.3,alpha=0.6)
        for y in range(0,int(ARENA_H)+1,2): ax.axhline(y,color='#161b22',lw=0.3,alpha=0.6)
        ax.add_patch(patches.Rectangle((0,0),ARENA_W,ARENA_H,fill=False,ec='#30363d',lw=2))

        path=info['path']; pidx=info['pidx']
        if len(path)>1:
            pa=np.array(path)
            ax.plot(pa[:pidx+1,0],pa[:pidx+1,1],'--',color='#4cc9f0',alpha=0.25,lw=1.5)
            ax.plot(pa[pidx:,0],pa[pidx:,1],'-',color='#4cc9f0',alpha=0.80,lw=2.0,label='Planned Path (NA*)')

        if len(robot.trail)>1:
            tr=np.array(robot.trail); n=len(tr)
            for i in range(1,n):
                al=0.2+0.8*i/n
                ax.plot(tr[i-1:i+1,0],tr[i-1:i+1,1],'-',color='#f72585',alpha=al*0.85,lw=2)

        preds=info['preds']
        for idx,obs in enumerate(arena.dynamic_obs):
            fp=preds.get(idx,[])
            if fp:
                fa=np.array(fp)
                ax.plot(fa[:,0],fa[:,1],'--',color='#ff9f1c',alpha=0.40,lw=1.5,
                        label='Predicted Traj.' if idx==0 else '')
                for j,fpos in enumerate(fp[:3]):
                    ax.add_patch(patches.Circle(fpos,obs.radius*0.7,color='#ff9f1c',alpha=0.4-j*0.12,fill=True))

        # Static obstacles with semantic colors
        sem_colors={'wall_segment':'#e63946','large_block':'#c77dff','checkpoint_cone':'#06d6a0'}
        for i,(ox,oy,r) in enumerate(arena.static_obs):
            stype=arena.static_sem[i]
            col=sem_colors.get(stype,'#e63946')
            ax.add_patch(patches.Circle((ox,oy),r,color=col,alpha=0.88))
            ax.add_patch(patches.Circle((ox,oy),r,fill=False,ec='#ff6b6b',lw=1.5))

        for obs in arena.dynamic_obs:
            ax.add_patch(patches.Circle(obs.pos,obs.radius,color='#ff9f1c',alpha=0.90))
            if obs.confidence>0:
                ax.add_patch(patches.Circle(obs.pos,obs.radius+0.22,fill=False,ec='#ffff00',lw=1.8,alpha=obs.confidence))
                ax.text(obs.pos[0],obs.pos[1]+obs.radius+0.45,f'{obs.confidence:.0%}',
                        color='#ffff00',fontsize=7,ha='center',fontweight='bold')
            if np.linalg.norm(obs.velocity)>0.05:
                ax.annotate('',xy=(obs.pos[0]+obs.velocity[0],obs.pos[1]+obs.velocity[1]),
                            xytext=tuple(obs.pos),arrowprops=dict(arrowstyle='->',color='#ff9f1c',lw=2))

        for i,d in enumerate(info['readings']):
            if d<SENSOR_RANGE*0.65:
                a=robot.angle+2*math.pi*i/NUM_RAYS
                ax.plot([robot.pos[0],robot.pos[0]+d*math.cos(a)],
                        [robot.pos[1],robot.pos[1]+d*math.sin(a)],color='#7b2fbe',alpha=0.22,lw=0.8)

        em=info.get('emergency',False)
        rc='#ff3333' if em else '#00ff9f'
        ax.add_patch(patches.Circle(robot.pos,ROBOT_RADIUS,color=rc,alpha=0.92,zorder=12))
        ax.add_patch(patches.Circle(robot.pos,ROBOT_RADIUS+0.5,fill=False,ec=rc,lw=1,alpha=0.28,linestyle='--',zorder=11))
        dx_=ROBOT_RADIUS*1.6*math.cos(robot.angle); dy_=ROBOT_RADIUS*1.6*math.sin(robot.angle)
        ax.annotate('',xy=(robot.pos[0]+dx_,robot.pos[1]+dy_),xytext=tuple(robot.pos),
                    arrowprops=dict(arrowstyle='->',color=rc,lw=2.5))

        for cent,col,lbl in [(arena.goal,'#06d6a0','GOAL'),(arena.start,'#4cc9f0','START')]:
            ax.add_patch(patches.Circle(cent,0.55,color=col,alpha=0.45,zorder=5))
            ax.add_patch(patches.Circle(cent,0.28,color=col,alpha=0.90,zorder=6))
            ax.text(cent[0],cent[1]+0.85,lbl,color=col,fontsize=8,ha='center',fontweight='bold')

        sc={'PLANNING':'#4cc9f0','NAVIGATING':'#00ff9f','REPLANNING':'#ffff00',
            'EMERGENCY':'#ff3333','ARRIVED':'#06d6a0'}
        st=info['state']
        ax.text(10,19.3,f'AI STATE: {st}',color=sc.get(st,'#ffffff'),fontsize=12,fontweight='bold',ha='center')

        m=info['metrics']
        dtg=np.linalg.norm(robot.pos-arena.goal)
        hud=(f"Step {step:04d}  |  Dist {robot.dist:.1f}m  |  Replans {m['replans']}  |  "
             f"Emergencies {m['emergencies']}  |  Clearance {info['clearance']:.2f}m  |  To Goal {dtg:.1f}m")
        self.fig.text(0.50,0.01,hud,ha='center',color='#718096',fontsize=8,family='monospace')
        self.fig.suptitle("ADVIT EdgeBot  ·  Advanced Edge AI  ·  RoboEdge CELESTAI'26",
                          color='#e2e8f0',fontsize=12,fontweight='bold',y=0.98)

        al=self.ax_lidar; al.set_facecolor('#0d1117')
        if info['readings']:
            angles=[2*math.pi*i/NUM_RAYS for i in range(NUM_RAYS)]+[0]
            r_vals=list(info['readings'])+[info['readings'][0]]
            al.plot(angles,r_vals,color='#7b2fbe',lw=1.8,alpha=0.85)
            al.fill(angles,r_vals,color='#7b2fbe',alpha=0.15)
            al.set_ylim(0,SENSOR_RANGE); al.set_yticklabels([]); al.set_xticklabels([])
            al.grid(color='#30363d',alpha=0.5)
            al.set_title(f'LiDAR ({NUM_RAYS} rays + Kalman)',color='#4cc9f0',fontsize=9)

        asp=self.ax_speed
        asp.set_facecolor('#0d1117')
        asp.set_title('Fuzzy Speed + Clearance',color='#f72585',fontsize=10,pad=6)
        for sp in asp.spines.values(): sp.set_color('#30363d')
        asp.tick_params(colors='#4a5568',labelsize=7)
        spd=m['speed']; clr=m['clearance']
        if spd:
            asp.plot(spd,color='#f72585',lw=1.6,label='Speed (m/s)')
            asp.axhline(MAX_SPEED,color='#ff6b6b',lw=0.8,ls='--',alpha=0.45)
        if clr:
            cnorm=[min(c/SENSOR_RANGE*MAX_SPEED,MAX_SPEED) for c in clr]
            asp.plot(cnorm,color='#4cc9f0',lw=1.2,alpha=0.7,label='Clearance (norm)')
        asp.set_xlim(0,max(60,len(spd))); asp.set_ylim(0,MAX_SPEED+0.30)
        asp.set_xlabel('Step',color='#4a5568',fontsize=8); asp.set_ylabel('m/s',color='#4a5568',fontsize=8)
        asp.legend(fontsize=7,loc='upper right',facecolor='#161b22',labelcolor='white',framealpha=0.75)

        leg=[patches.Patch(color='#e63946',label='Static Obstacle'),
             patches.Patch(color='#ff9f1c',label='Dynamic Obstacle'),
             patches.Patch(color='#00ff9f',label='Robot'),
             Line2D([0],[0],color='#4cc9f0',lw=2,label='Planned Path (NA*)'),
             Line2D([0],[0],color='#f72585',lw=2,label='Robot Trail'),
             Line2D([0],[0],color='#ff9f1c',lw=1.5,ls='--',label='Obstacle Forecast'),
             Line2D([0],[0],color='#7b2fbe',lw=1,label='LiDAR Rays')]
        ax.legend(handles=leg,loc='upper left',fontsize=7,facecolor='#161b22',labelcolor='white',framealpha=0.82)
        return self.fig

# ───────────────────────────────────────────────────────────────
#  DASHBOARD
# ───────────────────────────────────────────────────────────────
def generate_dashboard(ai, robot, out):
    plt.style.use('dark_background')
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle("ADVIT EdgeBot  ·  AI Performance Dashboard  ·  RoboEdge CELESTAI'26",
                 color='#e2e8f0',fontsize=13,fontweight='bold')
    m=ai.metrics; spd=m['speed']; clr=m['clearance']; conf=m['conf']
    def style(ax,title,col):
        ax.set_facecolor('#0d1117'); ax.set_title(title,color=col,fontsize=11)
        ax.tick_params(colors='#4a5568')
        for sp in ax.spines.values(): sp.set_color('#30363d')
    ax=axes[0,0]; style(ax,'Fuzzy Adaptive Speed Profile','#f72585')
    if spd:
        ax.plot(spd,color='#f72585',lw=1.6); ax.fill_between(range(len(spd)),spd,alpha=0.18,color='#f72585')
        ax.axhline(MAX_SPEED,color='#ff6b6b',ls='--',alpha=0.5,lw=0.9,label='Max Speed')
        ax.set_ylabel('m/s',color='#718096',fontsize=9); ax.legend(fontsize=8,labelcolor='white',facecolor='#161b22')
    ax=axes[0,1]; style(ax,'Safety Clearance Monitor','#4cc9f0')
    if clr:
        for i in range(1,len(clr)):
            c=clr[i]; col='#00ff9f' if c>2.0 else '#ff9f1c' if c>1.0 else '#e63946'
            ax.plot([i-1,i],[clr[i-1],c],color=col,lw=2.2)
        ax.axhline(1.0,color='#ff9f1c',ls='--',alpha=0.6,lw=1,label='Warning')
        ax.axhline(0.5,color='#e63946',ls='--',alpha=0.6,lw=1,label='Danger')
        ax.set_ylabel('meters',color='#718096',fontsize=9); ax.legend(fontsize=8,labelcolor='white',facecolor='#161b22')
    ax=axes[1,0]; style(ax,'Kalman Sensor Fusion Confidence','#00ff9f')
    if conf:
        ax.plot(conf,color='#00ff9f',lw=1.3,alpha=0.6,label='Raw')
        w=min(20,len(conf)); ra=np.convolve(conf,np.ones(w)/w,mode='valid')
        ax.plot(range(w-1,len(conf)),ra,color='#06d6a0',lw=2.5,label=f'Rolling avg ({w})')
        ax.set_ylim(0,1.1); ax.set_ylabel('Confidence',color='#718096',fontsize=9)
        ax.legend(fontsize=8,labelcolor='white',facecolor='#161b22')
    ax=axes[1,1]; ax.set_facecolor('#161b22'); ax.axis('off')
    ax.text(0.5,0.97,'PERFORMANCE SUMMARY',transform=ax.transAxes,
            color='#e2e8f0',fontsize=12,fontweight='bold',ha='center',va='top')
    stats=[('Total Distance Traveled',f'{robot.dist:.2f} m'),
           ('Path Replans (NA*)',f'{m["replans"]}'),
           ('Emergency Stops',f'{m["emergencies"]}'),
           ('Avg Sensor Confidence',f'{np.mean(conf):.1%}' if conf else 'N/A'),
           ('Avg Navigation Speed',f'{np.mean(spd):.2f} m/s' if spd else 'N/A'),
           ('Peak Speed Used',f'{max(spd):.2f} m/s' if spd else 'N/A'),
           ('Minimum Clearance',f'{min(clr):.2f} m' if clr else 'N/A'),
           ('Total AI Decisions',f'{len(spd)}')]
    for i,(lbl,val) in enumerate(stats):
        y=0.82-i*0.095
        ax.text(0.08,y,lbl+':',transform=ax.transAxes,color='#718096',fontsize=10,va='center')
        ax.text(0.78,y,val,transform=ax.transAxes,color='#4cc9f0',fontsize=10,va='center',fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out}/performance_dashboard.png',dpi=120,bbox_inches='tight',facecolor='#0d1117')
    plt.close(fig)
    print(f"  [✓] Dashboard → {out}/performance_dashboard.png")

# ───────────────────────────────────────────────────────────────
#  SAVE METRICS JSON
# ───────────────────────────────────────────────────────────────
def save_metrics(ai, robot, out):
    data={'total_distance_m':round(robot.dist,3),
          'path_replans':ai.metrics['replans'],
          'emergency_stops':ai.metrics['emergencies'],
          'avg_speed_ms':round(float(np.mean(ai.metrics['speed'])),4) if ai.metrics['speed'] else 0,
          'avg_confidence':round(float(np.mean(ai.metrics['conf'])),4) if ai.metrics['conf'] else 0,
          'min_clearance_m':round(float(min(ai.metrics['clearance'])),4) if ai.metrics['clearance'] else 0,
          'ai_decisions_total':len(ai.metrics['speed']),
          'algorithm':'Neural-Inspired A* (NA*) + Kalman EKF + Fuzzy Speed Governor + XAI Logger + Semantic Classifier'}
    with open(f'{out}/metrics.json','w') as f: json.dump(data,f,indent=2)
    print(f"  [✓] Metrics   → {out}/metrics.json")

# ───────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────
def run():
    np.random.seed(42)
    print("="*62)
    print("  ADVIT EdgeBot — Advanced Edge AI Navigation System")
    print("  RoboEdge | CELESTAI'26 | Dayananda Sagar University")
    print("="*62)
    os.makedirs(f'{OUTPUT_DIR}/frames',exist_ok=True)
    arena=Arena()
    robot=Robot(arena.start)
    xai=XAILogger(OUTPUT_DIR)
    ai=EdgeAIBrain(arena,xai)
    viz=Visualizer()
    dt=0.10; MAX_STEPS=500; SAVE_EVERY=8
    frame=0; info=None; step=0
    for step in range(MAX_STEPS):
        arena.update(dt)
        tgt,spd,info=ai.step(robot.pos,robot.angle,arena.goal,dt)
        robot.move(tgt,spd,dt,arena)
        if step%SAVE_EVERY==0:
            fig=viz.render(arena,robot,ai,info,step)
            path_f=f'{OUTPUT_DIR}/frames/frame_{frame:04d}.png'
            fig.savefig(path_f,dpi=95,bbox_inches='tight',facecolor='#0d1117')
            plt.close(fig); frame+=1
            dtg=np.linalg.norm(robot.pos-arena.goal)
            print(f"  Step {step:4d} | {info['state']:12s} | Pos ({robot.pos[0]:.1f},{robot.pos[1]:.1f}) | Goal {dtg:.1f}m | Speed {spd:.2f} | Clr {info['clearance']:.2f}m")
        if info and info['state']=='ARRIVED':
            print(f"\n  *** GOAL REACHED at step {step}! ***"); break
    if info:
        fig=viz.render(arena,robot,ai,info,step)
        fig.savefig(f'{OUTPUT_DIR}/final_state.png',dpi=130,bbox_inches='tight',facecolor='#0d1117')
        plt.close(fig)
        print(f"  [✓] Final    → {OUTPUT_DIR}/final_state.png")
    generate_dashboard(ai,robot,OUTPUT_DIR)
    save_metrics(ai,robot,OUTPUT_DIR)
    xai.save()
    print(f"\n  {'─'*50}")
    print(f"  Distance traveled : {robot.dist:.2f} m")
    print(f"  Replans           : {ai.metrics['replans']}")
    print(f"  Emergency stops   : {ai.metrics['emergencies']}")
    if ai.metrics['conf']: print(f"  Avg confidence    : {np.mean(ai.metrics['conf']):.1%}")
    print(f"  Frames saved      : {frame}")
    print(f"  Output folder     : ./{OUTPUT_DIR}/")
    print(f"  {'─'*50}\n")

if __name__=='__main__':
    run()
