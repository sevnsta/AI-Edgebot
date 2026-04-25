# 🤖 ADVIT EdgeBot — Advanced Edge AI Navigation System
### RoboEdge | CELESTAI'26 | Dayananda Sagar University

![Final State](final_state.png)

---

## 🏆 Overview

**ADVIT EdgeBot** is a full-stack Edge AI autonomous navigation system built for the **RoboEdge CELESTAI'26** Robotics Challenge. It demonstrates a 10-layer cognitive AI stack running entirely **on-device** — zero cloud, zero GPU required.

> Evaluation: **60% robotic task** + **40% Edge AI intelligence layer**  
> This system is purpose-built to dominate the 40% Edge AI marks.

---

## 🧠 10 AI Innovations (What No Other Team Has Combined)

| # | Innovation | What It Does |
|---|---|---|
| 1 | **Kalman EKF Obstacle Tracker** | Tracks each moving obstacle's position AND velocity in real-time |
| 2 | **Trajectory Forecasting** | Predicts where each obstacle will be 6 steps into the future |
| 3 | **Neural-Inspired A\* (NA\*)** | Path planner that avoids dense obstacle zones — like a neural net but runs in <5ms |
| 4 | **Proactive Path Safety Check** | Checks if planned path conflicts with *predicted* future positions before moving |
| 5 | **16-ray Kalman-Filtered LiDAR** | Each sensor ray noise-filtered individually through a Kalman filter |
| 6 | **Fuzzy Logic Speed Governor** | 3-input fuzzy system: clearance × confidence × curvature → smooth speed |
| 7 | **Semantic Obstacle Classifier** | Labels each obstacle (wall_segment, dynamic_mover, large_block) with urgency scores |
| 8 | **State Machine** | PLANNING → REPLANNING → NAVIGATING → EMERGENCY → ARRIVED |
| 9 | **XAI Decision Logger** | Every decision logged with natural language reasoning for full audit trail |
| 10 | **Adaptive Path Smoothing** | Gradient-descent path smoother removes zigzags from A\* output |

---

## 📊 Simulated Performance Results

| Metric | Value |
|---|---|
| Total AI Decisions | **500** |
| Path Replans (NA\*) | **28** |
| Emergency Stops | **0** |
| Average Sensor Confidence | **78.4%** |
| Average Navigation Speed | **0.37 m/s** |
| Minimum Clearance Maintained | **1.45 m** |
| Distance Traveled | **18.43 m** |

---

## 📁 Repository Structure

```
advit-edgebot/
├── simulation.py              ← Complete AI system (single file, fully runnable)
├── requirements.txt           ← Just numpy + matplotlib
├── README.md                  ← This file
├── docs/
│   └── robot_specifications.md ← Hardware build document (submission upload)
└── simulations/               ← ALL generated outputs (pre-run, ready to view)
    ├── final_state.png        ← Final arena visualization
    ├── performance_dashboard.png ← 4-panel analytics dashboard
    ├── metrics.json           ← Machine-readable performance metrics
    ├── xai_decision_log.json  ← 500 decisions with natural language reasoning
    └── frames/                ← 63 animation frames (frame_0000.png → frame_0062.png)
```

---

## 🚀 How to Run

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/advit-edgebot
cd advit-edgebot

# Install (only 2 packages)
pip install numpy matplotlib

# Run full simulation (generates all outputs automatically)
python simulation.py
```

**Outputs generated automatically:**
- `simulations/final_state.png` — arena map with robot path
- `simulations/performance_dashboard.png` — 4-panel analytics
- `simulations/metrics.json` — performance summary
- `simulations/xai_decision_log.json` — full XAI audit trail
- `simulations/frames/` — 63 animation frames

---

## 🔬 Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 5: XAI EXPLAINABILITY ENGINE                      │
│  Natural language reasoning for every decision           │
├─────────────────────────────────────────────────────────┤
│  LAYER 4: COGNITIVE MISSION STATE MACHINE               │
│  PLANNING → REPLANNING → NAVIGATING → EMERGENCY → ARRIVED│
├─────────────────────────────────────────────────────────┤
│  LAYER 3: SEMANTIC OBSTACLE CLASSIFICATION              │
│  wall_segment | dynamic_mover | large_block             │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: NEURAL-INSPIRED A* + KALMAN PREDICTION        │
│  Obstacle-density heuristic + future trajectory check   │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: 16-RAY LIDAR + FUZZY SPEED GOVERNOR           │
│  Per-ray Kalman filtering + 3-input fuzzy inference     │
└─────────────────────────────────────────────────────────┘
```

---

## 📸 Screenshots

### Final State — Arena Navigation
![Final State](final_state.png)

### Performance Dashboard
![Dashboard](performance_dashboard.png)

---

## 👥 Team

**Latency zero* — Dayananda Sagar College of Engineering (DSCE), Bangalore  
Dayananda Sagar University | RoboEdge CELESTAI'26

---

## 📄 Sample XAI Decision Log

```json
{
  "step": 25,
  "state": "REPLANNING",
  "position": [3.6, 3.0],
  "speed_ms": 1.2,
  "clearance_m": 2.31,
  "sensor_confidence": 0.812,
  "emergency": false,
  "reasoning": "Path replanned. Previous path conflicted with predicted obstacle trajectory. NA* found new corridor. Speed=1.20m/s"
}
```

Every one of the **500 decisions** in `simulations/xai_decision_log.json` includes: position, speed, clearance, confidence, and the **reason why** the AI made that decision.

---

*Built for RoboEdge CELESTAI'26 — Edge AI Intelligence Layer*
