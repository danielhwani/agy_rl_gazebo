# TurtleBot3 Gazebo Reinforcement Learning (DQN)

이 프로젝트는 **Gymnasium과 같은 서드파티 라이브러리에 종속되지 않고**, **ROS 2 Humble** 및 **Gazebo 시뮬레이터**와 직접 통신하며 학습하는 **순수 PyTorch 기반 심층 Q-네트워크(DQN) 강화학습 프레임워크**입니다.

---

## 🎯 학습 목표 및 규칙

경기장(Stage 1 Arena, $5\text{m} \times 5\text{m}$) 중앙에서 시작하는 **터틀봇(TurtleBot3 Burger)**이 경기장 내 무작위 위치에 생성되는 **붉은색 원형 목표 마커(`goal_box`)를 최단 경로로 찾아가도록 학습**합니다.

```
+-----------------------------------+
|               Wall                |
|                                   |
|        [🔴 Goal Box]              |
|                                   |
|             [🤖 TB3]              |
|             (Center)              |
|                                   |
|                                   |
|               Wall                |
+-----------------------------------+
```

### 1. 주요 판정 조건
* **🎯 목표 도달 성공 (Goal Reached)**: 로봇과 빨간색 목표 마커의 거리가 **$0.25\text{m}$ 이내**로 접근하면 성공 판정 (**`+200.0`** 보상).
* **💥 벽 충돌 실패 (Collision)**: 360도 LiDAR 센서로 측정한 벽과의 최소 거리가 **$0.13\text{m}$ 이하**가 되면 충돌 실패 판정 (**`-100.0`** 감점 후 에피소드 즉시 종료).
* **⏳ 시간 초과 (Timeout)**: 제한된 스텝(기본 400스텝) 내에 목표에 도달하지 못하면 실패 (**`-20.0`** 감점 후 다음 에피소드로 리셋).

### 2. 보상 함수 설계 (Reward Shaping)
* **거리 단축 진행 보상**: $(d_{\text{prev}} - d_{\text{curr}}) \times 50.0$ (목표에 가까워질수록 보상 부여)
* **방향 정렬 페널티**: $- \frac{|\Delta\theta|}{\pi} \times 0.15$ (목표 반대 방향을 바라볼수록 감점)
* **스텝 비용**: $-0.05$ (최단 시간 내 최단 경로 주행 유도)

---

## 🌟 관측(State) 및 행동(Action) 공간

### 1. 관측 상태 공간 (26차원)
* **LiDAR 거리 데이터 (24차원)**: 360도 레이저 스캔을 15도씩 24개 방위 섹터로 분할하고 최솟값을 추출하여 정규화 ($[0.0, 1.0]$).
* **목표까지의 거리 (1차원)**: 경기장 대각선 기준 정규화된 유클리드 거리 ($[0.0, 1.0]$).
* **목표를 향한 상대 방위각 (1차원)**: 현재 터틀봇의 Heading 각도와 목표점 방향 간의 각도 오차 ($[-\pi, \pi] \rightarrow [-1.0, 1.0]$).

### 2. 이산 제어 행동 공간 (5가지)
* `0`: 직진 ($v = 0.18 \text{ m/s}, \omega = 0.0 \text{ rad/s}$)
* `1`: 좌회전 ($v = 0.06 \text{ m/s}, \omega = 0.8 \text{ rad/s}$)
* `2`: 우회전 ($v = 0.06 \text{ m/s}, \omega = -0.8 \text{ rad/s}$)
* `3`: 완만한 좌회전 ($v = 0.14 \text{ m/s}, \omega = 0.35 \text{ rad/s}$)
* `4`: 완만한 우회전 ($v = 0.14 \text{ m/s}, \omega = -0.35 \text{ rad/s}$)

---

## 📁 디렉터리 구조

```text
/home/daniel/agy_rl_gazebo/
├── gazebo_rl/
│   ├── __init__.py
│   ├── gazebo_env.py       # ROS 2 & Gazebo 환경 인터페이스 (순수 Python/rclpy)
│   ├── dqn_agent.py        # PyTorch DQN 신경망 & Experience Replay Buffer
│   ├── train.py            # 메인 학습 루프, 체크포인트 및 실시간 그래프 생성
│   └── evaluate.py         # 학습된 가중치 기반 탐색 없는 최적 정책 평가
├── launch/
│   ├── start_gazebo.sh     # Gazebo Stage 1 시뮬레이션 단독 실행 스크립트
│   ├── start_training.sh   # Gazebo + DQN 학습 원클릭 실행 스크립트
│   └── start_evaluation.sh # Gazebo + DQN 평가 실행 스크립트
├── models/                 # 학습된 PyTorch 가중치 (best / latest .pth)
├── logs/                   # 학습 로그 (.json) 및 결과 그래프 (.png)
└── README.md
```

---

## 🚀 실행 방법 (일반 터미널 기준)

우분투 기본 터미널(`Ctrl + Alt + T` 또는 일반 쉘)에서 실행하는 두 가지 방법입니다.

---

### 방법 A. 원클릭 런처 스크립트 사용 (가장 간편함 ⭐)

스크립트 내부에서 ROS 2 환경 로드, Gazebo 실행 여부 확인 및 백그라운드 구동, `rl_gazebo` 가상환경 연결을 모두 자동으로 처리합니다.

```bash
# 1. 프로젝트 폴더로 이동
cd /home/daniel/agy_rl_gazebo

# 2. 강화학습 훈련 실행
./launch/start_training.sh --episodes 150 --max-steps 400

# (선택) 이전 모델에서 이어서 훈련할 경우:
./launch/start_training.sh --episodes 150 --load-model models/dqn_stage1_best.pth

# 3. 학습된 최적 모델 실시간 평가 주행
./launch/start_evaluation.sh --episodes 5 --max-steps 350
```

---

### 방법 B. 터미널 2개로 분리하여 실행 (로그 디버깅 & 수동 제어용)

Gazebo 시뮬레이션 창과 강화학습 노드를 각각 별도의 터미널에서 띄우고 싶을 때 사용합니다.

#### 📌 [터미널 1] Gazebo 시뮬레이션 실행
```bash
# ROS 2 및 TurtleBot3 환경 로드
source /opt/ros/humble/setup.bash
source /home/daniel/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/daniel/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models

# Gazebo 월드 실행
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
```

#### 📌 [터미널 2] PyTorch 강화학습 훈련 노드 실행
```bash
# rl_gazebo 가상환경 활성화 및 ROS 2 패키지 연동
conda activate rl_gazebo
source /opt/ros/humble/setup.bash
source /home/daniel/turtlebot3_ws/install/setup.bash

# Conda와 ROS2 C++ 라이브러리 간 호환성 환경변수 설정
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 프로젝트 디렉터리로 이동 후 학습 실행
cd /home/daniel/agy_rl_gazebo
python -m gazebo_rl.train --episodes 150 --max-steps 400

# (또는 평가 실행)
python -m gazebo_rl.evaluate --episodes 5 --max-steps 350
```

---

## 📊 실험 및 평가 결과 (초기 10회 학습 모델 기준)

### 1. 훈련 과정 로그
* **10 에피소드 훈련**: 4번째 에피소드에서 **97 스텝 만에 목표 지점에 정확히 도달** (`Reward: +286.9`)
* **최적 가중치 저장**: `models/dqn_stage1_best.pth`
* **훈련 지표 그래프**: `logs/training_plot_*.png`

### 2. 평가 주행 결과 (5개 에피소드 Greedy Policy)

| 에피소드 | 주행 결과 | 소요 스텝 | 보상 (Reward) | 상세 비고 |
| :---: | :---: | :---: | :---: | :--- |
| **Test Ep 01** | ⏳ TIMEOUT | 350 | `+0.9` | 벽 충돌 없이 안전 주행 |
| **Test Ep 02** | ⏳ TIMEOUT | 350 | `-3.7` | 벽 충돌 없이 안전 주행 |
| **Test Ep 03** | ⏳ TIMEOUT | 350 | `-0.1` | 벽 충돌 없이 안전 주행 |
| **Test Ep 04** | ⏳ TIMEOUT | 350 | `-98.2` | 벽 충돌 없이 안전 주행 |
| **Test Ep 05** | **🎯 SUCCESS** | **75** | **`+267.9`** | **목표 지점 최단 경로 도달 성공!** |

* **성공률**: **20.0%** (초기 10회 학습 가중치 기준)
* **벽 충돌 횟수**: **0회 (0%)** - LiDAR 기반 충돌 회피가 안정적으로 작동
* **목표 도달 스텝**: **75 스텝** (신속한 경로 탐색)
* **평균 보상**: **`+33.35`**
