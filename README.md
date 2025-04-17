# gymnasium遊戲實作

Artificial Intelligence Capstone 作業二
目標 : 使用強化學習模型訓練並遊玩gymnasium小遊戲

### demo 內容

- Cart Pole(Classic control)
- Breakout(Atari)

### 使用算法

- DQN
- PPO (only for Breakout)

### 訓練時間

Cart Pole : 45000+ episode (但最好的權重在前2000內)
Breakout(DQN) : 10510+ episode, ~12hr (還在訓練)
Breakout(PPO) : 10000 episode, ~8hr

總爆肝時間 : ~3天

### 訓練成果

Cart Pole : 200/200 (屹立不倒)
Breakout(DQN) : 8/864 (有基本接球操作)
Breakout(PPO) : 55/864 (堅持半分鐘，即將打穿第一堵牆)