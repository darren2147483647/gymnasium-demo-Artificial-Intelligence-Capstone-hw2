# gymnasium遊戲實作

Artificial Intelligence Capstone 作業二

目標 : 
- 使用強化學習模型訓練並遊玩gymnasium小遊戲
- 實驗增進模型效能

### demo 內容

- Cart Pole(Classic control)
- Breakout(Atari)

### 實驗內容

- 去掉FIRE動作，改為自動發射
- 網路從雙層卷積改為resnet34

### 使用算法

- DQN
- PPO (only for Breakout)

### 訓練時間

Cart Pole : 45000+ episode (但最好的權重在前2000內)

Breakout(DQN) : 20000+ episode, ~21hr (還在訓練)

Breakout(PPO) : 10000 episode, ~8hr

Breakout(PPO, 實驗方法) : 共計~30hr

總爆肝時間 : ~5天

### 訓練成果

Cart Pole : 200/200 (屹立不倒)

Breakout(DQN) : 8/864 (有基本接球操作)

Breakout(PPO) : 55/864 (堅持半分鐘，即將打穿第一堵牆)

Breakout(PPO,autoshoot) : 85/864

Breakout(PPO,autoshoot,resnet) : 0/864

### 影片連結

(之後可能刪除)

- Cart Pole:

    使用DQN，達到滿分200分(生存至少200禎)

    https://youtu.be/KimVFjLrQB8

- Breakout:

    使用DQN，達到8分(滿分864)，存在類似”接球”的行為

    https://youtu.be/Sj2U0GkLzk8

    使用PPO，達到55分(滿分864)

    https://youtu.be/5cflGU6pbXw

    實驗(autoshoot)

    https://youtu.be/dzWIn9S5bXU

    實驗(autoshoot+resnet)

    https://youtu.be/Z6nE9qVRlBo

### 參考文獻

- carpole模型與訓練流程，但強化學習的算法似乎有誤，被我修正後可以用

https://github.com/pyliaorachel/openai-gym-cartpole/tree/master

- breakout模型與訓練流程，參考大部分訓練設置

https://github.com/yyc0314/DQN_atari_breakout/blob/main/DQN_breakout.ipynb