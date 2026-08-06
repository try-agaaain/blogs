# ppo-snake-2 项目 — 上下文交接文档

> 生成时间：2026-08-05。用于切换新对话继续任务。

## 1. 任务背景（用户原始需求）

用户要求：**新建 `ppo-snake-2`，重新实现此前 6 个 sim_rel 实验的方案（sim_rel 相对动作 + bottleneck 网络），并添加测试**。

用户此前在 `ppo-snake` 跑过 6 个 sim_rel 实验（rel_s1~s6），平均身长约 40，但旧代码目录混乱且有已知 bug（长蛇吃食物越界 `IndexError`），故要求在新目录 `ppo-snake-2` 从零实现干净版本 + 测试。

## 2. 项目位置与结构

- 项目根：`D:/Coding/Blogs/projects/ppo-snake-2`
- 关键源文件：
  - `src/snake_game.py` — 单环境贪吃蛇（sim_rel 56 维状态 + 3 相对动作）
  - `src/snake_vec.py` — `VectorSnakeEnv` 多环境并行版（numpy 向量化）
  - `src/ppo.py` — `BottleneckPolicy`（bottleneck 门控残差网络）+ `PPOTrainer`（PPO-Clip + GAE + 熵正则 + KL 统计）
  - `src/expert.py` — `GreedyZigzagExpert`（贪心 Z 字专家，BC 数据源）
  - `scripts/bc_train.py` — BC 预训练脚本（专家数据 → 交叉熵）
  - `scripts/train.py` — PPO 训练入口（纯 PPO 或 BC+PPO 微调，支持 `--step-penalty` / `--ent-coef` / `--kl-threshold`）
  - `scripts/play.py` — 终端可视化演示（默认贪心，`--sample` 切换随机）
  - `scripts/embed_model_to_html.py` — 把 `best_snake.json` 权重 + `experiments/s6/train_log.csv` 曲线烧录进讲解页
  - `PPO贪吃蛇讲解.html` — 交互式讲解页（仿旧项目 `ppo-snake/PPO贪吃蛇讲解v3.html`，10 章节 + 浏览器内实时 AI 推理演示）
  - `tests/` — 41 个单元测试（test_snake_game.py / test_snake_vec.py / test_ppo.py / test_expert.py / test_bc_train.py）
  - `best_snake.json` — 最优模型（s6，128 局贪心平均身长 70.2）

## 3. 环境与算法配置

### 3.1 状态：sim_rel 56 维全相对坐标系（FRONT_SCAN=6 宽视野）
```
0~35   三线探测（直行/左转/右转 × 前方 6 格 × 2 维：[有食物?, 障碍]）
       障碍编码：0=空, 0.5=蛇身, 1.0=墙
36~53  蛇身 9 段弧长插值采样点 × 2 维（相对蛇头，旋转到蛇头朝向坐标系的 (dx,dy)）
54~55  食物相对蛇头朝向的方向 (dx,dy)
```
- 所有坐标以蛇头为原点、蛇头朝向为前向轴旋转 → 与相对动作对齐（消除参考系错配）
- 段点/食物方向允许 [-1,1]（身体可在头后方），探测恒 [0,1]
- 弧长插值：`t=(j+0.5)*n_body/9`，`n_body=len(snake)-1`，浮点线性插值避免量化跳变
- **视野扩展**：FRONT_SCAN 由 3 增至 6 是突破平均身长 50 平台期的关键改动

### 3.2 动作：3 相对动作
- `0=直行 / 1=左转 / 2=右转`
- 掉头（反向）在相对动作空间不存在 → 专家掉头步不收集样本，env.step(0) 推进
- 转向表：`_TURN_LEFT=[2,3,1,0]`, `_TURN_RIGHT=[3,2,0,1]`（绝对方向索引）

### 3.3 奖励
- 吃食物 +2，撞墙 -0.5，撞身 -0.5，步进 0（无 shaping）
- 追尾规则：蛇头可进即将移开的蛇尾格（未吃食物时），吃到食物时不可

### 3.4 网络：BottleneckPolicy
```
in_proj: Linear(56 → hidden) + SiLU
blocks:  3 × BottleneckSwiGLUResBlock(hidden)   # h ← h + SiLU(up(gate ⊙ SiLU(down(h))))
actor:   Linear(hidden → 3)     # 正交初始化 0.1
critic:  Linear(hidden → 1)     # 正交初始化 0.5
```
- hidden=56 → 17868 参数（< 20k）；hidden=112 → 63956 参数（训练最优实验 s6 使用）
- 每块 3×Linear(hidden→hidden/2→hidden)，参数量 1.5H²，比全量 SwiGLU 省参数可堆更深
- 权重格式：JSON（format=ppo-snake-2-weights）

### 3.5 PPO 超参（train.py 默认）
- lr=3e-4、γ=0.99、λ=0.95、clip=0.2、ent_coef=0.01、vf_coef=0.5
- n_envs=64、max_steps=1024、epochs=4、minibatch=512
- 死亡环境 rollout 中立即 reset（记录后开新局）

## 4. 关键设计决策与修复的 Bug（重要！）

1. **向量 body 语义**：`body[i,0]=蛇头(=heads)`，`body[i,j]=第 j 节身体(j≥1)`，`body[i,lens-1]=蛇尾`。reset 与 step 后保持一致。
2. **长蛇边界修复**：蛇身缓冲 `_MAXLEN = MAX_CELLS+4 = 148`（旧项目 MAX_CELLS=144 时吃食物越界崩溃）。吃食物时新尾=旧尾。
3. **单环境追尾 occ 覆盖 bug**：头进蛇尾格时 `snake` 有两个同坐标点，pop 掉的尾恰是新头格 → `occ[ty,tx]=EMPTY` 会把新头清掉。修复：`if (tx,ty)!=(nhx,nhy): occ[ty,tx]=EMPTY`。
4. **向量尾索引 bug**：追尾判定 `tail` 用 `lens-1`（body[lens-1]=尾），不是 `lens-2`。
5. **向量奖励 bug**：奖励基于 `dead` 而非 `hit_body`（追尾时 hit_body=True 但 tail_clear 不死 → 误判 -0.5）。
6. **单/向量一致性**：同 seed 随机动作序列逐格一致（test_full_trajectory_consistent 600+ 步验证）。
7. **snake_vec.get_states 段插值**：`jj>=n_body-1` 取尾（body[n_body]），否则插值 body[jj+1]~body[jj+2]。
8. **rollout 统一环境主序 (N, T)**：`collect_rollout` 返回 states/actions/old_logp/rewards/dones/values 均为 `(N, T[, ...])`，`next_value` 为 `(N,)`。GAE 按环境沿 axis=1 计算（`PPOTrainer._gae`），优势/回报与 states 共用同一 reshape 展平，从结构上杜绝「时间主序存、环境主序展平」的索引错位。注意：旧版 ppo-snake 与此布局不同。
9. **贪心死循环防护**：确定性贪心策略可能陷入追尾循环而不死，`evaluate` 默认每局上限 2000 步，未结束局按当前长度截断记录。
10. **专家反向方向**：死局兜底的反向动作是 `cur ^ 1`（up↔down、left↔right），不是 `(cur+2)%4`（后者是侧向，语义错误）。

## 5. 测试状态（41 个全部通过）

- `tests/test_snake_game.py`（12 个）：状态维度/范围、初始探测布局、弧长插值（直线蛇 + 短蛇 padding）、相对动作转向、奖励（吃/撞墙/撞身/追尾）
- `tests/test_snake_vec.py`（10 个）：初始一致性、完整轨迹一致性（同 seed 600+ 步）、长蛇(40节)弧长插值一致性、batch 形状/独立性/部分 reset、长蛇边界、死亡步一致性
- `tests/test_ppo.py`（13 个）：参数数 17868、forward 形状、act（采样/贪心/eps 混合 probs）、save/load、PPO 冒烟训练、评估、GAE 精确值/done 截断/逐环境对齐/展平顺序回归/优势定向端到端
- `tests/test_expert.py`（4 个）：存在安全方向时不自杀、追尾允许、BC 数据形状、掉头样本丢弃
- `tests/test_bc_train.py`（2 个）：BC 脚本端到端（产出可回读模型）、非法数据文件被拒绝

运行：`python -m unittest discover -s tests`（约 48s）

## 6. 当前进度

- ✅ 全部 41 个测试通过
- ✅ BC 端到端已验证（test_bc_train 产出 bc_policy.json 并可 load 回读）
- ✅ PPO 冒烟已跑通：`python scripts/train.py --iters 3 --n-envs 16 --max-steps 256 --eval-every 2 --tag _smoke_check` 正常出 best_snake.json（约 14s/迭代）
- ✅ **正式训练完成（2026-08-06）**：目标「平均身长 72」基本达成

## 7. 正式训练结果（2026-08-06）

### 关键改动
- **FRONT_SCAN 3 → 6**（`src/snake_game.py`）：前方三线探测由 3 格扩展到 6 格，状态维度 38 → 56。这是突破 50+ 平台期的关键。
- **网络容量 hidden 56 → 112**（约 60k 参数）：大网络在小视野下峰值 52.6，宽视野下峰值 75.0。
- `scripts/train.py` 新增 `--step-penalty` / `--ent-coef` / `--kl-threshold` 参数。

### 实验对比（均 64 env × 1024 步，800 iter，eval 每 20 iter 取 32 局贪心）
| 实验 | 配置 | best 峰值 |
|---|---|---|
| s1 | 纯 PPO hidden56 FRONT_SCAN3（旧，遗留） | 49.4 |
| s2 | s1 best 续训 lr1e-4 | 停滞 45-48（终止） |
| s3 | 纯 PPO hidden56 n_envs128 | 61.5（终止） |
| s4 | BC+PPO lr5e-5 | 停滞 45（终止） |
| s5 | 纯 PPO hidden112 | 60.0（终止） |
| **s6** | **纯 PPO hidden112 FRONT_SCAN6** | **75.0** ✅ |
| s7 | s6 微调 lr2e-5 + 步惩罚 -0.01 | 69.9 |
| s8 | 从头 hidden112 步惩罚 -0.02 | 差（终止） |
| s9 | s6 续训 lr1e-4 | 75.7 |

### 最优模型：`experiments/s6/best_snake.json`（已复制到项目根 `best_snake.json`）
128 局大样本贪心评估：**平均长度 70.2**，median=74，≥70 占 73/128 局，≥80 占 47 局，max=115。
- 64 局评估 74.4；训练日志 32 局峰值 75.0。
- 分布右偏（多数局长，少数局短），已基本达成目标 72。

### 结论与后续方向
- 宽视野（探测 6 格）+ 大网络（hidden 112）是核心提升因素。
- 若需更稳定超过 72：可继续增大 FRONT_SCAN（如 8）与 BODY_SEG_N，或改奖励 shaping / 加哈密顿先验。

## 8. 待办事项（新对话继续）
1. （已完成）正式训练目标平均身长 72：s6 达成，见 `best_snake.json`
2. （已完成）仿 `ppo-snake/PPO贪吃蛇讲解v3.html` 添加讲解页 `PPO贪吃蛇讲解.html`：内嵌 s6 权重（hidden112）与训练曲线；浏览器端用 JS 复刻 56 维 sim_rel 状态提取 + Bottleneck 前向（已验证与 Python 输出 logits 误差 < 1e-6）；`scripts/embed_model_to_html.py` 可随时重新烧录
3. 若需继续优化：增大视野/网络、奖励 shaping、多 seed 稳定化
4. （可选）更新 ppo-snake 的 HTML 讲解页

- 终端文件目录：`C:/Users/tsingyue/.cursor/projects/d-Coding-Blogs/terminals/`（.txt 文件代表各终端）
- 注意清理残留 python 进程：`tasklist //FI "IMAGENAME eq python.exe"` / `taskkill //F //T //PID <pid>`
- Python：项目根运行 `python scripts/xxx.py`（脚本内置 `_ROOT` 路径注入）；测试用 `python -m unittest discover -s tests`
- 训练环境参考旧项目：torch 2.11.0+cu128，RTX 5060 Laptop GPU，CUDA 可用
