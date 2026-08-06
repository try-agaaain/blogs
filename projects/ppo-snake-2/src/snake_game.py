"""ppo-snake-2 单环境贪吃蛇。

核心设计（对应旧项目 6 个 sim_rel 实验的方案，宽视野扩展版）：
  * 状态：sim_rel 全相对坐标系（FRONT_SCAN=6，56 维）
      - 0~35   三线探测（直行/左转/右转 × 前方 6 格 × 2 维：有食物?, 障碍）
      - 36~53  蛇身 9 段弧长插值采样点 × 2 维（旋转到蛇头朝向坐标系）
      - 54~55  食物相对蛇头朝向的方向 (dx, dy)
  * 动作：3 相对（0=直行 / 1=左转 / 2=右转）
  * 奖励：吃 +2，死亡 -0.5，无步惩罚、无 shaping

相比旧项目修复的已知 bug：
  * 蛇接近满盘（lens→MAX_CELLS）时吃食物不再越界
  * 单环境 reset 用独立随机源，避免与向量环境种子不同步
"""

import random

import numpy as np

# ---------------------------------------------------------------- 常量
WIDTH, HEIGHT = 12, 12
MAX_CELLS = WIDTH * HEIGHT

# 状态布局
FRONT_SCAN = 6          # 每条探测线向前扫描格数（3→6 扩大视野，助长蛇突破 50 平台）
BODY_SEG_N = 9          # 蛇身等距采样点数
STATE_DIM = (FRONT_SCAN * 3 * 2 + BODY_SEG_N * 2 + 2)   # = 36+18+2 = 56

# 奖励
REWARD_EAT = 2.0
REWARD_HIT_WALL = -0.5
REWARD_HIT_BODY = -0.5
REWARD_STEP = 0.0
STEP_LIMIT = 20000

# 动作常量
N_ACTIONS = 3                    # 0=直行 1=左转 2=右转
DIR_INDEX = {"up": 0, "down": 1, "left": 2, "right": 3}
ABS_ACTION = [
    ("up", (0, -1)),       # 0
    ("down", (0, 1)),      # 1
    ("left", (-1, 0)),     # 2
    ("right", (1, 0)),     # 3
]
# 相对转向：绝对方向 -> 左转/右转后的绝对方向
_TURN_LEFT = [2, 3, 1, 0]     # up→left, down→right, left→down, right→up
_TURN_RIGHT = [3, 2, 0, 1]    # up→right, down→left, left→up, right→down
_TURN_LEFT_NP = np.array(_TURN_LEFT, dtype=np.int32)
_TURN_RIGHT_NP = np.array(_TURN_RIGHT, dtype=np.int32)

# occ 占用位图数值
_OCC_EMPTY = 0
_OCC_BODY = 1
_OCC_HEAD = 2
_OCC_FOOD = 3


def rel_to_abs(rel, dirs):
    """相对动作 (0=直行 1=左转 2=右转) → 绝对方向索引。dirs 为当前绝对方向。"""
    rel = np.asarray(rel, dtype=np.int32)
    dirs = np.asarray(dirs, dtype=np.int32)
    return np.where(rel == 0, dirs,
                    np.where(rel == 1, _TURN_LEFT_NP[dirs],
                             _TURN_RIGHT_NP[dirs]))


def abs_to_rel(abs_a, dirs):
    """绝对方向 → 相对动作。掉头（反向）返回 -1（调用方应丢弃）。"""
    abs_a = np.asarray(abs_a, dtype=np.int32)
    dirs = np.asarray(dirs, dtype=np.int32)
    out = np.full_like(abs_a, -1)
    out[abs_a == dirs] = 0
    out[abs_a == _TURN_LEFT_NP[dirs]] = 1
    out[abs_a == _TURN_RIGHT_NP[dirs]] = 2
    return out


class SnakeEnv:
    """单环境贪吃蛇，sim_rel 全相对状态 + 3 相对动作。"""

    def __init__(self, seed=None):
        self.width, self.height = WIDTH, HEIGHT
        self.max_cells = MAX_CELLS
        self.rng = random.Random(seed)
        self.snake = []
        self.occ = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        self.direction = "right"
        self.food = None
        self.steps = 0
        self.food_eaten = 0
        self.done = False
        self.last_reward = 0.0
        self.info = {}

    # ---------------------------------------------------------- 重置
    def _reset(self):
        # 蛇初始在中间偏左，长度 3，方向向右
        cx, cy = self.width // 2 - 2, self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "right"
        self.steps = 0
        self.food_eaten = 0
        self.done = False
        self.last_reward = 0.0
        self.occ[:] = 0
        self.occ[cy, cx] = _OCC_HEAD
        self.occ[cy, cx - 1] = _OCC_BODY
        self.occ[cy, cx - 2] = _OCC_BODY
        self._place_food()
        self.info = {"length": len(self.snake), "food_eaten": 0}

    def reset(self):
        self._reset()
        return self._get_state()

    def _place_food(self):
        if self.food is not None:
            fx, fy = self.food
            if self.occ[fy, fx] == _OCC_FOOD:
                self.occ[fy, fx] = _OCC_EMPTY
        free = np.argwhere(self.occ == _OCC_EMPTY)
        if len(free) == 0:
            self.food = None
            return
        ry, rx = free[self.rng.randrange(len(free))]
        self.food = (int(rx), int(ry))
        self.occ[ry, rx] = _OCC_FOOD

    # ---------------------------------------------------------- 状态
    def _get_state(self):
        """返回 56 维 sim_rel 状态。

        0~35   三线探测 18 格 × 2 维（[有食物?, 障碍]）
               障碍编码：0=空, 0.5=蛇身, 1.0=墙
        36~53  蛇身 9 段等距采样点 × 2 维（相对蛇头，旋转后 (dx, dy)）
        54~55  食物相对蛇头朝向的 (dx, dy)
        """
        hx, hy = self.snake[0]
        di = DIR_INDEX[self.direction]
        fdx, fdy = ABS_ACTION[di][1]          # 前向轴
        ldx, ldy = -fdy, fdx                  # 左向轴

        # --- 1) 三线探测 ---
        lines = [di, _TURN_LEFT[di], _TURN_RIGHT[di]]
        scan = []
        for ld in lines:
            dx, dy = ABS_ACTION[ld][1]
            for k in range(1, FRONT_SCAN + 1):
                px, py = hx + dx * k, hy + dy * k
                has_food = 0.0
                block = 0.0
                if px < 0 or px >= self.width or py < 0 or py >= self.height:
                    block = 1.0
                else:
                    c = int(self.occ[py, px])
                    if c == _OCC_BODY:
                        block = 0.5
                    elif c == _OCC_FOOD:
                        has_food = 1.0
                scan += [has_food, block]
        scan = np.asarray(scan, dtype=np.float32)

        # --- 2) 9 段等距采样点（弧长插值，浮点坐标） ---
        body_pts = self.snake[1:]
        n_body = len(body_pts)
        seg_pts = []
        if n_body > 0:
            for j in range(BODY_SEG_N):
                t = (j + 0.5) * n_body / BODY_SEG_N   # 弧长位置
                jj = int(t)
                frac = t - jj
                if jj >= n_body - 1:
                    bx, by = body_pts[-1]
                else:
                    x0, y0 = body_pts[jj]
                    x1, y1 = body_pts[jj + 1]
                    bx = x0 + frac * (x1 - x0)
                    by = y0 + frac * (y1 - y0)
                wx = (bx - hx) / self.width
                wy = (by - hy) / self.height
                seg_pts += [wx * fdx + wy * fdy, wx * ldx + wy * ldy]
        if len(seg_pts) < 2 * BODY_SEG_N:
            seg_pts += [0.0] * (2 * BODY_SEG_N - len(seg_pts))
        seg_feat = np.asarray(seg_pts, dtype=np.float32)

        # --- 3) 食物方向 ---
        if self.food is not None:
            wx = (self.food[0] - hx) / self.width
            wy = (self.food[1] - hy) / self.height
            food = [wx * fdx + wy * fdy, wx * ldx + wy * ldy]
        else:
            food = [0.0, 0.0]
        food_feat = np.asarray(food, dtype=np.float32)

        return np.concatenate([scan, seg_feat, food_feat])

    # ---------------------------------------------------------- 步进
    def step(self, action):
        """执行相对动作 (0=直行 1=左转 2=右转)，返回 (next_state, reward, done, info)。"""
        if self.done:
            return self._get_state(), 0.0, True, self.info
        assert action in (0, 1, 2), "相对动作必须是 0/1/2"

        di = DIR_INDEX[self.direction]
        abs_a = rel_to_abs(np.array([action]), np.array([di]))[0]
        ndx, ndy = ABS_ACTION[abs_a][1]

        hx, hy = self.snake[0]
        nhx, nhy = hx + ndx, hy + ndy

        # 撞墙
        if nhx < 0 or nhx >= self.width or nhy < 0 or nhy >= self.height:
            reward = REWARD_HIT_WALL
            self.last_reward = reward
            self.steps += 1
            return self._get_state(), reward, True, self._finish()

        # 追尾判定：能走进蛇尾格（蛇尾即将移开），但不能走进其他身体格
        tail_x, tail_y = self.snake[-1]
        ate = (nhx == self.food[0] and nhy == self.food[1]) if self.food else False
        hit_body_cell = self.occ[nhy, nhx] in (_OCC_BODY, _OCC_HEAD)
        tail_clear = (nhx == tail_x and nhy == tail_y and not ate)

        if hit_body_cell and not tail_clear:
            reward = REWARD_HIT_BODY
            self.last_reward = reward
            self.steps += 1
            return self._get_state(), reward, True, self._finish()

        # 正常移动
        self.snake.insert(0, (nhx, nhy))
        self.occ[nhy, nhx] = _OCC_HEAD
        if ate:
            # 吃食物：尾巴不移除，蛇变长
            self.occ[hy, hx] = _OCC_BODY     # 旧头变身体
            self.food_eaten += 1
            self._place_food()
            reward = REWARD_EAT
        else:
            # 未吃：移除尾巴
            self.occ[hy, hx] = _OCC_BODY
            tx, ty = self.snake.pop()
            # 追尾时旧尾=新头所在格，pop 的是旧尾；不能清掉新头
            if (tx, ty) != (nhx, nhy):
                self.occ[ty, tx] = _OCC_EMPTY
            reward = REWARD_STEP

        self.direction = ABS_ACTION[abs_a][0]
        self.steps += 1
        self.last_reward = reward
        self.info = {"length": len(self.snake), "food_eaten": self.food_eaten}

        # 步数上限：蛇达到 STEP_LIMIT 步则正常结束（无惩罚，done 供引导截断）
        if self.steps >= STEP_LIMIT:
            self.done = True
            return self._get_state(), reward, True, self.info

        return self._get_state(), reward, False, self.info

    def _finish(self):
        self.done = True
        self.info = {"length": len(self.snake), "food_eaten": self.food_eaten}
        return self.info
