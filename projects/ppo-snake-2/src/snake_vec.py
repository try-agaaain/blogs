"""ppo-snake-2 向量化环境：n_envs 个并行贪吃蛇。

与单环境 `SnakeEnv` 状态/奖励/动作语义完全一致（sim_rel 56 维 + 3 相对动作），
用 numpy 批量实现，供 PPO 训练 rollout 使用。

设计要点：
  * 每环境独立 `random.Random` 随机源，`reset(which)` 只重置指定环境
  * 蛇身用 (N, MAX_CELLS+4, 2) 缓冲，`lens` 记录当前长度；
    死亡环境保持 done 状态（训练循环负责 reset）
  * 修复旧项目长蛇边界 bug：吃食物增长时缓冲上限 148，安全处理满盘
  * 语义对齐：向量 body[i,0]=蛇头(heads 同步)、body[i,j]=第 j 节身体(j≥1)、
    body[i,lens-1]=蛇尾 —— 与单环境 snake=[头,身1..尾] 完全一致
"""

import random

import numpy as np

from src.snake_game import (WIDTH, HEIGHT, MAX_CELLS, STATE_DIM,
                            FRONT_SCAN, BODY_SEG_N,
                            REWARD_EAT, REWARD_HIT_WALL, REWARD_HIT_BODY,
                            REWARD_STEP, STEP_LIMIT,
                            _TURN_LEFT_NP, _TURN_RIGHT_NP,
                            _OCC_EMPTY, _OCC_BODY, _OCC_HEAD, _OCC_FOOD)

# 蛇身缓冲上限：比 MAX_CELLS 多 4 格余量（修复旧项目吃食物越界 bug）
_MAXLEN = MAX_CELLS + 4
_DIRS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32)  # up/down/left/right


class VectorSnakeEnv:
    """n_envs 个并行 sim_rel 贪吃蛇环境。"""

    def __init__(self, n_envs=64, seeds=None):
        self.n = n_envs
        if seeds is None:
            seeds = list(range(n_envs))
        seeds = list(seeds)
        assert len(seeds) == n_envs, \
            f"seeds 长度({len(seeds)})必须等于 n_envs({n_envs})"
        self.seeds = seeds
        self.rngs = [random.Random(s) for s in self.seeds]

        self.heads = np.zeros((n_envs, 2), dtype=np.int32)
        self.body = np.zeros((n_envs, _MAXLEN, 2), dtype=np.int32)
        self.lens = np.zeros(n_envs, dtype=np.int32)
        self.dirs = np.zeros(n_envs, dtype=np.int32)      # 0=up 1=down 2=left 3=right
        self.foods = np.full((n_envs, 2), -1, dtype=np.int32)
        self.occ = np.zeros((n_envs, HEIGHT, WIDTH), dtype=np.int8)
        self.done = np.zeros(n_envs, dtype=bool)
        self.food_eaten = np.zeros(n_envs, dtype=np.int32)
        self.steps = np.zeros(n_envs, dtype=np.int32)

        self.reset(np.arange(n_envs))

    # ---------------------------------------------------------- 重置
    def reset(self, which):
        """重置指定索引的环境，返回所有环境状态 (N, 38)。"""
        which = np.atleast_1d(np.asarray(which, dtype=np.int32))
        for i in which:
            self._reset_one(i)
        return self.get_states()

    def _reset_one(self, i):
        cx, cy = WIDTH // 2 - 2, HEIGHT // 2
        self.heads[i] = (cx, cy)
        self.body[i, 0] = (cx, cy)
        self.body[i, 1] = (cx - 1, cy)
        self.body[i, 2] = (cx - 2, cy)
        self.lens[i] = 3
        self.dirs[i] = 3                      # right
        self.done[i] = False
        self.food_eaten[i] = 0
        self.steps[i] = 0
        self.occ[i].fill(_OCC_EMPTY)
        self.occ[i, cy, cx] = _OCC_HEAD
        self.occ[i, cy, cx - 1] = _OCC_BODY
        self.occ[i, cy, cx - 2] = _OCC_BODY
        self._place_food(i)

    def _place_food(self, i):
        if self.foods[i, 0] >= 0:
            fx, fy = self.foods[i]
            if self.occ[i, fy, fx] == _OCC_FOOD:
                self.occ[i, fy, fx] = _OCC_EMPTY
        free = np.argwhere(self.occ[i] == _OCC_EMPTY)
        if len(free) == 0:
            self.foods[i] = (-1, -1)
            return
        ry, rx = free[self.rngs[i].randrange(len(free))]
        self.foods[i] = (int(rx), int(ry))
        self.occ[i, ry, rx] = _OCC_FOOD

    # ---------------------------------------------------------- 状态
    def get_states(self):
        """返回 (N, 38) sim_rel 状态，与单环境逐元素一致。"""
        N = self.n
        hx = self.heads[:, 0][:, None, None]
        hy = self.heads[:, 1][:, None, None]

        # 前向/左向轴（世界坐标）
        fdx = _DIRS[self.dirs, 0].astype(np.float32)
        fdy = _DIRS[self.dirs, 1].astype(np.float32)
        ldx = -fdy
        ldy = fdx

        # --- 1) 三线探测：每条线 k=1..FRONT_SCAN 格，每格 (has_food, block) ---
        lines_dir = np.stack([
            self.dirs,
            _TURN_LEFT_NP[self.dirs],
            _TURN_RIGHT_NP[self.dirs],
        ], axis=1)                            # (N, 3)
        line_dx = _DIRS[lines_dir][..., 0]
        line_dy = _DIRS[lines_dir][..., 1]
        k = np.arange(1, FRONT_SCAN + 1)
        px = hx + line_dx[:, :, None] * k[None, None, :]   # (N, 3, K)
        py = hy + line_dy[:, :, None] * k[None, None, :]
        in_bounds = ((px >= 0) & (px < WIDTH) & (py >= 0) & (py < HEIGHT))
        pxc = np.clip(px, 0, WIDTH - 1)
        pyc = np.clip(py, 0, HEIGHT - 1)
        c = self.occ[np.arange(N)[:, None, None], pyc, pxc]   # (N,3,K)
        has_food = np.where(c == _OCC_FOOD, 1.0, 0.0)
        has_food = np.where(in_bounds, has_food, 0.0)
        block = np.where(in_bounds,
                         np.where(c == _OCC_BODY, 0.5, 0.0),
                         1.0)
        scan = np.stack([has_food, block], axis=-1)          # (N,3,K,2)
        scan = scan.reshape(N, FRONT_SCAN * 3 * 2).astype(np.float32)

        # --- 2) 9 段等距采样点（弧长插值，沿 body[1..lens-1] 即身体+尾） ---
        n_body = np.maximum(self.lens - 1, 0).astype(np.float32)   # 身体节数
        seg_pts = np.zeros((N, BODY_SEG_N, 2), dtype=np.float32)
        rows = np.arange(N)
        for j in range(BODY_SEG_N):
            t = (j + 0.5) * n_body / BODY_SEG_N              # (N,)
            jj = np.floor(t).astype(np.int32)                # 0..n_body-1
            frac = t - jj
            has_body = n_body > 0
            # 单环境语义：jj >= n_body-1 时取尾（body[lens-1]），否则插值
            # 向量 body[k] 对应单环境 snake[k]（body[0]=头），身体从 body[1] 起
            beyond = jj >= (n_body - 1).astype(np.int32)     # 取尾标志
            # 左端点 body[jj+1]
            x0 = self.body[rows, np.clip(jj + 1, 0, _MAXLEN - 1), 0].astype(np.float32)
            y0 = self.body[rows, np.clip(jj + 1, 0, _MAXLEN - 1), 1].astype(np.float32)
            # 右端点：正常时 body[jj+2]，越界时用尾 body[n_body]（即 lens-1）
            tail_x = self.body[rows, np.clip(n_body.astype(np.int32),
                                             0, _MAXLEN - 1), 0].astype(np.float32)
            tail_y = self.body[rows, np.clip(n_body.astype(np.int32),
                                             0, _MAXLEN - 1), 1].astype(np.float32)
            x1 = np.where(beyond, tail_x,
                          self.body[rows, np.clip(jj + 2, 0, _MAXLEN - 1),
                                    0].astype(np.float32))
            y1 = np.where(beyond, tail_y,
                          self.body[rows, np.clip(jj + 2, 0, _MAXLEN - 1),
                                    1].astype(np.float32))
            bx = np.where(has_body, x0 + frac * (x1 - x0), 0.0)
            by = np.where(has_body, y0 + frac * (y1 - y0), 0.0)
            wx = (bx - hx[:, 0, 0]) / WIDTH
            wy = (by - hy[:, 0, 0]) / HEIGHT
            seg_pts[:, j, 0] = wx * fdx + wy * fdy
            seg_pts[:, j, 1] = wx * ldx + wy * ldy
        seg_feat = seg_pts.reshape(N, BODY_SEG_N * 2).astype(np.float32)

        # --- 3) 食物方向 ---
        ok = self.foods[:, 0] >= 0
        fx = np.where(ok, self.foods[:, 0].astype(np.float32), hx[:, 0, 0])
        fy = np.where(ok, self.foods[:, 1].astype(np.float32), hy[:, 0, 0])
        wx = (fx - hx[:, 0, 0]) / WIDTH
        wy = (fy - hy[:, 0, 0]) / HEIGHT
        food_feat = np.stack([wx * fdx + wy * fdy, wx * ldx + wy * ldy],
                             axis=1).astype(np.float32)
        food_feat[~ok] = 0.0

        return np.concatenate([scan, seg_feat, food_feat], axis=1)

    # ---------------------------------------------------------- 步进
    def step(self, actions):
        """执行 (N,) 相对动作，返回 (states, rewards, dones, infos)。"""
        a = np.asarray(actions, dtype=np.int32)
        assert a.shape == (self.n,), "actions must be length-n_envs"
        N = self.n

        # 相对 -> 绝对方向（死/超时环境不更新方向，与单环境死亡步语义一致）
        abs_a = np.where(a == 0, self.dirs,
                         np.where(a == 1, _TURN_LEFT_NP[self.dirs],
                                  _TURN_RIGHT_NP[self.dirs]))
        ndx = _DIRS[abs_a, 0]
        ndy = _DIRS[abs_a, 1]
        nhx = self.heads[:, 0] + ndx
        nhy = self.heads[:, 1] + ndy

        # 撞墙 / 追尾判定
        hit_wall = ((nhx < 0) | (nhx >= WIDTH) | (nhy < 0) | (nhy >= HEIGHT))
        ate = ((nhx == self.foods[:, 0]) & (nhy == self.foods[:, 1]))
        tx = self.body[np.arange(N), np.maximum(self.lens - 1, 0), 0]
        ty = self.body[np.arange(N), np.maximum(self.lens - 1, 0), 1]
        tail_clear = (~ate) & (nhx == tx) & (nhy == ty)
        nxc = np.clip(nhx, 0, WIDTH - 1)
        nyc = np.clip(nhy, 0, HEIGHT - 1)
        c = self.occ[np.arange(N), nyc, nxc]
        hit_body = ((c == _OCC_BODY) | (c == _OCC_HEAD)) & (~hit_wall)
        hit_body_dead = hit_body & ~tail_clear

        # 碰撞死亡环境不移动、不记步；其余环境（含将超时者）完整执行本步
        collision_dead = hit_wall | hit_body_dead
        move = ~collision_dead
        ate_ok = ate & move

        self.steps[move] += 1
        timed_out = move & (self.steps >= STEP_LIMIT)
        done_mask = collision_dead | timed_out

        # 方向：碰撞死亡环境保持原方向（与单环境死亡步一致），其余更新
        self.dirs = np.where(collision_dead, self.dirs, abs_a)

        # ---- 更新蛇身（move 环境前移，碰撞死亡环境保持原样） ----
        # 语义：body[0] 恒为蛇头（=heads），身体节 body[1..lens-1]
        new_body = self.body.copy()
        new_head = np.stack([nhx, nhy], axis=1)
        new_body[move, 0] = new_head[move]
        new_body[move, 1:] = self.body[move, :-1]
        old_lens = self.lens.copy()
        new_lens = old_lens + ate_ok.astype(np.int32)
        for i in np.where(ate_ok)[0]:
            if old_lens[i] < _MAXLEN:
                # 吃食物：新尾 = 旧尾（body[old_lens-1]）
                new_body[i, new_lens[i] - 1] = self.body[i, old_lens[i] - 1]
        self.body = new_body
        self.heads = np.where(move[:, None], new_head, self.heads)
        self.lens = new_lens

        # ---- occ 重建（向量化） ----
        new_occ = np.zeros((N, HEIGHT, WIDTH), dtype=np.int8)
        flat = new_occ.reshape(N, MAX_CELLS)
        # 食物
        fok = self.foods[:, 0] >= 0
        if fok.any():
            flat[fok, self.foods[fok, 1] * WIDTH + self.foods[fok, 0]] = _OCC_FOOD
        # 身体节（j=1..lens-1，即 body[1..]），头位置最后覆盖为 HEAD
        body_idx = self.body[..., 1] * WIDTH + self.body[..., 0]   # (N, _MAXLEN)
        js = np.arange(_MAXLEN)[None, :] < self.lens[:, None]       # j < lens 有效
        row = np.repeat(np.arange(N), _MAXLEN)
        col = body_idx.ravel()
        valid = js.ravel()
        flat[row[valid], col[valid]] = _OCC_BODY
        # 头（覆盖 body[0] 位置；碰撞死亡环境用旧头，其余用新头）
        hidx = self.heads[:, 1] * WIDTH + self.heads[:, 0]
        flat[np.arange(N), hidx] = _OCC_HEAD
        self.occ = new_occ

        # ---- 奖励（撞墙/撞身惩罚；吃到食物 +2；超时无惩罚） ----
        rewards = np.where(hit_wall, REWARD_HIT_WALL,
                           np.where(hit_body_dead, REWARD_HIT_BODY, REWARD_STEP))
        rewards = np.where(ate_ok, REWARD_EAT, rewards)

        # ---- 吃食物后放置新食物 ----
        for i in np.where(ate_ok)[0]:
            self.food_eaten[i] += 1
            self._place_food(i)

        # ---- info ----
        infos = [{"length": int(self.lens[i]),
                  "food_eaten": int(self.food_eaten[i])} for i in range(N)]

        return self.get_states(), rewards, done_mask, infos
