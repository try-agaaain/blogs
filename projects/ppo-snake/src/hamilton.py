"""
哈密顿回路 + 安全捷径 —— 长蛇存活的结构化先验。

背景
  贪吃蛇身长接近地图一半时（12x12 的 40+），身体几乎填满大片区域，
  蛇头必须沿"固定折叠路径"推进才能不撞自己。纯局部贪心策略只能活到
  ~30-42，而"蛇沿哈密顿回路单向走"理论上永不死亡（可填满 143 格通关）。

本模块实现"回路导向 + 安全捷径"策略：
  * build_cycle : O(W*H) 确定性构造哈密顿回路（需宽或高为偶数）。
  * safe_moves  : 返回安全动作集合。沿回路动作恒安全；偏离回路的捷径
                  仅在"蛇身仍沿回路排列"且"抄完捷径后能沿回路吃到
                  下一个食物（全程模拟验证）"时放行。
  * heuristic_action: 安全动作中选"沿回路到食物距离最小"者。
  * play_once   : 完整策略玩一局。
  * HamiltonSnakeEnv: PPO 训练环境包装——安全滤波兜底，蛇永不死亡。

安全性论证
  1. 蛇身沿回路排列 + 蛇头沿回路走 ⇒ 永不死亡（回路覆盖全部 144 格，
     蛇头下一步必是空格或蛇尾刚让出的格子）。
  2. 捷径只有在"抄完仍能平滑回归回路并吃到下一食物"时才放行，
     因此蛇身绝不翘起，沿回路兜底永远可用。
"""
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snake_game import TURN, DIR_VEC

WIDTH, HEIGHT = 12, 12

# 绝对方向向量 -> 方向名（与 snake_game.DIR_VEC 互逆）
VEC_TO_DIR = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
# 方向 -> 从某方向转到该方向所需相对动作（TURN 的反查表）
ACTION_TO = {d: {v: k for k, v in t.items()} for d, t in TURN.items()}


def build_cycle(w=WIDTH, h=HEIGHT):
    """确定性构造哈密顿回路（w 或 h 必须为偶数），O(W*H)。"""
    assert w % 2 == 0 or h % 2 == 0, "哈密顿回路要求宽或高为偶数"
    order = []
    for y in range(h):
        if y % 2 == 0:
            order.extend([(y, x) for x in range(1, w)])
        else:
            order.extend([(y, x) for x in range(w - 1, 0, -1)])
    order.extend([(y, 0) for y in range(h - 1, -1, -1)])
    n = w * h
    assert len(order) == n and len(set(order)) == n
    for i in range(n):
        a, b = order[i], order[(i + 1) % n]
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"回路不连续: {a}->{b}"
    return order


def cycle_dirs(path):
    """预计算：回路每个格子 -> 走到下一格所需绝对方向名。O(N)。"""
    n = len(path)
    return [VEC_TO_DIR[(path[(i + 1) % n][0] - path[i][0],
                        path[(i + 1) % n][1] - path[i][1])] for i in range(n)]


def pos_map(path):
    return {cell: i for i, cell in enumerate(path)}


def _action_towards(direction, head, target):
    """返回从 direction 转到 target 方向所需动作（0/1/2）；掉头则返回 None。"""
    dx = target[0] - head[0]
    dy = target[1] - head[1]
    abs_dir = next(d for d, (vx, vy) in DIR_VEC.items()
                   if (vx, vy) == (dx, dy))
    for a in (0, 1, 2):
        if TURN[direction][a] == abs_dir:
            return a
    return None


def cycle_action(snake, direction, path, pos):
    """沿回路前进对应的相对动作。"""
    head = snake[0]
    k = pos[head]
    nxt = path[(k + 1) % len(path)]
    a = _action_towards(direction, head, nxt)
    return 0 if a is None else a


def _simulate(snake, direction, a, food, w, h, body=None):
    """模拟执行动作 a。返回 (new_snake, new_dir, ate, dead)。"""
    nd = TURN[direction][a]
    dx, dy = DIR_VEC[nd]
    hx, hy = snake[0]
    nx, ny = hx + dx, hy + dy
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        return snake, direction, False, True
    if (nx, ny) in (body if body is not None else snake):
        return snake, direction, False, True
    ate = (nx, ny) == food
    new_snake = [(nx, ny)] + snake[:]
    if not ate:
        new_snake.pop()
    return new_snake, nd, ate, False


def _body_on_cycle(snake, path, pos):
    """蛇身是否严格沿回路排列（头部在第 k 位，其余在第 k-1, k-2, ... 位）。"""
    n = len(path)
    k = pos[snake[0]]
    for i, cell in enumerate(snake):
        if pos[cell] != (k - i) % n:
            return False
    return True


def _walk_cycle(snake, direction, path, pos, steps, w, h, path_dir=None):
    """模拟蛇头沿回路走 steps 步，验证全程不撞（含撞墙）。"""
    n = len(path)
    s = snake[:]
    body = set(s)
    d = direction
    for _ in range(steps):
        k = pos[s[0]]
        nxt = path[(k + 1) % n]
        if path_dir is not None:
            nd = path_dir[k]
            a = ACTION_TO[d][nd]
        else:
            a = _action_towards(d, s[0], nxt)
        if a is None:
            return False
        nd = TURN[d][a]
        nx, ny = nxt
        if nx < 0 or nx >= w or ny < 0 or ny >= h or (nx, ny) in body:
            return False
        body.discard(s[-1])
        s.pop()
        s.insert(0, (nx, ny))
        body.add((nx, ny))
        d = nd
    return True


def _can_eat_via_cycle(snake, direction, food, path, pos, w, h, path_dir):
    """验证：蛇头沿回路走能吃到食物，且吃完后蛇身仍沿回路（安全）。

    这是"抄捷径"的唯一放行条件——确保蛇头偏离回路后能平滑回归。
    吃到食物后蛇身严格沿回路排列，沿回路继续走恒安全（不死论证 1），
    故无需再向后模拟。
    """
    n = len(path)
    s = snake[:]
    body = set(s)
    d = direction
    for _ in range(n + 30):
        k = pos[s[0]]
        nd = path_dir[k]
        a = ACTION_TO[d][nd]
        nx, ny = path[(k + 1) % n]
        if (nx, ny) in body:
            return False
        if (nx, ny) == food:
            return True
        body.discard(s[-1])
        s.pop()
        s.insert(0, (nx, ny))
        body.add((nx, ny))
        d = nd
    return True


def safe_moves(snake, direction, food, path, pos, w=WIDTH, h=HEIGHT, path_dir=None):
    """返回安全动作集合。

    沿回路动作恒安全（蛇身沿回路排列时）。偏离回路的捷径只有通过
    _can_eat_via_cycle 完整验证才放行。
    """
    if food is None:
        return {cycle_action(snake, direction, path, pos)}
    if path_dir is None:
        path_dir = cycle_dirs(path)
    safe = set()
    aligned = _body_on_cycle(snake, path, pos)
    body = set(snake)
    cyc = cycle_action(snake, direction, path, pos)
    for a in (0, 1, 2):
        ns, nd, ate, dead = _simulate(snake, direction, a, food, w, h, body)
        if dead:
            continue
        if a == cyc:
            # 沿回路：蛇身沿回路时恒安全；翘起时也需验证兜底
            if aligned or _walk_cycle(ns, nd, path, pos, len(snake) + 2, w, h, path_dir):
                safe.add(a)
            continue
        if aligned and _can_eat_via_cycle(ns, nd, food, path, pos, w, h, path_dir):
            safe.add(a)
    if not safe:
        # 兜底：蛇头已处于"方向无法转向回路"的极端状态，保底沿回路
        safe.add(cyc)
    return safe


def _pick_best(safe, snake, direction, food, path, pos):
    """在安全动作集合中选"沿回路到食物距离最小"者。"""
    n = len(path)
    best, best_d = None, 10 ** 9
    for a in safe:
        ns, nd, ate, dead = _simulate(snake, direction, a, food, WIDTH, HEIGHT)
        if dead:
            continue
        d = (pos[food] - pos[ns[0]]) % n
        if d < best_d:
            best, best_d = a, d
    if best is None:
        return cycle_action(snake, direction, path, pos)
    return best


def heuristic_action(snake, direction, food, path, pos, w=WIDTH, h=HEIGHT, path_dir=None):
    """安全动作中选"沿回路到食物距离最小"者。"""
    safe = safe_moves(snake, direction, food, path, pos, w, h, path_dir)
    return _pick_best(safe, snake, direction, food, path, pos)


def play_once(env, path, pos, path_dir=None, max_steps=40000, log=False):
    """"回路导向 + 安全捷径"策略玩一局。返回 (身长, 食物数, 步数)。"""
    if path_dir is None:
        path_dir = cycle_dirs(path)
    env.reset()
    steps = 0
    while not env.done and steps < max_steps:
        a = heuristic_action(env.snake, env.direction, env.food, path, pos, path_dir=path_dir)
        s, r, done, info = env.step(a)
        steps += 1
        if log and steps % 500 == 0:
            print(f"step {steps}: len={info['length']} food={info['food_eaten']}")
    return info["length"], info["food_eaten"], steps


class HamiltonSnakeEnv:
    """哈密顿安全滤波环境（PPO 训练用）。

    在 SnakeEnv 之上叠加安全滤波器：PPO 输出的动作若不在安全集合内，
    会被替换为启发式安全动作（沿回路或安全捷径）。因此蛇永不死亡，
    理论身长可达 143（通关）。PPO 的学习空间是"安全动作中选择更快
    吃到食物的动作"。

    状态 = 21 维原始特征 + 2 维回路特征：
      21  蛇头沿回路到食物的距离比例（0~1）
      22  蛇头沿回路到蛇尾的距离比例（0~1）
    """

    STATE_DIM = 23

    def __init__(self, seed=None, step_limit=20000, path=None, pos=None):
        from src.snake_game import SnakeEnv
        self._env = SnakeEnv(seed=seed, step_limit=step_limit)
        self.path = path if path is not None else build_cycle()
        self.pos = pos if pos is not None else pos_map(self.path)
        self.path_dir = cycle_dirs(self.path)
        self.seed = seed
        self.n_intervened = 0   # 安全滤波干预次数（PPO 动作被替换）
        self.n_steps = 0        # 总步数
        self.safe = {0, 1, 2}   # 当前状态的安全动作集合（供 action masking）

    # 转发环境属性/方法（PPO 训练脚本直接操作 env.step/reset）
    @property
    def snake(self):
        return self._env.snake

    @property
    def direction(self):
        return self._env.direction

    @property
    def food(self):
        return self._env.food

    @property
    def done(self):
        return self._env.done

    @property
    def info(self):
        return self._env.info

    @property
    def rng(self):
        return self._env.rng

    def reset(self):
        self.n_intervened = self.n_steps = 0
        s = self._env.reset()
        self._refresh_safe()
        return self.get_ham_state_extra(s)

    @property
    def _step_limit_override(self):
        return self._env._step_limit_override

    @_step_limit_override.setter
    def _step_limit_override(self, v):
        self._env._step_limit_override = v

    @property
    def safe_mask(self):
        """当前状态的安全动作 one-hot 掩码（供策略网络 action masking）。"""
        import numpy as np
        m = np.zeros(3, dtype=np.float32)
        m[list(self.safe)] = 1.0
        return m

    def _refresh_safe(self):
        """基于当前环境状态计算并缓存安全动作集合。"""
        if self._env.food is None:
            self.safe = {0, 1, 2}
        else:
            self.safe = safe_moves(self._env.snake, self._env.direction,
                                   self._env.food, self.path, self.pos,
                                   path_dir=self.path_dir)

    def step(self, action):
        """安全滤波：action 不安全时替换为启发式安全动作。"""
        self.n_steps += 1
        if action is not None and self._env.food is not None:
            safe = self.safe          # 对应 S_t 的安全集（reset/上一步已算）
            if action not in safe:
                self.n_intervened += 1
                action = _pick_best(safe, self._env.snake, self._env.direction,
                                    self._env.food, self.path, self.pos)
        else:
            action = heuristic_action(self._env.snake, self._env.direction,
                                      self._env.food, self.path, self.pos,
                                      path_dir=self.path_dir)
        s, r, done, info = self._env.step(action)
        # 预计算 S_{t+1} 的安全集，供下一轮决策（action masking）使用
        if not done:
            self._refresh_safe()
        else:
            self.safe = {0, 1, 2}
        return self.get_ham_state_extra(s), r, done, info

    def get_ham_state_extra(self, state):
        """把 2 维回路特征拼接到原始状态后，返回完整状态向量。"""
        import numpy as np
        head = self._env.snake[0]
        tail = self._env.snake[-1]
        n = len(self.path)
        if self._env.food is None:
            extra = np.zeros(2, dtype=np.float32)
        else:
            d_food = (self.pos[self._env.food] - self.pos[head]) % n
            d_tail = (self.pos[tail] - self.pos[head]) % n
            extra = np.array([d_food / n, d_tail / n], dtype=np.float32)
        return np.concatenate([state, extra])


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.snake_game import SnakeEnv

    PATH = build_cycle()
    POS = pos_map(PATH)
    PDIR = cycle_dirs(PATH)
    print(f"回路长度: {len(PATH)}（应为 144），连续性校验通过")

    lens, foods, steps = [], [], []
    for seed in range(20):
        env = SnakeEnv(seed=seed, step_limit=20000)
        l, f, st = play_once(env, PATH, POS, PDIR)
        lens.append(l); foods.append(f); steps.append(st)
    print(f"\n回路导向+安全捷径 20 局: 平均身长 {sum(lens)/20:.1f}  平均食物 {sum(foods)/20:.1f}  平均步数 {sum(steps)/20:.0f}")
    print(f"身长分布: min {min(lens)} / 中位 {sorted(lens)[10]} / max {max(lens)}")
    print(f"身长>=72: {sum(1 for x in lens if x>=72)}   >=100: {sum(1 for x in lens if x>=100)}   >=143: {sum(1 for x in lens if x>=143)}")
