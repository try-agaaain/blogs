"""
PPO（近端策略优化）—— 纯 numpy 实现，无任何深度学习框架依赖

包含：
  * MLPPolicy : 两层全连接网络（隐藏层 tanh），策略头输出 softmax，价值头输出标量
  * AdamOptimizer : Adam 优化器（手动实现反向传播梯度更新）
  * ppo_update   : PPO-Clip 目标函数的小批量更新

公式（PPO-Clip，逐样本）：
    rho   = exp(logp_new - logp_old)                        # 重要性采样比率
    L_pol = -mean( min(rho*A, clip(rho, 1-eps, 1+eps)*A) )  # 策略损失
    L_val = mean( (V - R_hat)^2 )                           # 价值损失
    L_ent = -mean( entropy(pi) )                            # 熵正则（鼓励探索）
"""

import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MLPPolicy:
    """两层 MLP 策略网络（纯 numpy）。state_dim -> hidden -> [n_actions, 1]。"""

    def __init__(self, state_dim, n_actions, hidden=64, seed=0,
                 separate_critic=False):
        rng = np.random.RandomState(seed)
        self.sd, self.na, self.h = state_dim, n_actions, hidden
        self.separate_critic = separate_critic
        # 权重初始化：Xavier / He 风格
        self.W1 = rng.randn(state_dim, hidden) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden)
        self.W2p = rng.randn(hidden, n_actions) * np.sqrt(2.0 / hidden)
        self.b2p = np.zeros(n_actions)
        if separate_critic:
            # 独立 Critic：自己的隐藏层与价值头，完全不受策略更新影响
            self.W1v = rng.randn(state_dim, hidden) * np.sqrt(2.0 / state_dim)
            self.b1v = np.zeros(hidden)
            self.W2v = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
            self.b2v = np.zeros(1)
        else:
            self.W2v = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
            self.b2v = np.zeros(1)

    # ---------------- 前向 ----------------
    def forward(self, x):
        """x: (N, state_dim) -> (logits, value)。value 形状 (N,)"""
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2p + self.b2p
        if self.separate_critic:
            hv = np.tanh(x @ self.W1v + self.b1v)
            value = (hv @ self.W2v + self.b2v).squeeze(-1)
        else:
            value = (h @ self.W2v + self.b2v).squeeze(-1)
        return logits, value, h

    def act(self, x, eps=0.0):
        """给定状态，返回 (动作, 对数概率, 价值, 概率分布)。
        输入为单行向量时返回 Python 标量，方便直接喂给 env.step。

        eps>0 时使用 epsilon-greedy 行为策略：以 eps 概率随机选动作，
        以 1-eps 概率按策略 π 选。返回的 logp 是行为策略 b 下的对数概率
        （b(a) = (1-eps)·π(a) + eps/n），这样 PPO 的重要性采样比
        ρ = π_new(a)/b(a) 数学上仍然正确。
        这能防止训练早期策略塌缩成确定性策略（梯度消失、永不突破）。
        """
        single = x.ndim == 1 if isinstance(x, np.ndarray) else True
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        logits, value, _ = self.forward(x)
        probs = softmax(logits)
        if eps and eps > 0:
            # 行为策略 b(a) = (1-eps)·π(a) + eps/n，采样用它
            behavior = (1 - eps) * probs + eps / self.na
            actions = np.array([np.random.choice(self.na, p=pr) for pr in behavior])
            logp = np.log(np.clip(behavior[np.arange(len(actions)), actions], 1e-12, 1.0))
        else:
            actions = np.array([np.random.choice(self.na, p=pr) for pr in probs])
            logp = np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-12, 1.0))
        if single:
            return int(actions[0]), float(logp[0]), float(value[0]), probs[0]
        return actions, logp, value, probs

    def logp_actions(self, x, actions):
        logits, _, _ = self.forward(np.asarray(x, dtype=np.float32))
        probs = softmax(logits)
        return np.log(np.clip(probs[np.arange(len(actions)), actions], 1e-12, 1.0))

    # ---------------- 保存 / 加载 ----------------
    def parameters(self):
        if self.separate_critic:
            return [self.W1, self.b1, self.W2p, self.b2p,
                    self.W1v, self.b1v, self.W2v, self.b2v]
        return [self.W1, self.b1, self.W2p, self.b2p, self.W2v, self.b2v]

    def save(self, path):
        """保存为通用 JSON 格式（Python 与 JS 都能直接读，无需中间转换）。"""
        import json

        d = dict(
            format="ppo-snake-weights",
            version=2,
            state_dim=self.sd,
            hidden=self.h,
            n_actions=self.na,
            separate_critic=self.separate_critic,
            W1=self.W1.tolist(), b1=self.b1.tolist(),
            W2p=self.W2p.tolist(), b2p=self.b2p.tolist(),
            W2v=self.W2v.tolist(), b2v=self.b2v.tolist(),
        )
        if self.separate_critic:
            d["W1v"] = self.W1v.tolist()
            d["b1v"] = self.b1v.tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def load(self, path):
        """从 JSON（或兼容的 npz）加载权重。"""
        import json

        if path.endswith(".npz"):
            # 兼容旧版 npz 权重
            d = np.load(path)
            self.W1, self.b1 = d["W1"], d["b1"]
            self.W2p, self.b2p = d["W2p"], d["b2p"]
            self.W2v, self.b2v = d["W2v"], d["b2v"]
            if self.separate_critic:
                if "W1v" in d:
                    self.W1v, self.b1v = d["W1v"], d["b1v"]
                else:
                    self.W1v = self.W1.copy()
                    self.b1v = self.b1.copy()
            return

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.sd = d.get("state_dim", self.sd)
        self.h = d.get("hidden", self.h)
        self.na = d.get("n_actions", self.na)
        self.separate_critic = d.get("separate_critic", self.separate_critic)
        self.W1 = np.array(d["W1"], dtype=np.float32)
        self.b1 = np.array(d["b1"], dtype=np.float32)
        self.W2p = np.array(d["W2p"], dtype=np.float32)
        self.b2p = np.array(d["b2p"], dtype=np.float32)
        self.W2v = np.array(d["W2v"], dtype=np.float32)
        self.b2v = np.array(d["b2v"], dtype=np.float32)
        if self.separate_critic:
            self.W1v = np.array(d.get("W1v", d["W1"]), dtype=np.float32)
            self.b1v = np.array(d.get("b1v", d["b1"]), dtype=np.float32)


class AdamOptimizer:
    """Adam 优化器（作用于一组参数张量）。"""

    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            mhat = self.m[i] / (1 - self.beta1 ** self.t)
            vhat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def ppo_update(net, optimizer, states, actions, old_logp, advantages, returns,
               clip_eps=0.2, ent_coef=0.01, val_coef=0.5, epochs=10,
               minibatch=256):
    """
    在收集好的经验上做多轮小批量 PPO 更新。

    参数：
      states/actions/old_logp : 采样时记录的数据
      advantages             : GAE 优势估计（必要时已归一化）
      returns                : 折扣回报（价值网络回归目标）

    返回：
      (policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl)
    """
    n = len(states)
    pol_losses, val_losses, ent_losses, clip_fracs, kl_acc = [], [], [], [], 0.0
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            stats = _compute_update(net, optimizer,
                                    states[idx], actions[idx], old_logp[idx],
                                    advantages[idx], returns[idx],
                                    clip_eps, ent_coef, val_coef)
            pol_losses.append(stats[1]); val_losses.append(stats[2])
            ent_losses.append(stats[3]); clip_fracs.append(stats[4])
            kl_acc += stats[5]
    return (float(np.mean(pol_losses)), float(np.mean(val_losses)),
            float(np.mean(ent_losses)), float(np.mean(clip_fracs)),
            float(kl_acc / max(len(pol_losses), 1)))


def _compute_update(net, opt, S, A, old_logp, adv, ret,
                    clip_eps, ent_coef, val_coef):
    """单个小批量的前向 + 手写反向传播 + 参数更新。返回统计量。"""
    B = len(S)

    # ---------- 前向 ----------
    h = np.tanh(S @ net.W1 + net.b1)                      # (B, h)
    logits = h @ net.W2p + net.b2p                         # (B, na)
    if net.separate_critic:
        hv = np.tanh(S @ net.W1v + net.b1v)
        value = (hv @ net.W2v + net.b2v).squeeze(-1)       # (B,)
    else:
        value = (h @ net.W2v + net.b2v).squeeze(-1)        # (B,)
    probs = softmax(logits)                                # (B, na)
    logp = np.log(np.clip(probs[np.arange(B), A], 1e-12, 1.0))

    # ---------- 重要性采样比率 + clip ----------
    rho = np.exp(logp - old_logp)                          # (B,)
    rho_clipped = np.clip(rho, 1 - clip_eps, 1 + clip_eps)
    # PPO 策略目标：min(rho*A, clip(rho)*A)
    p_obj = np.minimum(rho * adv, rho_clipped * adv)       # (B,)
    pol_loss = -p_obj.mean()

    # ---------- 熵（鼓励探索） ----------
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)  # (B,)
    ent_loss = -entropy.mean()

    # ---------- 价值损失 ----------
    val_loss = ((value - ret) ** 2).mean()

    # ---------- 反向传播 ----------
    # 精确做法：d min(rhoA, clipA)/d rho 在 A>0 时取 clip 分支的 rho；在 A<0 时相反。
    use_clip = ((adv > 0) & (rho > 1 + clip_eps)) | ((adv < 0) & (rho < 1 - clip_eps))
    eff_ratio = np.where(use_clip, rho_clipped, rho)

    onehot = np.zeros_like(probs)
    onehot[np.arange(B), A] = 1.0
    g_logits = eff_ratio[:, None] * adv[:, None] * (probs - onehot) / B   # (B, na)

    # 熵项对 logits 的梯度：d(-H)/dz_j = p_j * (H + log p_j)
    g_ent = ent_coef * probs * (entropy[:, None] + np.log(np.clip(probs, 1e-12, 1.0))) / B

    # 价值项
    g_val = val_coef * 2.0 * (value - ret) / B                            # (B,)

    # 汇总 logits 梯度，并向上传播到共享隐藏层
    g_logits_all = g_logits + g_ent
    if net.separate_critic:
        # 独立 Critic：价值梯度只回传自己的隐藏层
        g_hv = g_val[:, None] * net.W2v.T                                 # (B, h)
        g_hv *= (1 - hv * hv)
        g_h = g_logits_all @ net.W2p.T                                    # (B, h)
    else:
        g_h = g_logits_all @ net.W2p.T + g_val[:, None] * net.W2v.T       # (B, h)

    # 隐藏层内部梯度（tanh' = 1 - h^2）
    g_h *= (1 - h * h)

    # 各参数梯度
    g_W2p = h.T @ g_logits_all
    g_b2p = g_logits_all.sum(axis=0)
    if net.separate_critic:
        g_W2v = hv.T @ (g_val[:, None])
        g_b2v = g_val.sum()
        g_W1v = S.T @ g_hv
        g_b1v = g_hv.sum(axis=0)
    else:
        g_W2v = h.T @ (g_val[:, None])
        g_b2v = g_val.sum()
    g_W1 = S.T @ g_h
    g_b1 = g_h.sum(axis=0)

    # ---------- Adam 更新 ----------
    if net.separate_critic:
        opt.step([g_W1, g_b1, g_W2p, g_b2p, g_W1v, g_b1v, g_W2v, g_b2v])
    else:
        opt.step([g_W1, g_b1, g_W2p, g_b2p, g_W2v, g_b2v])

    # 统计
    clip_frac = ((rho > 1 + clip_eps) | (rho < 1 - clip_eps)).mean()
    approx_kl = ((rho - 1) - np.log(rho)).mean()
    return (0.0, pol_loss, val_loss, ent_loss, clip_frac, approx_kl)


def compute_gae(rewards, values, dones, next_values, gamma=0.99, lam=0.95):
    """
    计算 GAE 优势与折扣回报。
      rewards / values / dones / next_values : 长度一致的数组
      done = 1 表示该步为回合终点（next_value 应视为 0）
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def bc_update(net, opt, states, actions, returns=None, label_smooth=0.1, val_coef=0.3):
    """
    行为克隆更新：监督学习，让策略模仿专家动作（带标签平滑的交叉熵），
    同时用折扣回报训练 Critic（价值头），为 PPO 的 GAE 提供可靠的 V(s)。

    目标分布 t(a) = (1-s)*onehot + s/n（s 为平滑系数），
    避免策略对某个动作过度自信——否则 PPO 微调时 π(a)≈1，梯度消失无法改进。

    对 logits 的梯度是 (probs - t)，直接复用手写反向传播。
    """
    B = len(states)
    na = net.na
    h = np.tanh(states @ net.W1 + net.b1)
    logits = h @ net.W2p + net.b2p
    probs = softmax(logits)
    logp = np.log(np.clip(probs[np.arange(B), actions], 1e-12, 1.0))
    # 带平滑的交叉熵损失：只对专家动作项贡献（其余项在梯度中体现）
    loss = -((1 - label_smooth) * logp).mean()

    target = np.full_like(probs, label_smooth / na)
    target[np.arange(B), actions] += 1.0 - label_smooth
    g_logits = (probs - target) / B          # (B, na)
    g_h = g_logits @ net.W2p.T

    # 价值头回归（可选）：V(s) -> returns
    if returns is not None:
        returns = np.asarray(returns, dtype=np.float32)
        if net.separate_critic:
            hv = np.tanh(states @ net.W1v + net.b1v)
            value = (hv @ net.W2v + net.b2v).squeeze(-1)
        else:
            value = (h @ net.W2v + net.b2v).squeeze(-1)
        val_loss = ((value - returns) ** 2).mean()
        g_val = val_coef * 2.0 * (value - returns) / B
        if net.separate_critic:
            g_hv = g_val[:, None] * net.W2v.T * (1 - hv * hv)
            g_W1v = states.T @ g_hv
            g_b1v = g_hv.sum(axis=0)
            g_W2v = hv.T @ (g_val[:, None])
            g_b2v = g_val.sum()
            g_h = g_h  # 策略梯度不混合价值梯度
        else:
            g_h = g_h + g_val[:, None] * net.W2v.T
            g_W2v = h.T @ (g_val[:, None])
            g_b2v = g_val.sum()
    else:
        val_loss = 0.0
        g_W2v = np.zeros_like(net.W2v)
        g_b2v = np.zeros_like(net.b2v)

    g_h *= (1 - h * h)
    g_W2p = h.T @ g_logits
    g_b2p = g_logits.sum(axis=0)
    g_W1 = states.T @ g_h
    g_b1 = g_h.sum(axis=0)
    if net.separate_critic:
        opt.step([g_W1, g_b1, g_W2p, g_b2p, g_W1v, g_b1v, g_W2v, g_b2v])
    else:
        opt.step([g_W1, g_b1, g_W2p, g_b2p, g_W2v, g_b2v])
    return float(loss)
