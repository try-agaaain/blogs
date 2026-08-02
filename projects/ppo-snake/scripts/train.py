"""
训练入口：用 PPO 教 AI 玩贪吃蛇

运行：
    python train.py                    # 默认参数训练
    python train.py --iters 300 --eval-every 10
    python train.py --seed 42 --save-model best.json

输出：
    * 每 eval-every 轮打印评估分数
    * 训练日志写入 logs/train_log.csv（供 HTML 页面绘制学习曲线）
    * 模型保存到 checkpoint/ppo_snake.json（-1 轮）与 best.json（最优）

稳定训练三件套（推荐，能显著消除种子敏感性）：
    python train.py --pretrain --separate-critic --lr 3e-4 --iters 500
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np

# 保证从项目任意位置运行都能导入 src 包
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.snake_game import SnakeEnv, STATE_DIM
from src.ppo import MLPPolicy, AdamOptimizer, ppo_update, compute_gae

# Windows 控制台可能默认 GBK，统一切到 UTF-8 避免中文打印崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ------------------------------------------------------------ 采样
def collect_rollout(net, envs, max_steps=150, eps=0.0):
    """
    向量化采样：n_envs 个环境同步跑 max_steps 步。

    数据布局：第 t 帧第 i 个环境展开后的下标为 t*n_envs + i。
    next_value 取"同环境下一帧的 value"；最后一帧用网络对 rollout
    结束后真实状态的估值自引导（bootstrap）。done 处由 GAE 的
    (1-done) 因子自动截断，next_value 值无关紧要。

    eps>0 时采样用 epsilon-greedy 行为策略（见 MLPPolicy.act），
    防止训练早期策略塌缩、无法探索到食物。
    """
    n_envs = len(envs)
    states = np.stack([e.reset() for e in envs]).astype(np.float32)

    all_states, all_actions, all_logp, all_values = [], [], [], []
    all_rewards, all_dones = [], []

    for _ in range(max_steps):
        actions, logp, values, _ = net.act(states, eps=eps)
        all_states.append(states)
        all_actions.append(actions)
        all_logp.append(logp)
        all_values.append(values)

        next_states = []
        for i, env in enumerate(envs):
            ns, r, done, _ = env.step(int(actions[i]))
            if done:
                ns = env.reset()
            next_states.append(ns)
            all_rewards.append(r)
            all_dones.append(1.0 if done else 0.0)
        states = np.stack(next_states).astype(np.float32)
    final_states = states   # rollout 结束后每个 env 的真实当前状态

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    old_logp = np.concatenate(all_logp)
    values = np.concatenate(all_values)
    rewards = np.array(all_rewards, dtype=np.float64)
    dones = np.array(all_dones, dtype=np.float64)

    # next_value：同环境下一帧的 value；最后一帧用最终状态估值自引导
    next_values = np.zeros_like(values)
    if len(values) > n_envs:
        next_values[:-n_envs] = values[n_envs:]
    tail_actions, tail_logp, tail_v, _ = net.act(final_states, eps=eps)
    next_values[-n_envs:] = tail_v

    return states, actions, old_logp, rewards, dones, values, next_values


# ------------------------------------------------------------ 评估
def evaluate(net, n_episodes=24, seed=None):
    """跑 n_episodes 局，返回平均分数 / 平均食物数 / 平均存活步数。"""
    total_r = total_food = total_len = 0
    for i in range(n_episodes):
        env = SnakeEnv(seed=seed)
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a, _, _, _ = net.act(s)
            s, r, done, info = env.step(a)
            ep_r += r
        total_r += ep_r
        total_food += info["food_eaten"]
        total_len += info["length"]
    n = n_episodes
    return total_r / n, total_food / n, total_len / n


# ------------------------------------------------------------ 同步讲解页
def embed_weights_to_html(model_json, html_path, curve_csv="logs/full_curve.csv"):
    """把训练产物内嵌进讲解页。

    AI_MODEL：best_snake.json 的策略权重，供「AI 自动玩」实时推理。
    TRAIN_DATA：logs/full_curve.csv 提取的训练曲线数据，供学习曲线绘图。
    v3 页面同时嵌入两者；v1/v2 页面只嵌入权重（兼容旧逻辑）。
    """
    with open(model_json, encoding="utf-8") as f:
        d = json.load(f)

    def to_js(obj):
        return json.dumps(obj, separators=(",", ":"))

    # 只提取策略网络部分（价值头仅训练用，浏览器推理用不到）
    ai_model = {
        "h": d["hidden"],
        "na": d["n_actions"],
        "W1": d["W1"],
        "b1": d["b1"],
        "W2p": d["W2p"],
        "b2p": d["b2p"],
    }

    curve = None
    if os.path.exists(curve_csv):
        iters, scores, foods, lens, kl = [], [], [], [], []
        with open(curve_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if row["score"] != "nan":
                        iters.append(int(row["iter"]))
                        scores.append(float(row["score"]))
                        foods.append(float(row["food"]))
                        lens.append(float(row["len"]))
                    kl.append(float(row["kl"]))
                except (KeyError, ValueError):
                    continue
        curve = {"iters": iters, "scores": scores, "foods": foods,
                 "lens": lens, "kl": kl}

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    if "var TRAIN_DATA" in html:
        # v3 页面：替换占位符
        html = re.sub(r"var AI_MODEL = [^;]*;",
                      "var AI_MODEL = " + to_js(ai_model) + ";",
                      html, count=1)
        if curve is not None:
            html = re.sub(r"var TRAIN_DATA = [^;]*;",
                          "var TRAIN_DATA = " + to_js(curve) + ";",
                          html, count=1)
    else:
        # v1/v2 页面：替换 AI_MODEL 块（兼容旧逻辑）
        script = (
            "<script>/* 浏览器内实时运行的策略网络权重（由 scripts/train.py 训练结束后"
            "自动从 checkpoint/best_snake.json 内嵌，供「AI 自动玩」现场决策使用） */\n"
            "var AI_MODEL = " + to_js(ai_model) + ";\n"
            "</script>"
        )
        pattern = re.compile(
            r"<script>/\* 浏览器内实时运行的策略网络权重[\s\S]*?</script>", re.MULTILINE)
        if pattern.search(html):
            html = pattern.sub(lambda m: script, html, count=1)
        else:
            anchor = "<script>\n"
            assert anchor in html, f"{html_path} 里找不到 <script> 锚点"
            html = html.replace(anchor, script + "\n" + anchor, 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已同步 {os.path.basename(html_path)}（AI_MODEL + TRAIN_DATA）")


# ------------------------------------------------------------ 主流程
def main():
    parser = argparse.ArgumentParser(description="PPO 训练贪吃蛇")
    parser.add_argument("--iters", type=int, default=800, help="训练轮数")
    parser.add_argument("--eval-every", type=int, default=10, help="每 N 轮评估一次")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--seeds", type=int, default=1,
                        help="并行跑多少个随机种子，取最优结果（应对RL的随机性）")
    parser.add_argument("--n-envs", type=int, default=48, help="并行环境数")
    parser.add_argument("--max-steps", type=int, default=200, help="每轮 rollout 步数")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="初始 epsilon-greedy 探索率（实验表明 >0 会破坏收敛，默认关闭）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率（小步慢走更稳定）")
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    parser.add_argument("--epochs", type=int, default=6, help="PPO 内层 epoch 数")
    parser.add_argument("--ent-coef", type=float, default=0.08, help="熵正则系数")
    parser.add_argument("--val-coef", type=float, default=0.3, help="价值损失系数")
    parser.add_argument("--separate-critic", action="store_true",
                        help="独立价值头（不共享隐藏层），价值梯度不再干扰策略")
    parser.add_argument("--pretrain", action="store_true",
                        help="先做行为克隆预热（用启发式专家数据监督训练初始策略），"
                             "大幅消除种子敏感性")
    parser.add_argument("--pretrain-iters", type=int, default=40, help="预训练轮数")
    parser.add_argument("--save-model", type=str, default=None,
                        help="模型保存路径（默认 checkpoint/ppo_snake.json）")
    parser.add_argument("--no-embed", action="store_true",
                        help="训练结束后不把权重同步进 PPO贪吃蛇讲解.html")
    parser.add_argument("--embed-only", action="store_true",
                        help="只把 checkpoint/best_snake.json 同步进讲解页，不训练")
    args = parser.parse_args()

    html_paths = [
        os.path.join(_ROOT, "PPO贪吃蛇讲解.html"),
        os.path.join(_ROOT, "PPO贪吃蛇讲解v2.html"),
        os.path.join(_ROOT, "PPO贪吃蛇讲解v3.html"),
    ]

    # 仅同步模式：把现有模型权重烧录进讲解页后直接退出
    if args.embed_only:
        if not os.path.exists(best_model := "checkpoint/best_snake.json"):
            raise SystemExit(f"找不到 checkpoint/best_snake.json，请先训练")
        for hp in html_paths:
            embed_weights_to_html(best_model, hp)
        return

    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    model_path = args.save_model or "checkpoint/ppo_snake.json"
    best_path = "checkpoint/best_snake.json"

    print("=" * 64)
    print("PPO 训练贪吃蛇")
    print(f"  状态维度={STATE_DIM}  动作空间=3(直行/左转/右转)  网格=12x12")
    print(f"  并行环境={args.n_envs}  rollout步数={args.max_steps}  lr={args.lr}")
    print(f"  隐藏层={args.hidden}  PPO内层epoch={args.epochs}  熵正则={args.ent_coef}")
    print(f"  随机种子: {list(range(args.seed, args.seed + args.seeds))}")
    print("=" * 64)

    t_global = time.time()
    global_best = -float("inf")
    global_best_seed = None

    for si in range(args.seeds):
        seed = args.seed + si
        csv_path = f"logs/train_log_s{seed}.csv"
        seed_best_path = f"checkpoint/best_snake_s{seed}.json"
        t_start = time.time()

        np.random.seed(seed)
        net = MLPPolicy(STATE_DIM, 3, hidden=args.hidden, seed=seed,
                        separate_critic=args.separate_critic)
        opt = AdamOptimizer(net.parameters(), lr=args.lr)
        envs = [SnakeEnv() for _ in range(args.n_envs)]
        best_score = -float("inf")

        # 行为克隆预热：用启发式专家数据监督训练，消除种子敏感性
        if args.pretrain:
            from src.pretrain import collect_data, bc_update_wrapper
            print(f"[seed={seed}] 行为克隆预热 {args.pretrain_iters} 轮...")
            for _ in range(args.pretrain_iters):
                states, actions, returns = collect_data(400, 0.1, seed)
                bc_update_wrapper(net, opt, states, actions, returns)
            print(f"[seed={seed}] 预热完成，开始 PPO 微调")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iter", "avg_reward", "avg_food", "avg_length", "clip_frac", "policy_loss", "value_loss", "approx_kl"])

        for it in range(1, args.iters + 1):
            # 学习率、熵系数与探索率线性衰减（前 80% 线性衰减到 10%）
            lr_scale = max(0.1, 1.0 - 0.9 * (it / args.iters))
            opt.lr = args.lr * lr_scale
            ent_coef = args.ent_coef * lr_scale
            eps = args.eps * lr_scale   # epsilon-greedy 探索：前期保持探索，后期收敛

            # 采样
            states, actions, old_logp, rewards, dones, values, next_values = \
                collect_rollout(net, envs, max_steps=args.max_steps, eps=eps)

            # GAE 优势 + 回报
            advantages, returns = compute_gae(rewards, values, dones, next_values,
                                              gamma=0.99, lam=0.95)
            # 优势归一化 + 裁剪（稳定训练）
            adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
            advantages = np.clip((advantages - adv_mean) / adv_std, -3.0, 3.0)

            # PPO 更新
            pol_loss, val_loss, ent_loss, clip_frac, approx_kl = ppo_update(
                net, opt, states, actions, old_logp, advantages, returns,
                clip_eps=0.2, ent_coef=ent_coef, val_coef=args.val_coef,
                epochs=args.epochs, minibatch=512)

            # 定期评估
            if it % args.eval_every == 0 or it == 1:
                avg_r, avg_food, avg_len = evaluate(net)
                elapsed = time.time() - t_start
                print(f"[s{seed}|iter {it:4d}/{args.iters}] 得分 {avg_r:+7.2f} | "
                      f"食物 {avg_food:5.2f} | 长度 {avg_len:6.2f} | "
                      f"clip {clip_frac:.2f} | KL {approx_kl:.3f} | {elapsed:6.1f}s")

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([it, round(avg_r, 3), round(avg_food, 3),
                                     round(avg_len, 3), round(clip_frac, 3),
                                     round(pol_loss, 4), round(val_loss, 4),
                                     round(approx_kl, 4)])

                # 保存最优模型
                if avg_r > best_score:
                    best_score = avg_r
                    net.save(seed_best_path)

            # 定期 checkpoint
            if it % 50 == 0:
                net.save(model_path)

        print(f"\n[seed={seed}] 训练 {args.iters} 轮完成，耗时 {time.time()-t_start:.1f}s，最优得分 {best_score:+.2f}")
        if best_score > global_best:
            global_best = best_score
            global_best_seed = seed
            net.save(best_path)   # 只把最好的种子写进最佳模型

    print("=" * 64)
    print(f"全部 {args.seeds} 个种子训练完成，总耗时 {time.time()-t_global:.1f}s")
    print(f"最优种子 seed={global_best_seed}，最优平均得分 = {global_best:+.2f}")
    print(f"最佳模型已保存: {best_path}")
    print(f"训练日志:   logs/train_log_s*.csv（每个种子一份）")

    # 把最新权重同步进讲解页（除非 --no-embed）
    if args.no_embed:
        print("已跳过讲解页同步（--no-embed）。")
        print(f"如需手动同步，可运行:\n"
              f"    python scripts/train.py --embed-only   # 只同步讲解页，不训练")
    elif os.path.exists(best_path):
        for hp in html_paths:
            embed_weights_to_html(best_path, hp)


if __name__ == "__main__":
    main()
