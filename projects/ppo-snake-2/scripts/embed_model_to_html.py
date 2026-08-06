"""把训练产物（策略权重 + 训练曲线数据）同步进讲解页 PPO贪吃蛇讲解.html。

讲解页在 file:// 协议下用浏览器直接打开，无法 fetch 本地 JSON，
所以训练结束后把结果「烧录」进 HTML：

  * AI_MODEL    —— BottleneckPolicy 网络权重（in_proj + blocks + actor），
                   供「AI 自动玩」实时推理
  * TRAIN_DATA  —— 训练曲线数据（从 experiments/s6/train_log.csv 提取），
                   供学习曲线绘图

用法（在项目根运行）：
    python scripts/embed_model_to_html.py
    python scripts/embed_model_to_html.py --model experiments/s6/best_snake.json
    python scripts/embed_model_to_html.py --csv experiments/s6/train_log.csv
"""
import argparse
import csv
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HTML_PATH = os.path.join(ROOT, "PPO贪吃蛇讲解.html")

parser = argparse.ArgumentParser()
parser.add_argument("--html", default=HTML_PATH, help="讲解页路径")
parser.add_argument("--model", default=None,
                    help="权重 JSON 路径（默认取项目根 best_snake.json）")
parser.add_argument("--csv", default=None,
                    help="训练曲线 CSV 路径（默认取 experiments/s6/train_log.csv）")
args = parser.parse_args()

model_json = args.model or os.path.join(ROOT, "best_snake.json")
csv_path = args.csv or os.path.join(ROOT, "experiments", "s6", "train_log.csv")

with open(model_json, encoding="utf-8") as f:
    d = json.load(f)

# 只提取策略网络部分（Critic 价值头仅训练用，浏览器推理用不到）
w = d["weights"]
ai_model = {
    "format": d["format"],
    "sd": d["state_dim"],
    "na": d["n_actions"],
    "h": d["hidden"],
    "nb": d["n_blocks"],
    "w": {
        "ip_w": w["in_proj.weight"],
        "ip_b": w["in_proj.bias"],
        "blk": [
            {
                "dn_w": w[f"blocks.{i}.down.weight"],
                "dn_b": w[f"blocks.{i}.down.bias"],
                "gt_w": w[f"blocks.{i}.gate.weight"],
                "gt_b": w[f"blocks.{i}.gate.bias"],
                "up_w": w[f"blocks.{i}.up.weight"],
                "up_b": w[f"blocks.{i}.up.bias"],
            }
            for i in range(d["n_blocks"])
        ],
        "ac_w": w["actor.weight"],
        "ac_b": w["actor.bias"],
    },
}

# 压缩数值：保留 6 位小数（权重量级 ~1e0，float32 精度足够）
def shrink(obj):
    if isinstance(obj, list):
        return [shrink(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


ai_model["w"] = shrink(ai_model["w"])


def to_js(obj):
    return json.dumps(obj, separators=(",", ":"))


def load_curve():
    """把 train_log.csv 提取为可绘图的 JS 对象。"""
    iters, scores, foods, lens, clip, kl = [], [], [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                iters.append(int(row["iter"]))
                scores.append(float(row["avg_reward"]))
                foods.append(float(row["avg_food"]))
                lens.append(float(row["avg_length"]))
                clip.append(float(row["clip_frac"]))
                kl.append(float(row["approx_kl"]))
            except (KeyError, ValueError):
                continue
    return {"iters": iters, "scores": scores, "foods": foods,
            "lens": lens, "clip": clip, "kl": kl}


curve = load_curve()

with open(args.html, encoding="utf-8") as f:
    html = f.read()

assert "var AI_MODEL" in html, f"{args.html} 里找不到 AI_MODEL 占位符"
html = re.sub(r"var AI_MODEL = [^;]*;",
              "var AI_MODEL = " + to_js(ai_model) + ";",
              html, count=1)
html = re.sub(r"var TRAIN_DATA = [^;]*;",
              "var TRAIN_DATA = " + to_js(curve) + ";",
              html, count=1)

with open(args.html, "w", encoding="utf-8") as f:
    f.write(html)

size = (len(to_js(ai_model)) + len(to_js(curve))) // 1024
print(f"已同步 {os.path.basename(args.html)}（数据约 {size} KB）")
print(f"  模型: hidden={d['hidden']} blocks={d['n_blocks']}  "
      f"曲线点: {len(curve['iters'])}")
