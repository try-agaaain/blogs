"""把训练产物（策略权重 + 训练曲线数据）同步进讲解页。

讲解页在 file:// 协议下用浏览器直接打开，无法 fetch 本地 JSON，
所以训练结束后把结果「烧录」进 HTML：

  * AI_MODEL    —— 策略网络权重（W1/b1/W2p/b2p），供「AI 自动玩」实时推理
  * TRAIN_DATA  —— 训练曲线数据（从 logs/full_curve.csv 提取），供学习曲线绘图

v1/v2 页面只嵌入 AI_MODEL；v3 页面（PPO贪吃蛇讲解v3.html）同时嵌入两者。

用法（在项目根运行）：
    python scripts/embed_model_to_html.py
    python scripts/embed_model_to_html.py --html PPO贪吃蛇讲解v3.html
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
MODEL_JSON = os.path.join(ROOT, "checkpoint", "best_snake.json")
CURVE_CSV = os.path.join(ROOT, "logs", "full_curve.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--html", default=None,
                    help="指定单个讲解页路径（默认同时同步所有讲解页）")
args = parser.parse_args()

HTMLS = ([args.html] if args.html else [
    os.path.join(ROOT, "PPO贪吃蛇讲解.html"),
    os.path.join(ROOT, "PPO贪吃蛇讲解v2.html"),
    os.path.join(ROOT, "PPO贪吃蛇讲解v3.html"),
])
if not os.path.exists(MODEL_JSON):
    raise SystemExit(f"找不到模型 {MODEL_JSON}，请先运行 python scripts/train.py")

with open(MODEL_JSON, encoding="utf-8") as f:
    d = json.load(f)

# 只提取策略网络部分（价值头仅训练用，浏览器推理用不到）
ai_model = {
    "h": d["hidden"],
    "na": d["n_actions"],
    "W1": d["W1"],
    "b1": d["b1"],
    "W2p": d["W2p"],
    "b2p": d["b2p"],
}

def to_js(obj):
    return json.dumps(obj, separators=(",", ":"))

def load_curve():
    """把 full_curve.csv 提取为 v3 页面可绘图的 JS 对象。"""
    if not os.path.exists(CURVE_CSV):
        return None
    iters, scores, foods, lens, kl = [], [], [], [], []
    with open(CURVE_CSV, encoding="utf-8") as f:
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
    return {"iters": iters, "scores": scores, "foods": foods,
            "lens": lens, "kl": kl}

curve = load_curve()

def embed_v3(html, ai_model, curve):
    """v3 页面：替换 AI_MODEL / TRAIN_DATA 占位符。"""
    if "var TRAIN_DATA" in html:
        html = re.sub(r"var AI_MODEL = [^;]*;",
                      "var AI_MODEL = " + to_js(ai_model) + ";",
                      html, count=1)
        if curve is not None:
            html = re.sub(r"var TRAIN_DATA = [^;]*;",
                          "var TRAIN_DATA = " + to_js(curve) + ";",
                          html, count=1)
    else:
        # 旧页面兜底：找不到 TRAIN_DATA 占位符时按旧逻辑处理
        script = (
            "<script>/* 浏览器内实时运行的策略网络权重（由 scripts/train.py 训练结束后"
            "自动从 checkpoint/best_snake.json 内嵌，供「AI 自动玩」现场决策使用） */\n"
            "var AI_MODEL = " + to_js(ai_model) + ";\n"
            "</script>"
        )
        pattern = re.compile(
            r'<script>/\* 浏览器内实时运行的策略网络权重[\s\S]*?</script>', re.MULTILINE)
        if pattern.search(html):
            html = pattern.sub(lambda m: script, html, count=1)
        else:
            anchor = "<script>\n"
            assert anchor in html, f"{html_path} 里找不到 <script> 锚点"
            html = html.replace(anchor, script + "\n" + anchor, 1)
    return html

for html_path in HTMLS:
    if not os.path.exists(html_path):
        print(f"跳过（不存在）：{os.path.basename(html_path)}")
        continue

    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    html = embed_v3(html, ai_model, curve)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    size = (len(to_js(ai_model)) + (len(to_js(curve)) if curve else 0)) // 1024
    print(f"已同步 {os.path.basename(html_path)}（约 {size} KB）")
