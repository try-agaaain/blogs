"""把 checkpoint/best_snake.json 的策略权重嵌入讲解页。

策略网络权重（W1/b1/W2p/b2p）以通用 JSON 格式保存，Python 和 JS 都能直接读；
本脚本只负责把其中的「策略部分」提取出来，内嵌为页面的 var AI_MODEL，
供「AI 自动玩」在浏览器里实时推理（每次点击都是一局新的实时对局）。

用法（在项目根运行）：
    python scripts/embed_model_to_html.py
默认同步两个讲解页：PPO贪吃蛇讲解.html 与 PPO贪吃蛇讲解v2.html
可选：
    --html <讲解页路径>   # 只同步指定页面
"""
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--html", default=None,
                    help="指定单个讲解页路径（默认同时同步两个讲解页）")
args = parser.parse_args()

HTMLS = ([args.html] if args.html else [
    os.path.join(ROOT, "PPO贪吃蛇讲解.html"),
    os.path.join(ROOT, "PPO贪吃蛇讲解v2.html"),
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

script = (
    "<script>/* 浏览器内实时运行的策略网络权重（由 scripts/train.py 训练结束后"
    "自动从 checkpoint/best_snake.json 内嵌，供「AI 自动玩」现场决策使用） */\n"
    "var AI_MODEL = " + to_js(ai_model) + ";\n"
    "</script>"
)

for HTML in HTMLS:
    if not os.path.exists(HTML):
        raise SystemExit(f"找不到 {HTML}，请用 --html 显式指定")

    with open(HTML, encoding="utf-8") as f:
        html = f.read()

    # 替换已有的 AI_MODEL 块；没有则插到第一个内联 <script> 之前
    pattern = re.compile(
        r'<script>/\* 浏览器内实时运行的策略网络权重[\s\S]*?</script>', re.MULTILINE)
    if pattern.search(html):
        html = pattern.sub(lambda m: script, html, count=1)
    else:
        anchor = "<script>\n"
        assert anchor in html, f"{HTML} 里找不到 <script> 锚点"
        html = html.replace(anchor, script + "\n" + anchor, 1)

    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"已把 AI_MODEL（{len(script)//1024} KB）嵌入 {os.path.basename(HTML)}")
