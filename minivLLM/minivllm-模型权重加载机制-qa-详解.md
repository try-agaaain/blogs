# MinivLLM 模型权重加载机制 · Q&A 详解

> 本文以问答形式，系统梳理 MinivLLM（一个从零实现的 vLLM 推理引擎）如何自定义模型网络结构，并加载 HuggingFace 预训练权重。内容覆盖命名规范、加载流程、张量并行分片、以及新模型适配方法。

---

## Q1 · MinivLLM 是自己实现了大模型的网络结构吗？如果不是，它是如何加载权重的？

**两者兼有。** MinivLLM 用纯 PyTorch 从零实现了 LLaMA / Qwen3 的模型架构，但**权重来自 HuggingFace 预训练 checkpoint**。

**网络结构方面**：`src/myvllm/models/` 和 `src/myvllm/layers/` 下全是手写的 PyTorch 模块，不依赖 `transformers` 中的任何模型类。具体包括：

| 文件 | 内容 |
|---|---|
| `models/llama.py` | LlamaModel、LlamaDecoderLayer、LlamaAttn、LlamaMLP、LlamaForCausalLM |
| `models/qwen3.py` | Qwen3Model、Qwen3DecoderLayer、Qwen3Attention、Qwen3MLP、Qwen3ForCausalLM |
| `layers/linear.py` | ColumnParallelLinear、RowParallelLinear、MergedColumnParallelLinear、QKVColumnParallelLinear（支持 Tensor Parallelism） |
| `layers/attention.py` | 手写 Triton Kernel 实现的 Flash Attention（prefill）和 Paged Attention（decode） |
| `layers/embedding_head.py` | VocabParallelEmbedding、ParallelLMHead（支持 Vocab 维度的 TP） |
| `layers/rotary_embedding.py` | RoPE（含 LLaMA 3.2 特有实现） |
| `layers/layernorm.py` | RMSNorm |
| `layers/activation.py` | SiLU + 逐元素乘（SwiGLU 实现） |
| `layers/sampler.py` | 温度采样 |

**权重加载方面**：`src/myvllm/utils/loader.py` 读取 HuggingFace checkpoint 中的 `.safetensors` 文件，将张量按名称映射到自定义模型的参数上。命名一致则直接拷贝，命名不一致则手动拼接（如 QKV 合并、gate_up 合并），形状不完全匹配则按较小维度截断。

加载时只用到了 `transformers` 库的 `AutoConfig` 和 `safetensors` 的 `safe_open`，不需要加载 HF 的模型对象。

---

## Q2 · 开发者是怎么知道模型结构的？如果来了一个新模型（比如 Qwen3.6），能复现吗？

**模型结构的知识来源有两个：HuggingFace 的 `config.json` + 论文/源码中对架构的理解。**

具体来说：

**第一步，获取超参数。** 每个 HF 模型仓库的根目录都有一个 `config.json`，里面记录了全部架构参数：

```json
{
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "intermediate_size": 3072,
  "vocab_size": 151936,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": true,
  ...
}
```

MinivLLM 将这些值搬到 `main.py` 的 config 字典中（如 `main.py:23-41`）。注意 `AutoConfig` 虽然在 `loader.py` 中被 import 了，但并未用来**自动**解析参数——参数是人工查阅后硬编码的。

**第二步，理解架构模板。** 现代 LLaMA-style 模型的网络结构高度同质化。任意一层 Decoder Layer 都由相同的积木拼成：

```
Layer(x):
    residual = x
    x = RMSNorm(x)
    x = Attention(x)     ← RoPE(QKV投影) → Flash/Paged Attention → O投影
    x = x + residual
    residual = x
    x = RMSNorm(x)
    x = MLP(x)           ← gate_up → SiLU(gate) × up → down投影
    x = x + residual
    return x
```

All layers share the same shape — only count (`num_layers`) changes.

只要新模型没有引入全新的操作（比如某种尚未出现的注意力变体），只需**调整 config 参数**即可适配，模型代码本身几乎不需要改动。比如 Qwen3 → 假想的 Qwen3.6，大概率只改超参数值。

**第三步，确认命名规范。** 用几行代码读 `.safetensors` 文件的全部 key：

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for k in f.keys():
        print(k)
```

如果新模型的 key 格式与旧模型一致（大概率如此，因为都是 `transformers` 库的 `save_pretrained()` 生成的），loader 中的映射代码可以直接复用。

---

## Q3 · 模型权重的命名规范是什么样的？我自己写网络时如何让参数名与 checkpoint 对上？

**HuggingFace 的权重命名规范，本质上是 PyTorch 属性路径 + `transformers` 库保存时的约定。**

PyTorch 中 `nn.Module` 的参数名完全由 Python 属性名决定——你在 `__init__` 里写 `self.xxx = SomeModule()`，参数名中就会多一级 `.xxx`。嵌套层级即为路径深度。

HuggingFace 在 `save_pretrained()` 时同样遵循此规则。因此一个典型的 HF checkpoint 中的 key 长这样：

```
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.post_attention_layernorm.weight
...
model.norm.weight
lm_head.weight
```

**让你的模型参数名与之匹配的方法就是——用相同的属性名。** 看一看 MinivLLM 中各层是怎么命名的：

```python
# Qwen3ForCausalLM (qwen3.py:288)
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, ...):
        self.model = Qwen3Model(...)       # → "model.xxx"
        self.lm_head = ParallelLMHead(...) # → "lm_head.weight"

# Qwen3Model (qwen3.py:234)
class Qwen3Model(nn.Module):
    def __init__(self, ...):
        self.embed_tokens = VocabParallelEmbedding(...)  # → "model.embed_tokens.weight"
        self.layers = nn.ModuleList([...])               # → "model.layers.N.xxx"
        self.norm = LayerNorm(...)                       # → "model.norm.weight"

# Qwen3DecoderLayer (qwen3.py:162)
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, ...):
        self.input_layernorm = LayerNorm(...)            # → "model.layers.N.input_layernorm.weight"
        self.self_attn = Qwen3Attention(...)             # → "model.layers.N.self_attn.xxx"
        self.post_attention_layernorm = LayerNorm(...)   # → "model.layers.N.post_attention_layernorm.weight"
        self.mlp = Qwen3MLP(...)                         # → "model.layers.N.mlp.xxx"

# Qwen3MLP (qwen3.py:132)
class Qwen3MLP(nn.Module):
    def __init__(self, ...):
        self.gate_up = MergedColumnParallelLinear(...)   # → "model.layers.N.mlp.gate_up.weight"
        self.down_proj = RowParallelLinear(...)          # → "model.layers.N.mlp.down_proj.weight"
```

> 注意：`gate_up` 和 `qkv_projection` 这两个名字**与 HF 不一致**——HF 中分别叫 `gate_proj`/`up_proj` 和 `q_proj`/`k_proj`/`v_proj`。这是为优化推理效率有意为之的，loader 中专门写了映射逻辑来处理（详见 Q4 和 Q5）。

**实操建议**：在动手写模型代码之前，先用三行代码把 `.safetensors` 的全部 key 打印出来，贴在一个文本文件里对照。同时运行 `model.named_parameters()` 看你模型的输出。两边逐行比对，不一致的地方就是你要写映射的位置。

---

## Q4 · 给定一个权重名称（如 `model.embed_tokens.weight`），如何确定它对应的是什么模块？

**直接结论：从名字本身无法推断。** `model.embed_tokens.weight` 这个字符串只是键，不包含类型信息。

**正确的做法是反过来：先看 HF 源码中这个属性是什么模块，然后自己的实现保持一致即可。** 具体来说：

1. 打开 HF 模型对应的 GitHub 仓库（或 `transformers` 库源码），找到同名类（如 `Qwen3Model`），看 `__init__` 中 `self.embed_tokens` 赋的什么值——通常是 `nn.Embedding(vocab_size, hidden_size)`。

2. 在你的自定义实现中，这个位置也可以放 `nn.Embedding`，也可以放 `nn.Linear`，甚至可以放一个自己写的类——比如 MinivLLM 用的是 `VocabParallelEmbedding`（`embedding_head.py:12`），它能支持 Vocab 维度的 Tensor Parallelism。

**判定标准只有一个**：该模块的 `.weight` 参数形状必须与 HF checkpoint 中同名 key 的 tensor 形状一致。形状对了，前向计算的数学结果就对了（`nn.Embedding` 和 `nn.Linear` 在功能上都做的是"查表 / 矩阵乘"）。

```python
# HF checkpoint 中的形状
model.embed_tokens.weight → (vocab_size, hidden_size)   # 例如 (151936, 1024)

# MinivLLM 中 VocabParallelEmbedding 的 weight 形状
# 单卡时: (151936, 1024)  ← 完全匹配
# TP=2时:  (75968, 1024)   ← 每 GPU 一半，由 weight_loader 从完整权重中切片
```

同理，`model.layers.0.self_attn.o_proj.weight` 在 HF 中是 `nn.Linear` 的权重，在 MinivLLM 中包在 `RowParallelLinear` 里——但它的 `weight` 参数形状和名字都一致，加载就能成功。

---

## Q5 · QKV 和 gate_up 的合并加载究竟是怎么回事？能否详细讲讲完整流程？

这是 MinivLLM 最核心的优化之一。为了减少矩阵乘法次数，它将多个投影矩阵合并为一个，但 HF checkpoint 中它们是分开存储的。loader 需要感知这个差异并手动拼接。

### 以 QKV 合并为例的完整时序：

```
时间线 ———————————————————————————————————————————————————————————→

[步骤 1] 创建模型对象
────────────────────────────────
model = Qwen3ForCausalLM(vocab_size=151936, hidden_size=1024, ...)
↓
此时模型内所有参数都是 torch.empty() 创建的空张量，值无意义。
GPU 显存已分配好，但内容随机。

参数 qkv_projection.weight 的形状是 (head_dim × (num_heads + 2×num_kv_heads), hidden_size)
                              = (128 × (16 + 2×8), 1024)
                              = (4096, 1024)


[步骤 2] 读取 HF checkpoint 文件
────────────────────────────────
safetensor_files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

读入后得到字典 hf_weights:
{
    "model.layers.0.self_attn.q_proj.weight":  tensor(2048, 1024),  ← Q: 16 heads × 128
    "model.layers.0.self_attn.k_proj.weight":  tensor(1024, 1024),  ← K:  8 heads × 128
    "model.layers.0.self_attn.v_proj.weight":  tensor(1024, 1024),  ← V:  8 heads × 128
    "model.layers.0.self_attn.o_proj.weight":  tensor(1024, 2048),  ← O 投影与HF一致
    ...
}


[步骤 3] 遍历 hf_weights，逐条匹配规则
────────────────────────────────
当遍历到 "model.layers.0.self_attn.q_proj.weight" 时:

  (a) 识别出这是 q_proj
      → loader.py:76 的 if '.self_attn.q_proj.weight' in hf_name

  (b) 提取层号 "0"
      → 正则匹配 re.search(r'layers\.(\d+)', hf_name)

  (c) 去 hf_weights 中找到同层的 k_proj 和 v_proj
      → k_name = hf_name.replace('q_proj', 'k_proj')
      → v_name = hf_name.replace('q_proj', 'v_proj')

  (d) 沿第 0 维度拼接三个矩阵
      → qkv_weight = torch.cat([q, k, v], dim=0)
        形状: (2048, 1024) + (1024, 1024) + (1024, 1024) = (4096, 1024)

  (e) 用 get_parameter 定位到自定义模型中的参数，拷贝进去
      → param = model.get_parameter(
            "model.layers.0.self_attn.qkv_projection.weight"
        )
      → param.data.copy_(qkv_weight)
        形状 (4096, 1024) ← 与上面拼接的结果完美匹配


[步骤 4] 对于 o_proj、down_proj、input_layernorm 等名字一致的参数
────────────────────────────────
直接按名字拷贝，无额外处理:
  param = model.get_parameter(hf_name)
  param.data.copy_(hf_weight)
```

### gate_up 合并的逻辑完全类似

HF 文件中有 `gate_proj` 和 `up_proj` 两个矩阵，形状各为 `(3072, 1024)`。loader 将它们拼接成 `(6144, 1024)` 的一个矩阵，拷入 `model.layers.0.mlp.gate_up.weight`。

前向计算时（`activation.py:7-18`），用一个 `chunk(2, dim=-1)` 把拼接结果拆回两半，然后 `SiLU(gate) × up`：

```python
def forward(self, x):
    gate, up = x.chunk(2, dim=-1)   # 沿最后一维拆成两半
    return F.silu(gate) * up        # gate走激活，up走恒等，逐元素乘
```

**拼接维度必须和 `chunk` 的维度一致**，否则拆出来的就不是原来的 gate 和 up 了。

---

## Q6 · `main_llama32.py` 的 config 里只有参数值，没有权重名称，两者是如何对应的？

**权重名称由模型类的属性结构决定，config 参数只决定矩阵的形状。**

以 LLaMA-3.2-1B 为例，`main_llama32.py:29-44` 的 config 字典中每个参数的作用如下：

```python
config = {
    'vocab_size': 128256,          # → embed_tokens.weight 和 lm_head.weight 的第 0 维
    'hidden_size': 2048,           # → 所有 LayerNorm 的长度、线性层的列宽
    'head_dim': 64,                # → 单个注意力头的维度
    'num_qo_heads': 32,            # → Q 部分输出维度: 32 × 64 = 2048
    'num_kv_heads': 8,             # → K/V 部分输出维度:  8 × 64 = 512
    'intermediate_size': 8192,     # → MLP 中间宽度: gate_up 输出 = 2 × 8192 = 16384
    'num_layers': 16,              # → ModuleList 长度，决定有多少层
    'rms_norm_epsilon': 1e-5,      # → RMSNorm 的 ε
    'rope_base': 500000,           # → RoPE 的 base
    'max_position_embeddings': 32768,  # → RoPE 的最大位置
    'tie_word_embeddings': True,   # → lm_head.weight 是否与 embed_tokens.weight 共享
}
```

这些参数传入构造函数的路径是：

```
main_llama32.py:config
  → model_runner.py:52-67  LlamaForCausalLM(vocab_size=..., hidden_size=..., ...)
    → llama.py:285          self.model = LlamaModel(vocab_size=..., hidden_size=..., ...)
                            每个参数控制子模块的形状
    → llama.py:300          self.lm_head = ParallelLMHead(...)
```

可以画成一张映射表：

| config 参数 | 决定哪些参数 | 实例形状 |
|---|---|---|
| `vocab_size=128256` | `model.embed_tokens.weight` 第0维<br>`lm_head.weight` 第0维 | (128256, 2048) |
| `hidden_size=2048` | 所有 `*.layernorm.weight` 长度<br>所有线性层的 hidden 维度 | (2048,) / 对应位置 |
| `num_qo_heads=32, head_dim=64` | `qkv_projection.weight` 中 Q 子块 | (2048, 2048) |
| `num_kv_heads=8, head_dim=64` | `qkv_projection.weight` 中 K/V 子块 | (1024, 2048) |
| `intermediate_size=8192` | `gate_up.weight` 输出宽度<br>`down_proj.weight` 输入宽度 | (16384, 2048) / (2048, 8192) |
| `num_layers=16` | `ModuleList` 长度 | 16 层 |

**一句话比喻：config 是施工图纸（规定了墙的尺寸），构造函数是工头（按图砌墙并自动贴门牌号），loader 是快递员（按门牌号投递 HF 送来的包裹）。**

---

## Q7 · 什么是 Sharding（分片）？ColumnParallelLinear 在项目中用到吗？分片后的权重从哪来？

### Sharding 是什么

**Sharding 在此处特指 Tensor Parallelism（张量并行）时的权重切分。** 当有多张 GPU 时，把一个大矩阵沿某个维度切成多份，每张 GPU 只存一个分片。前向计算时各 GPU 算各自的片，必要时通过通信合并结果。

以 `ColumnParallelLinear` 为例——它沿**输出维度**切分：

```
HF checkpoint 中的完整权重:
    weight.shape = (4096, 2048)    ← 磁盘上存着完整的矩阵

TP=2 时，每张 GPU 只持有:
    GPU 0:  weight.shape = (2048, 2048) ← 完整权重的上半行
    GPU 1:  weight.shape = (2048, 2048) ← 完整权重的下半行
```

前向计算时，两张 GPU 各自做自己的矩阵乘法，各算一半的输出，不需要通信。这就是"列并行"的含义。

而 `RowParallelLinear` 沿**输入维度**切分，每个 GPU 得到完整的输出但只乘了部分输入。因此前向计算末尾需要 `dist.all_reduce` 把各 GPU 的部分结果求和。

### ColumnParallelLinear 在项目中使用情况

**基类 `ColumnParallelLinear` 本身从未被直接实例化，只有它的两个子类被使用：**

| 类名 | 使用位置 | 用途 |
|---|---|---|
| `QKVColumnParallelLinear` | `llama.py:33`, `qwen3.py:38` | 合并 Q/K/V 三个线性变换为一次矩阵乘法 |
| `MergedColumnParallelLinear` | `llama.py:121`, `qwen3.py:140` | 合并 gate/up 两个线性变换为一次矩阵乘法 |

当前 `main.py:20` 和 `main_llama32.py:24` 中 `world_size=1`，此时 TP 不生效：

```python
# linear.py:91, 当 tp_size = 1 时
super().__init__(input_size, output_size // 1, ...)
# 等价于
super().__init__(input_size, output_size, ...)
```

即退化为普通的线性层，形状与 HF checkpoint 完全一致。

### 分片后的权重从哪来

**还是同一个 HF checkpoint。** 加载时通过 `weight_loader` 方法从完整权重中切出当前 GPU 的分片：

```python
# linear.py:96-106 (ColumnParallelLinear.weight_loader)
def weight_loader(self, param, loaded_weights):
    """
    param.data:      当前GPU的分片容器，形状 (2048, 1024) [TP=2]
    loaded_weights:  从HF读出的完整权重，形状   (4096, 1024)
    """
    shard_size = 4096 // self.tp_size     # 2048
    start = self.tp_rank * shard_size     # GPU0: 0, GPU1: 2048
    param.data.copy_(
        loaded_weights.narrow(0, start, shard_size)  # 沿第0维切一片
    )
```

> 注意：当前 MinivLLM 的 loader.py 在加载 QKV 和 gate_up 时走的是手动拼接路径（Q5 所述），路径中直接调用 `param.data.copy_(qkv_weight)`，**并没有走 weight_loader**。`weight_loader` 的设计是为未来更自动化的 TP 加载场景准备的——比如直接从 checkpoint key 按名加载子矩阵时。

---

## Q8 · MinivLLM 定义了很多自定义模块（RowParallelLinear、Attention 等），它们凭什么能安全加载 HF 的原始权重？

**因为加载权重只看两样东西：参数的名字和形状。至于这个参数包装在什么类里、前向计算怎么写，完全不影响加载。**

你可以把这个过程想象成"快递投递"：

- HF checkpoint 是一个巨大的仓库，里面每个包裹贴着标签（参数名），装着货物（张量）。
- 你的模型是一栋楼，每个房间有门牌号（也是参数名）和指定大小的柜子（形状）。
- 加载过程就是：读标签 → 找到同名房间 → 把货物塞进柜子。如果柜子尺寸不匹配，快递员会报错。但除此之外，他不会关心你房间里后来拿这个货物做了什么。

因此：

- `RowParallelLinear` 虽然继承自 `LinearBase`，功能与 `nn.Linear` 不同，但它的 `.weight` 参数名称和形状如果与 HF 一致，加载就毫无问题。
- `VocabParallelEmbedding` 虽然是自定义类，但它的 `self.weight` 同样是 `nn.Parameter`，名字是 `model.embed_tokens.weight`，与 HF 一致。
- `Attention` 模块甚至没有权重参数——它只持有 KV cache 引用和 Triton Kernel，加载过程根本不涉及它。

**核心原则：模型代码的职责分两块——"骨架"由构造函数负责（决定参数名称和形状），"灵魂"由 loader 负责（填充参数值）。两块各干各的，通过参数名这个接口对接。**

---

## Q9 · 为一个新模型适配这套加载机制，需要做哪些事？

假设来了一个新模型（比如假想的 Qwen3.6），适配步骤非常清晰：

### Step 1：搞清楚超参数

打开 HF 模型仓库的 `config.json`，记录所有架构相关参数。关注这些字段尤其重要：

```
hidden_size, num_hidden_layers, num_attention_heads,
num_key_value_heads, intermediate_size, vocab_size,
rope_theta, rms_norm_eps, tie_word_embeddings,
head_dim (如果有的话)
```

### Step 2：查看 safetensors 的 key 列表

几行代码打出全部 key 和形状：

```python
from safetensors import safe_open
for fname in sorted(os.listdir(checkpoint_path)):
    if fname.endswith('.safetensors'):
        with safe_open(os.path.join(checkpoint_path, fname), framework="pt") as f:
            for k in f.keys():
                print(f"{k:60s}  {list(f.get_tensor(k).shape)}")
```

对照你的模型 `named_parameters()` 输出。完全一致的 key 无需处理，不一致的才需要映射。

### Step 3：判断架构是否变化

如果新模型仍是标准的 LLaMA 架构（RMSNorm → Attention → RMSNorm → MLP），那 Qwen3 或 Llama 的代码可以直接复用，只改 config 参数。如果模型引入了新的独特模块（如 Qwen3 的 `q_norm` / `k_norm`），需要额外实现该子模块并处理对应的权重加载。

### Step 4：在 model_runner 中注册新模型

`model_runner.py:32-69` 中按 `model_name` 分发到不同的模型类。添加新的 case 分支即可。

### Step 5：处理命名分歧

如果需要合并或重命名某些矩阵（类似 QKV / gate_up），在 loader 中添加对应的 if 分支。核心操作只有三个 API：

- `model.get_parameter(name)` — 按路径名定位目标参数
- `torch.cat([...], dim=N)` — 拼接多个 HF 矩阵
- `param.data.copy_(tensor)` — 拷贝

### Step 6：测试

运行一次前向，看是否报形状不匹配的错。如果报错，大概率是 config 参数写错了，导致模型初始化时创建的参数形状与 HF checkpoint 不符。

---

## Q10 · 开发者容易踩哪些坑？

**① 不打印 key 就猜**：先 `safe_open(...).keys()` 全部打印出来，再写模型代码，是最省时间的做法。

**② 拼接维度搞错**：QKV 合并时注意 dim。HF 的 Q/K/V 各自是按头数 × head_dim 排列的，拼接后必须与前向计算中 `split` 或 `chunk` 的维度一致。

**③ vocab_size 不匹配**：你的 config 中的 `vocab_size` 必须与 HF 的一致。如果设小了，部分 token 无法表示；设大了，多出来的行保留随机值会导致乱码。loader 已经处理了大小不一时的截断（`loader.py:160`），但最好还是设成一样。

**④ 设备不对**：MinivLLM 选的是先 `model.cuda()` 再加载权重，这样张量拷贝在 GPU 内存中直连完成。如果反过来（CPU 加载再搬 GPU），大模型可能撑爆 CPU 内存。

**⑤ TP 场景下忘了同步**：多卡 TP 时，所有 rank 必须对同一份 HF checkpoint 各自加载自己的分片。如果某个 rank 的 weight_loader 写错了切片逻辑，算出来的结果就会不一致。

**⑥ 误以为需要"配套"**：很多人直觉认为用 HF 权重就必须要用 HF 模型类，实则不然。拿到 `model.embed_tokens.weight` 这个张量，你可以包在任何自定义模块里——只要前向计算正确、参数名称匹配，就行了。

---

> **一句话总结**：模型权重文件本质上是 `dict[str, Tensor]`。PyTorch 的参数名由属性路径决定。加载就是按 key 找人，找到就拷。名字对不上就写映射，形状对不上就报错。前向计算怎么实现，加载过程完全不关心。
