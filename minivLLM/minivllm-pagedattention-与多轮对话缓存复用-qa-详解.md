# MinivLLM PagedAttention 与多轮对话缓存复用 · Q&A 详解

> 本文系统梳理 MinivLLM 中 PagedAttention 的完整工作流程：prefill 和 decode 的注意力机制差异、QKV 的计算时机、多轮对话中 prefill 与 prefix cache 的互动，以及新请求如何复用历史 cache。

---

## Q1 · Prefill 和 Decode 的注意力机制有什么不同？Q 在 prefill 中需要新算吗？

两者使用的注意力机制不同，但 Q、K、V 的计算方式对单个 token 而言完全一致：

| | Prefill | Decode |
|---|---|---|
| **处理 token 数** | prompt 全部（如 128 个） | 每序列 1 个 |
| **Q** | 新算（全部 prompt token 的 Q） | 新算（仅 1 个 token 的 Q） |
| **K** | 新算（全部 prompt token 的 K） | 新算（仅 1 个 token 的 K） |
| **V** | 新算（全部 prompt token 的 V） | 新算（仅 1 个 token 的 V） |
| **注意力实现** | Flash Attention（QKV 全在显存中） | Paged Attention（Q 新算，K、V 从 cache 读） |
| **K、V 存放** | 全量写入 cache（建立初始缓存） | 单 token 追加写入 cache |
| **Q 存放** | 不存放，用完即弃 | 不存放，用完即弃 |

关键在于：prefill 的 Q 当然需要新算——和 K、V 一样，都在当前 forward pass 中通过 `qkv_projection(x)` 实时计算。只不过 prefill 结束后，Q 被直接丢弃（后续步骤不再需要它），而 K、V 需要保留到 cache 中。

对单个 token 来说，无论 prefill 还是 decode，QKV 计算路径完全相同：

```
token → embedding → 每层循环:
  x → qkv_projection(x) → split(Q, K, V) → RoPE(Q, K) → Q/K norms
```

差别不在"怎么算"，而在"何时算"和"算完之后怎么用"。

---

## Q2 · Prefill 的 Flash Attention 和 Decode 的 Paged Attention 具体工作方式是什么？

以一个包含 3 个 token 的 prompt `[A, B, C]` 为例，block_size=256，分配在物理块 5。

### Prefill → Flash Attention

```
输入: [A, B, C] (1D, 3 tokens)

第 N 层 forward:
  x → QKV投影 → Q(3,16,128), K(3,8,128), V(3,8,128)    ← 3 个 token 全部一次性获得
  → RoPE → Flash Attention(Q, K, V, cu_seqlens=[0,3])
    ┌───────────────────────────────────────────────┐
    │  Flash Attention 内部 (分块计算, 不做大矩阵):    │
    │  Q₀ × [K₀]        → softmax → ×V₀ → out₀      │
    │  Q₁ × [K₀, K₁]    → softmax → ×[V₀,V₁] → out₁  │
    │  Q₂ × [K₀, K₁, K₂] → softmax → ×[V₀,V₁,V₂] → out₂│
    │  (因果掩码: 只看当前及以前的 token)               │
    └───────────────────────────────────────────────┘
  → store_kvcache: K→物理块5 offset[0,1,2], V→物理块5 offset[0,1,2]
  → 输出 (3, 16×128)
```

### Decode → Paged Attention

```
输入: [D] (1D, 1 token, 假设是刚生成的第 4 个 token)

第 N 层 forward:
  x → QKV投影 → Q(1,16,128), K(1,8,128), V(1,8,128)    ← 仅 1 个 token
  → RoPE → store_kvcache: K,V→物理块5 offset 3          ← 追加写入
  → Paged Attention(Q, k_cache, v_cache, block_table, context_len=4):
    ┌─────────────────────────────────────────────────────┐
    │  分 64 token 的 chunk 遍历 (context_len=4, BLOCK_N=64) │
    │  chunk 0: 处理 token 0~3                            │
    │    Token 0 (A): block=0,off=0 → k_cache[5,0] → Q·K  │
    │    Token 1 (B): block=0,off=1 → k_cache[5,1] → Q·K  │
    │    Token 2 (C): block=0,off=2 → k_cache[5,2] → Q·K  │
    │    Token 3 (D): block=0,off=3 → k_cache[5,3] → Q·K  │
    │    softmax → ×[V₀,V₁,V₂,V₃] → output                │
    └─────────────────────────────────────────────────────┘
  → 输出 (1, 16×128)
```

注意：decode 中刚写入的 token D 的 K、V 也会立即被读回参与 attention——因为 `store_kvcache` 在 `paged_attention_decode` 之前执行。

---

## Q3 · Prefill 只在第一轮对话中进行吗？追加新问题时还会 prefill 吗？

**不是。每次调用 `generate(prompt)` 都会对传入的 prompt 执行 prefill。**

以两轮对话为例，调用方（应用层）的典型做法：

```python
# Turn 1
prompt_1 = tokenizer.apply_chat_template([
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"}
])
output_1 = llm.generate([prompt_1], sampling_params)
# 内部: prefill(system + Q1) → decode → 得到 A1

# Turn 2
prompt_2 = tokenizer.apply_chat_template([
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": output_1},
    {"role": "user", "content": "今天天气怎么样"}
])
output_2 = llm.generate([prompt_2], sampling_params)
# 内部: prefill(system + Q1 + A1 + Q2) → decode → 得到 A2
```

Turn 2 的 prompt 包含了完整历史，**全部内容都会走 prefill**。那 KV cache 加速了什么？

- **加速了 decode 阶段**：生成 A2 时，每个新 token 不需要重新计算 system+Q1+A1+Q2 的 K、V——它们已经存在于 cache 中了。
- **prefill 阶段本身也可能被加速**（见 Q4 的 prefix caching）。

---

## Q4 · 当系统收到 Turn 2 的 prompt 时，它怎么知道和 Turn 1 有关？怎么知道应该复用哪个 KV cache？

**通过内容哈希自动匹配，不需要显式标记会话 ID 或主动"关联"。**

核心机制在 `block_manager.py:67-97` 的 `allocate()` 方法。当 Turn 2 的新 prompt 被 tokenize 后，BlockManager 按 block（256 token）为单位遍历这些 token：

### 前置条件：Turn 1 结束后发生了什么？

Turn 1 的序列完成后，scheduler 调用 `block_manager.deallocate(seq)` (`block_manager.py:99-107`)。块的 `ref_count` 减 1，如果为 0 则归还到 free list。**但 `hash_to_block_id` 映射没有被清除。**

### Turn 2 到达时的匹配流程

```
Turn 2 prompt 被 tokenize: [system+Q1+A1+Q2]
按 block_size=256 切成块: [块0], [块1], [块2], [块3]

遍历这些块:
┌──────────────────────────────────────────────────────────────┐
│ 块0: token_ids = system+Q1 前半部分 (256个token)              │
│   hash = compute_hash(token_ids, prefix=-1) = 0xABCD         │
│   block_id = hash_to_block_id.get(0xABCD) = 物理块5           │
│   blocks[5].token_ids == 块0的token_ids?                     │
│   → 情况A: 是 (物理块5还没被覆盖) → 命中!                      │
│           seq.num_cached_tokens += 256                        │
│           不重新 prefill 这 256 个 token                       │
│   → 情况B: 否 (物理块5已被其他序列回收) → 未命中               │
│           从 free list 分配新块, prefill 重算                  │
├──────────────────────────────────────────────────────────────┤
│ 块1: token_ids = system+Q1 后半部分                           │
│   类似块0的匹配逻辑...                                        │
├──────────────────────────────────────────────────────────────┤
│ 块2: token_ids = A1 部分 (Turn 1 的生成结果, 之前没缓存过)     │
│   hash = compute_hash(token_ids, 块1的前缀hash)               │
│   hash_to_block_id 中查不到 → 未命中                           │
│   从 free list 分配新块 → prefill 计算这 256 个 token          │
├──────────────────────────────────────────────────────────────┤
│ 块3: token_ids = Q2 部分                                      │
│   同样未命中 → 分配新块 → prefill                               │
└──────────────────────────────────────────────────────────────┘
```

最终，`seq.num_cached_tokens` = 命中的 token 数。在 `prepare_prefill` (`model_runner.py:283`)：

```python
input_ids.extend(token_ids[num_cached_tokens:])  # 只把新 token 送入模型
```

只有未命中的块（A1+Q2）需要被真实 prefill。命中的块（system+Q1）的 K、V 已经在 cache 中了，直接复用其物理块。

### 关键要点

- **关联靠内容哈希，不靠会话 ID**。两个完全相同前缀的请求，就算来自不同用户、不同会话，也会自动共享物理块。
- **哈希是链式的**：`compute_hash(当前块token_ids, 前缀块的哈希值)`，前缀变了，后续哈希全变。这保证了上下文敏感——同样的 256 个 token，如果前面的内容不同，hash 也不同。
- **只有当物理块还在、且内容完全匹配时才命中**。如果 Turn 1 结束后 KV cache 池紧张，物理块被新序列覆盖，则 Turn 2 的 prefix cache 就会 miss，需要重新 prefill。

---

## Q5 · 如果 Turn 1 结束后 KV cache 池满了，Turn 1 的块就被清除了吗？

**不一定，取决于块的引用计数和 free list 的状态。**

`block_manager.deallocate()` 的逻辑 (`block_manager.py:99-107`)：

```python
def deallocate(self, seq):
    for block_id in seq.block_table:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)  # 归还到 free list
            # 注意: hash_to_block_id 没有被清除
```

归还到 free list 的块只是"标记为可回收"，并不是立即清除其 GPU 内存中的数据。物理块实际被覆盖发生在以下时刻：

1. 有新序列通过 `_allocate_block()` 从 free list 弹出该块
2. 新序列在这个块中写入新的 K、V 数据

在这之前，旧数据仍然在 GPU 内存中。但由于 `block.token_ids` 在 `_deallocate_block` 中被清空 (`block_manager.py:55-60`)：

```python
def _deallocate_block(self, block_id):
    block = self.blocks[block_id]
    block.token_ids = []          # ← 清空元数据
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

下一个请求做 prefix cache 匹配时 (`block_manager.py:78`)：

```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    no_cache_found = True
```

因为 `token_ids = [] ≠ 新请求的 token_ids`，即使 GPU 内存中数据还在，也会判定为未命中，需要重新 prefill。**所以在当前实现中，prefix cache 跨序列生效的前提是块还没有被回收。**

---

## Q6 · Prefill 和 Decode 计算单个 token 的 K、V 有差别吗？

**没有。** 无论是 prefill 还是 decode，对单个 token 的 Q、K、V 计算完全一致：

```
一个 token 经过 Layer N:
  x = 该 token 的 hidden state (1, 1024)
  qkv = qkv_projection.weight @ x   → (1, 4096)   [Q:2048 + K:1024 + V:1024]
  q, k, v = split(qkv)              → Q(1,16,128), K(1,8,128), V(1,8,128)
  q, k = RoPE(q, k, position)       → 按 token 在序列中的位置旋转
  q, k = qnorm(q), knorm(k)         → Qwen3 特有的 Q/K 归一化
```

差别只在**批量大小**和**后续处理**：

| 阶段 | 批量大小 | 后续处理 |
|---|---|---|
| Prefill | prompt 全量 tokens 并行（如 128 个） | Flash Attention + 全量写入 cache |
| Decode | 每序列 1 个 token | 单点追加 cache + Paged Attention 读全量 cache |

prefill 就是一次大的矩阵乘法算完所有 tokens，decode 是每次算一个。两者的 K/V 投影操作本身无差异。

---

## Q7 · 一条完整的多轮对话，从头到尾的时间线是怎样的？

以 `block_size=4`（方便画图，实际是 256），两轮对话为例：

```
═══════ Turn 1 ═══════
prompt = system+Q1 (8 tokens: [S0,S1,S2,S3, Q0,Q1,Q2,Q3])

① Prefill:
   allocate序列: 2个块(物理块5, 物理块7) → prefill输入8个token
   → 28层各自算出K,V → 写入cache:
       物理块5: [K_S0,K_S1,K_S2,K_S3], [V_S0,V_S1,V_S2,V_S3]
       物理块7: [K_Q0,K_Q1,K_Q2,K_Q3], [V_Q0,V_Q1,V_Q2,V_Q3]
   → 最后一个位置的logit → sample → token A0
   hash_to_block_id: {hash([S0..S3]): 5, hash([Q0..Q3]): 7}

② Decode step 1 (生成 A0):
   新K_A0,V_A0 → 写入物理块7 offset4? 
   → 物理块7已满 → block_manager.append: 分配新物理块12
   物理块12: [K_A0,V_A0] offset=0

③ Decode step 2~N: 继续生成 A1,A2,...依次追加到物理块12
   最终物理块12: [K_A0,V_A0, K_A1,V_A1, K_A2,V_A2, K_A3,V_A3]
   hash_to_block_id: {hash([A0..A3]): 12}

④ 序列结束: scheduler标记FINISHED → block_manager.deallocate
   物理块5,7,12 归还 free list (ref_count→0)
   hash_to_block_id 保留: {hash([S0..S3]):5, hash([Q0..Q3]):7, hash([A0..A3]):12}
   blocks[5].token_ids=[]  blocks[7].token_ids=[]  blocks[12].token_ids=[]

═══════ Turn 2 ═══════
prompt = system+Q1+A1+Q2 (16 tokens: [S0..S3, Q0..Q3, A0..A3, Q2_0..Q2_3])

① allocate序列:
   块0 [S0..S3]: hash=hash([S0..S3])=H0 → hash_to_block_id[H0]=5
                blocks[5].token_ids==[]? → 否, 不匹配 → 未命中 (块已回收)
                从free list分配新块(假设物理块9)
   块1 [Q0..Q3]: 同样未命中 → 分配新块物理块2
   块2 [A0..A3]: 未命中 → 分配新块物理块15
   块3 [Q2_0..Q2_3]: 未命中 → 分配新块物理块8
   num_cached_tokens = 0

② Prefill: 全部16 tokens重新走模型 → 写入物理块9,2,15,8

③ Decode: 逐token生成A2_0,A2_1,... 
```

**注意**：Turn 2 中 system+Q1 虽然和 Turn 1 内容相同，但因为块在 Turn 1 结束后被回收（token_ids 被清空），prefix cache 无法命中，需要重新 prefill。这是在**没有显式跨序列 cache 保留策略**的情况下。

如果 Turn 1 和 Turn 2 的序列在**同一时间共存**（比如 Turn 2 在 Turn 1 还未完成时到达），共用前缀的块就不会被 deallocate，prefix cache 可以直接命中——哈希匹配成功，`num_cached_tokens` 增加，只 prefill 新增内容。

---

## Q8 · K 和 V 在 decode 计算中得到新值时，是不是 K"矩阵"大部分从 cache 中得到，新增的只有一行？

你的直觉是对的。在 decode 阶段，注意力计算需要的完整 K 数据几乎全部来自 cache。具体流程是：

```
decode 阶段，生成一个 token:
  ├─ 第一步: QKV投影 → 新算 K_new(1,8,128), V_new(1,8,128)
  ├─ 第二步: store_kvcache(K_new, V_new) → 追加写入 cache
  └─ 第三步: Paged Attention
       Q(1,16,128) × [K₀...K_old...K_new 全部从 cache 读] → softmax
       × [V₀...V_old...V_new 全部从 cache 读] → output
```

如果把所有历史 K 视为一个"逻辑 K 矩阵"：

```
逻辑K = [K₀ | K₁ | ... | K_历史最大值 | K_new]   ← shape (context_len, 8, 128)
                                                  ← 假想的"完整矩阵"

实际: 从不创建此矩阵, 按 64 token chunk 流式从 cache 读取。
      K_new 先被追加写入了 cache, 然后在同一个 chunk 循环中被读回。
```

新增确实只有一行（新 token 的 K 和 V），其余全部通过 block_table 从 cache 获取。这也是单步 decode 计算量恒定为 O(context_len) 的原因。
