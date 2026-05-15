# 解读 Hermes Agent 的 Skill 系统：创建后不验证、修改后不测试、"自进化"全靠 Agent 自觉

> 代码分析 · 基于 Nous Research Hermes Agent 源码

---

这篇博客回答我在阅读 Hermes Agent Skill 系统源码时最关心的几个问题：

1. **Skill 创建之后，系统会验证它能不能正常工作吗？会试跑一下吗？**
2. **对于已有的 Skill，系统怎么保证它在持续变好，而不是越来越臃肿？**
3. **修改了 Skill 之后，系统会检查是不是改坏了吗？**
4. **有没有什么机制来判断一个 Skill "写够了"？**

---

## 问题一：创建了 Skill 之后，系统会验证它吗？

**不会。** 只校验文件格式，不验证内容是否能工作。

创建一条 Skill 的入口是 `skill_manage(action='create')`，核心实现在 `tools/skill_manager_tool.py:375-427`。整个流程只做四件事：

1. **校验命名**（第 178-190 行）：只允许 `a-z0-9._-`，防止跨目录攻击
2. **校验分类**（第 192-215 行）：分类名必须是单层目录，不能 `../` 逃逸
3. **校验 frontmatter 结构**（第 217-253 行）：检查 `---` 开头、YAML 能被解析、`name` 和 `description` 字段存在、description 不超过 1024 字符、正文不为空
4. **校验大小**：SKILL.md 不超过 100,000 字符（约 36k tokens），附属文件不超过 1 MiB
5. **检查重名**：所有 skill 目录中不允许同名
6. **安全扫描**（第 78-102 行）：写入后做 prompt injection 检测，发现危险则回滚。但这条**默认关闭**，要用户手动配置才生效

代码写完就写到磁盘，写完之后**不会**做任何下面这些事情：

- ❌ 不提取 Skill 中的命令去终端跑一遍
- ❌ 不检查引用的脚本、文件路径是否存在
- ❌ 不检查步骤描述是否有逻辑漏洞、是否完整
- ❌ 不与同类 Skill 做比较，看有没有重复或矛盾

---

## 问题二：创建完了，Agent 当前会话能看一眼吗？

**不能。**

Skill 的索引在会话启动时一次性从磁盘加载到缓存，之后新增的文件不会更新缓存。这一点在 `hermes-agent-skill-authoring/SKILL.md` 第 126 行明确说明了：

> *"Note: the CURRENT session's skill loader is cached — `skill_view` / `skills_list` will not see the new skill until a new session. This is expected, not a bug."*

所以完整的时间线是：

```
第 1 次对话会话
   Agent 完成任务 → 觉得"这个值得存为 Skill"
   → skill_manage(create) → 格式校验通过 → 写入磁盘 → 返回成功
   → 但当前会话看不到它，不可用
   用户结束会话

第 2 次对话会话（可能是几小时后、几天后、几周后）
   新会话启动，缓存重建，这个 Skill 出现在索引中
   某次 Agent 恰好遇见 Skill 描述里写的场景
   → skill_view() 加载内容 → 照着做
   → 才发现指令可能已经过时了
```

"创建"和"首次使用"之间是一条时间鸿沟，跨度可能是几分钟，也可能是几周。

---

## 问题三：那系统有什么机制来判断"能不能用"？

系统在加载时（`skill_view()`）做的是**环境就绪检查**，不是**内容正确性检查**。

`tools/skills_tool.py:1266-1368` 中检查三个东西：

**1. 环境变量是否配齐**

```yaml
required_environment_variables:
  - name: OPENAI_API_KEY
    optional: false
```

检查 `~/.hermes/.env` 和系统环境变量中这个值是否存在。

**2. 凭证文件是否存在**

检查 `required_credential_files` 声明的文件。

**3. 操作系统是否兼容**

检查 `platforms: [macos, linux]` 声明。

这些检查能回答的是："环境依赖准备好了吗？"

不能回答的是："指令本身对吗？"

一个 Skill 可以通过所有环境检查，但每一步指令都是错的——引用了一个不存在的 API 端点、使用了一个已废弃的 CLI 参数。系统发现不了。

---

## 问题四：已有的 Skill 怎么"进化式优化"？

**没有自动优化机制。全靠 Agent 在使用时自己发现问题、自己决定修复。**

系统提示（`agent/prompt_builder.py:179-186`）中的 `SKILLS_GUIDANCE` 是唯一的"持续改进"策略：

> *"After completing a complex task (5+ tool calls), fixing a tricky error, or discovering a non-trivial workflow, save the approach as a skill..."*
>
> *"When using a skill and finding it outdated, incomplete, or wrong, patch it immediately with skill_manage(action='patch') — don't wait to be asked. Skills that aren't maintained become liabilities."*

Agent"优化"已有 Skill 的路径是：

```
一个 Skill 被使用 → 照着做 → 出错了
    ↓
Agent 判断"这个 skill 过时了/错了/不完整"
    ↓
调用 skill_manage(action='patch')
    ↓
fuzzy match 引擎在原文档中定位目标文本
    ↓
做原子替换
    ↓
安全扫描通过 → 更新成功
    ↓
patch_count +1，存档，等待下次使用
```

`_patch_skill()` 函数（第 463-554 行）提供了一些保护：
- 使用模糊匹配引擎（`tools/fuzzy_match.py`），处理空格差异和缩进变化，Agent 不需要精确复现原文就能匹配
- 写入前备份原文，写入后跑安全扫描，发现注入则自动回滚
- 如果 patc 的是 SKILL.md，会重新校验 frontmatter 结构

有几件事情不做：

- **修改后不验证正确性。** Agent 在 patch 时可能引入同样的错误，没有第二轮检查。
- **修改后无回归检测。** 没有机制比较修改前后的行为变化。
- **未被使用的 Skill 得不到修复。** 一个从创建起就是错的 Skill，如果没被加载过，永远不会被修正。

---

## 问题五：怎么避免 Skill 系统越来越臃肿？

设计了三条防线，但全部**依赖 Agent 的主动行为**。

### 防线一：同名去重

`_find_all_skills()`（`tools/skills_tool.py:549-623`）用 `seen_names: set` 过滤同名文件。但**只防同名**。两个不同名的 Skill 写的是同一个工作流，系统不知道。

### 防线二：引导 Agent "创建之前先看看"

`hermes-agent-skill-authoring` 这个内置 Skill（教 Agent 怎么写 Skill 的 Skill）包含两条约束：

> *"Survey peers in the target category before creating. Read 2-3 peer SKILL.md files to match tone and structure."*
>
> *"Prefer extending an existing skill to creating a narrow sibling."*

这是 prompt 层面的约定，不是强制性规则。系统信任 Agent 自己判断该不该新建。

### 防线三：删除必须声明合并去向

`_delete_skill()`（第 557-611 行）要求传入 `absorbed_into` 参数：

- `absorbed_into=""` → 真的剪掉了，无转发目标
- `absorbed_into="xxx"` → 内容被吸收进了伞 Skill xxx

并且验证目标存在：

```python
target = _find_skill(target_name)
if not target:
    return {"error": f"absorbed_into='{target_name}' does not exist. Create or patch the umbrella skill first, then retry the delete."}
```

先合并后删除，不能先删了再说。

### 防线四（被动）：使用频率生命周期

`tools/skill_usage.py` 通过 `.usage.json` 文件追踪 view_count、use_count、patch_count。超过配置天数未使用的 Skill，curator 自动归档。

但这个机制**只看使用频率，不看内容价值**。一个内容全错的 Skill 如果每天被人尝试加载一次，它会被保留。一条完美可用的 Skill 如果半年没人触发对应场景，它会被归档。

---

## 问题六：系统对"这个 Skill 写够了"有判断吗？

**没有。** 没有任何机制能回答"这段指令是否完整描述了它该做的事情"。

`hermes-agent-skill-authoring` Skill 中推荐的结构模板确实是：

```
## Overview → ## When to Use → body → ## Common Pitfalls → ## Verification Checklist
```

但这是约定，不是强制。Agent 可以写一个只有 `# Title` 和一行文字的文件，格式校验能通过，系统认为一切正常。

---

## 综合回答：这个系统到底能做什么、不能做什么

把完整的闭环画出来，边界就很清楚了：

```
创建 Skill
    ↓
系统做：格式校验、重名检查、大小检查
系统不做：验证内容、试跑
    ↓
当前会话不可见，等下次会话
    ↓
加载 Skill 时
    ↓
系统做：检查环境变量、凭证文件、平台兼容
系统不做：检查指令逻辑、验证引用路径
    ↓
Agent 按指令执行
    ↓
成功：无事发生，等一下次使用
失败：系统提示 "patch it"，Agent 自主修复
    ↓
patch 时
    ↓
系统做：模糊匹配、原子写入、安全检查、frontmatter 结构校验
系统不做：验证修改是否正确、回归比较
```

**系统能保证的：**

| 方面 | 机制 | 代码位置 |
|------|------|----------|
| 文件格式正确 | frontmatter 校验 | `skill_manager_tool.py:217-253` |
| 不重名 | `seen_names` 去重 + 创建时 `_find_skill` | `skills_tool.py:549-623`、`skill_manager_tool.py:393-398` |
| 修改不破坏格式 | patch 后重新校验 frontmatter | `skill_manager_tool.py:533-540` |
| 删除合并有声明 | `absorbed_into` 验证目标存在 | `skill_manager_tool.py:557-611` |
| 长期不用自动归档 | curator 基于使用频率 | `skill_usage.py` |

**系统不保证的：**

| 方面 | 问题 |
|------|------|
| 创建时验证内容 | 不试跑、不检查逻辑完整性 |
| 修改后验证正确性 | 不回测、无回归检测 |
| 内容去重 | 只防同名，不防内容重复 |
| 判断"写够了" | 没有完备性标准 |
| 从未使用的 Skill | 不会获得任何修复机会 |

---

## 最后

Hermes Agent 的 Skill 系统把"内容质量"完全交给了 Agent 自己的判断力。它做的是结构层面的保障——格式、命名、大小、安全——这些都是确定性的、可以自动判断的事。内容层面的验证它不做，因为自然语言指令没有唯一的"正确"定义，无法在离线状态下判断。

至于 Agent 能不能承担好这个责任——能不能准确判断"这个值得存"、能不能在使用中发现所有缺陷、能不能在 patc 时不引入新问题——那是另一回事了。
