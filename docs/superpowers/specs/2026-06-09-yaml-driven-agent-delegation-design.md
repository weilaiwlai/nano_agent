# YAML 驱动动态委派架构设计

**日期**: 2026-06-09  
**状态**: 已确认  
**范围**: Python 端 (`agent_service/`)

---

## 1. 概述

### 1.1 目标

将 NanoAgent 从硬编码 Agent 路由重构为 **YAML 驱动的动态委派架构**：

1. **YAML 定义 Agent**：每个 Agent 类型由独立的 `.agent.yaml` 文件定义，包含触发规则、上下文、工具绑定、循环配置和报告格式
2. **注册中心统一管理**：`AgentRegistry` 在启动时扫描并校验所有 YAML 文件，构建关键词索引，提供统一查询接口
3. **Server 层关键词智能规划**：用户请求先经 `KeywordMatcher` 匹配 → 高置信度直接路由，模糊场景回退 LLM 分类
4. **独立上下文和循环**：每个 Agent 实例拥有隔离的 `AgentContext`，自带内部循环能力，执行完返回结构化报告

### 1.2 核心收益

| 收益 | 说明 |
|------|------|
| **降低 LLM 成本** | 确定性路由不再每次调 LLM，仅在关键词匹配不足时回退 |
| **扩展零代码** | 新增 Agent = 写一个 YAML + 一个 prompt 文件，注册中心自动发现 |
| **上下文隔离** | 每个 Agent 独享消息历史，杜绝跨 Agent 状态串扰 |
| **结构化输出** | 每个 Agent 返回 JSON Schema 定义的结构化报告，前端/API 消费者可解析 |

### 1.3 与 LangGraph 的关系

**混合模式**：
- **单 Agent 场景**：注册中心匹配 → `AgentRunner` 直调，不经过 LangGraph
- **多 Agent 协作**：保留 LangGraph 顶层编排器，节点从硬编码函数变为 `AgentRegistry` 动态获取 Agent 实例

---

## 2. YAML Agent 定义契约

### 2.1 文件位置

```
agent_service/agents/
├── data_analyst.agent.yaml
├── reporter.agent.yaml
├── assistant.agent.yaml
├── data_analyst.system.md          # 独立 prompt 文件
├── reporter.system.md
├── assistant.system.md
├── data_analyst.report.schema.json # 报告 JSON Schema
├── reporter.report.schema.json
└── assistant.report.schema.json
```

### 2.2 Schema

```yaml
id: data_analyst
name: 数据分析师
version: 1.0.0
description: >
  负责数据库查询、指标计算和趋势分析。

routing:
  keywords:
    - [销售额, 销量, 营收]
    - [库存, 周转, 滞销]
    - [利润, 成本, 财务]
    - [同比, 环比, 趋势]
  min_confidence: 0.6
  priority: 10

context:
  system_prompt: data_analyst.system.md
  memory:
    enabled: true
    max_turns: 20
  skills: []

loop:
  max_iterations: 8
  termination:
    - type: no_tool_calls
    - type: pattern_match
      pattern: "FINAL_ANSWER:"
  timeout_seconds: 120

tools:
  - name: query_database
    risk: high
  - name: get_current_time
    risk: safe

report:
  schema: data_analyst.report.schema.json
  format: json

llm:
  model: null
  temperature: 0.1
  max_tokens: 4096
```

### 2.3 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 唯一标识，与文件名一致 |
| `name` | string | ✅ | 展示名称 |
| `version` | semver | ✅ | 语义化版本 |
| `description` | string | ✅ | 一句话描述 + 适用场景 |
| `routing.keywords` | list[list[str]] | ✅ | 触发关键词词组，内层列表表示高权重共现 |
| `routing.min_confidence` | float | ✅ | 置信度阈值 (0-1)，低于此值回退 LLM |
| `routing.priority` | int | ❌ | 冲突优先级，默认 0，越大越优 |
| `context.system_prompt` | filepath | ✅ | 相对 agents/ 的 prompt 文件路径 |
| `context.memory` | object | ❌ | 用户长期记忆注入配置 |
| `context.skills` | list[str] | ❌ | 可用技能白名单，空=全部 |
| `loop.max_iterations` | int | ✅ | 最大循环轮次 |
| `loop.termination` | list | ✅ | 终止条件列表 |
| `loop.timeout_seconds` | int | ✅ | 超时秒数 |
| `tools[].name` | string | ✅ | 工具名，匹配 MCP 注册名 |
| `tools[].risk` | enum | ✅ | `high`（需审批）/ `safe`（直接执行） |
| `report.schema` | filepath | ❌ | JSON Schema 文件路径 |
| `report.format` | enum | ✅ | `json` / `markdown` / `mixed` |
| `llm` | object | ❌ | Agent 级 LLM 覆盖，null 字段用全局默认 |

### 2.4 Pydantic 校验模型

启动时，`AgentRegistry` 使用 Pydantic 校验所有 YAML 文件：
- 结构合法性
- `id` 与文件名一致性
- `context.system_prompt` 文件存在性
- `report.schema` 文件存在性（若指定）
- `tools[].name` 在 MCP 工具注册表中的存在性

---

## 3. Agent 注册中心

### 3.1 AgentRegistry 接口

```python
class AgentRegistry:
    """全局单例，启动时初始化，全生命周期可用"""

    # ── 内部数据结构 ──
    registry: dict[str, AgentDefinition]    # id → 完整定义
    keyword_index: dict[str, set[str]]      # 分词 → {agent_id}

    # ── 生命周期 ──
    @classmethod
    async def init(cls, agents_dir: str) -> "AgentRegistry":
        """扫描 agents/ 目录，解析 YAML，构建索引，返回单例"""

    async def reload(self) -> None:
        """重新扫描目录（仅用于手动触发的热更新，不依赖 watchdog）"""

    # ── 查询 ──
    def match(self, user_input: str) -> MatchResult:
        """关键词匹配 → (agent_id, confidence, matched_keywords) | None"""

    def get(self, agent_id: str) -> AgentDefinition | None:
        """按 id 获取定义"""

    def list_all(self) -> list[AgentDefinition]:
        """列出所有已注册 agent"""


@dataclass
class MatchResult:
    agent_id: str
    confidence: float
    matched_keywords: list[str]
```

### 3.2 关键词匹配算法

```
输入: "查一下上个月华东区的销售额"

1. jieba 分词 → [查, 上个月, 华东区, 销售额]
2. 倒排索引查询 → 每个词属于哪些 agent
   - 销售额 → {data_analyst, reporter}
   - 华东区 → {}  (未被任何 agent 列为关键词)
3. 计分:
   score(agent) = matched_keywords_count / total_agent_keywords
   - data_analyst: 1/12 = 0.083 → 归一化后 × 权重
4. 归一化: 所有候选 agent scores → 总和为 1 的概率分布
5. 判定:
   - top_agent.confidence >= top_agent.min_confidence 且
     top_agent.confidence - second_agent.confidence > 0.15 (gap阈值)
   → 直接路由
   - 否则 → 返回 None → 触发 LLM 分类回退
```

### 3.3 LLM 分类器（回退路径）

关键词匹配失败时，调用精简的 LLM 分类调用：

- **输入**: 用户消息 + 所有 agent 的 `(id, name, description, keywords)`
- **输出**: `{"agent_id": "x", "confidence": 0.92, "reasoning": "..."}`
- **可用 agent 列表动态构建**（来自 AgentRegistry），不硬编码

如果 LLM 分类分数也低于阈值 → 回退到 `assistant` agent。

---

## 4. Server 层路由

### 4.1 路由决策流程

修改 `routes.py` 的 `chat()` 端点，加入路由层：

```python
# routes.py  chat() 端点内的新流程

user_input = request.message

# Step 1: 精确指令检测
if user_input.startswith("/agent "):
    agent_id = user_input.split()[1]
    agent_def = registry.get(agent_id)
    if agent_def:
        return await run_single_agent(agent_def, user_input)

# Step 2: 关键词匹配
match = registry.match(user_input)
if match and match.confidence >= registry.get(match.agent_id).routing.min_confidence:
    agent_def = registry.get(match.agent_id)
    return await run_single_agent(agent_def, user_input)

# Step 3: 多 Agent 意图检测（仍走 LangGraph）
if _detect_multi_agent_intent(match):
    # 回到当前 LangGraph orchestration 流程
    return await run_langgraph_workflow(user_input, request)

# Step 4: LLM 分类
llm_result = await llm_classifier.classify(user_input, registry.list_all())
if llm_result.confidence > LLM_CLASSIFY_THRESHOLD:
    agent_def = registry.get(llm_result.agent_id)
    return await run_single_agent(agent_def, user_input)

# Step 5: 兜底 → assistant
return await run_single_agent(registry.get("assistant"), user_input)
```

### 4.2 多 Agent 意图检测

保留当前 LangGraph `orchestrator_node` 的判断逻辑，增加一个简单前置检查：

- 如果关键词匹配出现 **两个 agent 的置信度相近** (差值 < 0.1)，判定为潜在多意图
- 将此类请求完整交给 LangGraph 编排器

### 4.3 路由决策矩阵

| 场景 | 判断条件 | 处理方式 |
|------|----------|----------|
| 精确指令 | `/agent data_analyst` | 跳过匹配，直接路由 |
| 关键词高置信 | top.conf > min_confidence 且 gap > 0.15 | AgentRunner 直调 |
| 潜在多 Agent | 两个 agent 置信度接近 | LangGraph 编排 |
| 模糊描述 | 所有 agent conf 低于阈值 | LLM 分类 → AgentRunner |
| 完全无法分类 | LLM 也判断不了 | assistant 兜底 |

---

## 5. Agent 独立执行引擎 (AgentRunner)

### 5.1 生命周期

```
输入 (agent_def, user_input, user_state)
  │
  ├─ 1. 构建 AgentContext
  │      ├─ 加载 system_prompt 文件
  │      ├─ 注入用户长期记忆 (ChromaDB)
  │      └─ 组装首条用户消息
  │
  ├─ 2. 进入循环
  │      ├─ LLM 调用 (当前 AgentContext)
  │      ├─ 若 LLM 返回 tool_calls → 按 risk 分级执行
  │      │    ├─ safe  → 直接执行，结果回填 messages
  │      │    └─ high  → 发送 approval_required SSE，挂起等待
  │      ├─ 若 LLM 返回纯文本 → 检查 termination 条件
  │      │    ├─ 匹配终止 → 退出循环
  │      │    └─ 未匹配   → 追加到 messages，继续下一轮
  │      └─ 若超过 max_iterations 或 timeout → 强制终止
  │
  ├─ 3. 生成结构化报告
  │      ├─ 提取 FINAL_ANSWER: 后的 JSON
  │      ├─ 用 report.schema 校验
  │      └─ 格式不符 → 最多重试 2 次
  │
  └─ 4. 返回 AgentResult + 清理
```

### 5.2 AgentContext（上下文隔离）

```python
@dataclass
class AgentContext:
    """每个 Agent 实例独享，不污染全局 LangGraph State"""
    agent_id: str
    system_prompt: str               # 从文件加载的完整 prompt
    messages: list[BaseMessage]      # 仅本 Agent 的对话历史
    memory_context: str              # 用户长期记忆（只读注入）
    loop_count: int                  # 当前循环轮次
    tool_history: list[ToolCallResult]  # 本轮已执行的工具调用及结果
    started_at: float                # time.time()
```

与当前 `AgentState` 的关键区别：
- `messages` 不再被多个 Agent 节点共享追加
- 不包含 `current_agent`（不再需要全局状态来跟踪谁在执行）
- 不包含 `orchestrator_context`（不再需要编排器预先规划）

### 5.3 内部循环实现

```python
class AgentRunner:
    async def run(
        self, agent_def: AgentDefinition, user_input: str, user_state: UserState
    ) -> AsyncGenerator[SSEEvent, None]:
        """流式执行一个 Agent，产出 SSE 事件"""

        ctx = await self._build_context(agent_def, user_input, user_state)

        while ctx.loop_count < agent_def.loop.max_iterations:
            ctx.loop_count += 1

            # LLM 调用 + 流式输出 token
            response = await self._call_llm(agent_def, ctx)
            async for event in response.stream():
                yield event

            # 处理工具调用
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_def = agent_def.get_tool(tc.name)
                    if tool_def.risk == "high":
                        yield ApprovalRequiredEvent(tc)
                        decision = await self._wait_for_approval(tc)
                        if not decision.approved:
                            yield ToolRejectedEvent(tc)
                            continue
                    result = await self._execute_tool(tc)
                    ctx.tool_history.append(result)
                    ctx.messages.append(ToolMessage(result))
                    yield ToolResultEvent(tc, result)
                continue  # 有工具调用 → 继续循环

            # 无工具调用 → 检查终止条件
            if self._check_termination(response, agent_def.loop.termination):
                break

            # 未终止 → 追加文本作为新一轮上下文
            ctx.messages.append(AIMessage(response.text))

        # 循环结束 → 生成报告
        report = await self._generate_report(agent_def, ctx)
        yield ReportEvent(report)

        # 持久化对话历史
        await self._save_to_memory(ctx, user_state)
```

### 5.4 安全分级

工具执行复用当前 MCP 代理模式，风险由 YAML 的 `tools[].risk` 声明：

| risk | 行为 |
|------|------|
| `safe` | 直接 `POST http://mcp_server:8000/tools/{name}` 执行 |
| `high` | 中断 → 发送 `approval_required` SSE → 等待用户 `/api/v1/chat/resume` → 执行或拒绝 |

不再需要 `high_risk_tools` 和 `safe_tools` 这两个硬编码的图节点。

### 5.5 结构化报告

```json
{
  "agent_id": "data_analyst",
  "status": "success",
  "summary": "2024年1月华东区销售额：1,234万元，同比增长 12.3%",
  "data": { "total_sales": 12340000, "yoy_growth": 0.123 },
  "tool_calls": [
    {"tool": "query_database", "risk": "high", "approved": true, "latency_ms": 420}
  ],
  "duration_ms": 3400,
  "loop_iterations": 3,
  "error": null
}
```

报告生成逻辑：
1. 从最后一轮 LLM 输出中提取 `FINAL_ANSWER:` 后的 JSON
2. 用 `report.schema` 指向的 JSON Schema 校验
3. 格式不符 → 重试（最多 2 次），每次重试用简短 prompt 要求 LLM 按 Schema 输出
4. 3 次均失败 → 返回 `status: "error"` + 原始文本 + `error: "report_schema_validation_failed"`

### 5.6 LangGraph 集成

对于需要多 Agent 协作的场景，LangGraph 图结构调整为：

```
START → memory_retriever → orchestrator
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
        data_analyst       reporter         assistant
        (AgentRunner)    (AgentRunner)     (AgentRunner)
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                             FINISH
```

- 每个 Agent 节点内部调用 `AgentRunner.run()`，节点自身不再包含 LLM 调用逻辑
- `high_risk_tools` / `safe_tools` 节点**移除**，工具安全分级在 `AgentRunner` 内部处理
- `OrchestratorNode` 简化为：判断需要哪些 Agent、以什么顺序执行 → 按序调度 `AgentRunner`

---

## 6. 文件结构变更

### 6.1 新增文件

```
agent_service/
├── agents/                            # ⭐ 新增
│   ├── data_analyst.agent.yaml
│   ├── reporter.agent.yaml
│   ├── assistant.agent.yaml
│   ├── data_analyst.system.md
│   ├── reporter.system.md
│   ├── assistant.system.md
│   ├── data_analyst.report.schema.json
│   ├── reporter.report.schema.json
│   └── assistant.report.schema.json
├── graph/
│   ├── registry.py                    # ⭐ 新增: AgentRegistry
│   ├── agent_runner.py                # ⭐ 新增: AgentRunner + AgentContext
│   ├── keyword_matcher.py             # ⭐ 新增: 关键词匹配算法
│   ├── llm_classifier.py              # ⭐ 新增: LLM 分类回退
│   ├── models.py                      # ⭐ 新增: Pydantic 模型 (AgentDefinition 等)
│   └── ...
```

### 6.2 修改文件

| 文件 | 变更 |
|------|------|
| `routes.py` | `chat()` 端点加入路由决策层 |
| `main.py` | 启动时调用 `AgentRegistry.init()` |
| `graph/workflow.py` | 图节点改用 AgentRunner，移除 high_risk_tools/safe_tools |
| `graph/nodes.py` | orchestrator_node 简化，data_analyst_node/reporter_node/assistant_node 改为 AgentRunner 包装 |
| `graph/state.py` | AgentState 简化，移除 orchestrator_context、current_agent |
| `graph/prompts.py` | 各 Agent 的 prompt 迁移到 agents/*.system.md |

### 6.3 移除文件

| 文件 | 原因 |
|------|------|
| `graph/routes.py` | 路由逻辑移到 server 层 + `keyword_matcher.py` |

---

## 7. 迁移路径

### Phase 1: 基础设施（不影响现有功能）
1. 创建 `agents/` 目录结构
2. 实现 Pydantic 模型 (`models.py`)
3. 实现 `AgentRegistry`（只扫码、不介入路由）
4. 从 `nodes.py` 和 `prompts.py` 提取现有 Agent 配置，写出对应的 YAML + prompt 文件

### Phase 2: 单 Agent 直通（灰度上线）
5. 实现 `KeywordMatcher`
6. 实现 `AgentRunner`
7. 实现 `LLMClassifier`
8. 在 `routes.py` 加入路由决策层，但加一个 feature flag `YAML_ROUTING_ENABLED`
9. 默认关闭 -> 测试环境开启 -> 验证单 Agent 场景覆盖率

### Phase 3: 全面替换
10. 移除 `high_risk_tools` / `safe_tools` 图节点
11. 简化 LangGraph 图结构
12. 删除 `graph/routes.py`
13. 删除 `AgentState` 中的冗余字段
14. 移除 feature flag，YAML 路由成为唯一路径

---

## 8. 自我审查

### 8.1 内部一致性
- ✅ 所有 YAML 字段在 Pydantic 模型中有对应
- ✅ KeywordMatcher 的 `MatchResult` 与 AgentRegistry.match() 返回值一致
- ✅ AgentRunner 产出的 SSE 事件类型与前端期望匹配
- ✅ 安全分级复用了现有 HITL 审批流程

### 8.2 与现有系统的兼容
- ✅ MCP 工具调用接口不变（`POST /tools/{name}`）
- ✅ ChromaDB 记忆注入方式不变
- ✅ JWT / API Key 认证不变
- ✅ SSE 事件流格式扩展兼容（新增 `report` 事件类型）
- ✅ Feature flag 确保安全灰度

### 8.3 边界情况
- 空 YAML 目录 → AgentRegistry 为空，所有请求走 LLM 分类
- 所有 Agent 关键词匹配失败 + LLM 也失败 → assistant 兜底
- 工具调用超时 → AgentRunner 捕获异常，填入报告 `status: "error"`
- 用户中断（断开 SSE） → AgentRunner 检测连接状态，清理上下文

### 8.4 未覆盖项（有意不做）
- 热重载（watchdog）→ 延后，当前手动 reload 足够
- Agent 间依赖图（`depends_on`）→ 首次实现不包含，保持 Agent 独立
- 动态注册 API → 首次实现不走 API 注册，只走文件扫描
