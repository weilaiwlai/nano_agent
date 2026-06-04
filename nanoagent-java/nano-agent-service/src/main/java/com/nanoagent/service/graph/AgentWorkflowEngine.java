package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.CompileConfig;
import com.alibaba.cloud.ai.graph.CompiledGraph;
import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.StateGraph;
import com.alibaba.cloud.ai.graph.action.AsyncEdgeAction;
import com.alibaba.cloud.ai.graph.action.AsyncNodeAction;
import com.alibaba.cloud.ai.graph.state.StateSnapshot;
import com.alibaba.cloud.ai.graph.state.strategy.ReplaceStrategy;
import com.nanoagent.service.config.LlmClientConfig;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.graph.skills.SkillRegistry;
import com.nanoagent.service.memory.UserMemoryManager;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class AgentWorkflowEngine {

    private static final Logger log = LoggerFactory.getLogger(AgentWorkflowEngine.class);

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;
    private final McpToolClient mcpToolClient;
    private final UserMemoryManager memoryManager;

    private CompiledGraph compiledGraph;
    private final Map<String, RunnableConfig> activeConfigs = new ConcurrentHashMap<>();

    public AgentWorkflowEngine(NanoAgentProperties properties, LlmClientConfig llmClientConfig,
                                McpToolClient mcpToolClient, UserMemoryManager memoryManager) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
        this.mcpToolClient = mcpToolClient;
        this.memoryManager = memoryManager;
    }

    @PostConstruct
    void init() {
        buildGraph();
    }

    private void buildGraph() {
        log.info("Building StateGraph workflow...");

        try {
            StateGraph graph = new StateGraph();

            // ── 添加节点 ──────────────────────────────────────────────────────
            graph.addNode("memory_retriever",
                    AsyncNodeAction.node_async(new RetrieveMemoryNode(memoryManager, properties)));
            graph.addNode("orchestrator",
                    AsyncNodeAction.node_async(new OrchestratorNode(properties, llmClientConfig)));
            graph.addNode("data_analyst",
                    AsyncNodeAction.node_async(new DataAnalystNode(properties, llmClientConfig)));
            graph.addNode("reporter",
                    AsyncNodeAction.node_async(new ReporterNode(properties, llmClientConfig)));
            graph.addNode("assistant",
                    AsyncNodeAction.node_async(new AssistantNode(properties, llmClientConfig)));
            graph.addNode("high_risk_tools",
                    AsyncNodeAction.node_async(new HighRiskToolsNode(mcpToolClient)));
            graph.addNode("safe_tools",
                    AsyncNodeAction.node_async(new SafeToolsNode(mcpToolClient)));

            // ── 添加边 ──────────────────────────────────────────────────────

            // 入口
            graph.addEdge(StateGraph.START, "memory_retriever");
            graph.addEdge("memory_retriever", "orchestrator");

            // orchestrator 条件路由 → 三个 Agent
            graph.addConditionalEdges("orchestrator",
                    AsyncEdgeAction.edge_async(new OrchestratorRouter()),
                    Map.of("DATA_ANALYST", "data_analyst",
                           "REPORTER", "reporter",
                           "ASSISTANT", "assistant",
                           "FINISH", StateGraph.END));

            // data_analyst 条件路由 → high_risk_tools 或 END
            graph.addConditionalEdges("data_analyst",
                    AsyncEdgeAction.edge_async(new RouteAfterAnalyst()),
                    Map.of("HIGH_RISK_TOOLS", "high_risk_tools", "FINISH", StateGraph.END));

            // reporter 条件路由 → high_risk_tools 或 END
            graph.addConditionalEdges("reporter",
                    AsyncEdgeAction.edge_async(new RouteAfterReporter()),
                    Map.of("HIGH_RISK_TOOLS", "high_risk_tools", "FINISH", StateGraph.END));

            // assistant 条件路由 → safe_tools 或 END
            graph.addConditionalEdges("assistant",
                    AsyncEdgeAction.edge_async(new RouteAfterAssistant()),
                    Map.of("SAFE_TOOLS", "safe_tools", "FINISH", StateGraph.END));

            // high_risk_tools 回跳 → 根据 current_agent 回到对应 Worker
            graph.addConditionalEdges("high_risk_tools",
                    AsyncEdgeAction.edge_async(new RouteAfterHighRiskTools()),
                    Map.of("DATA_ANALYST", "data_analyst", "REPORTER", "reporter"));

            // safe_tools 回跳 → 回到 assistant
            graph.addEdge("safe_tools", "assistant");

            CompileConfig compileConfig = CompileConfig.builder()
                    .interruptBefore("high_risk_tools")
                    .build();

            compiledGraph = graph.compile(compileConfig);
            log.info("StateGraph workflow built successfully");
        } catch (com.alibaba.cloud.ai.graph.exception.GraphStateException e) {
            throw new RuntimeException("Failed to build StateGraph workflow", e);
        }
    }

    public Map<String, Object> runWorkflow(String userId, String threadId,
                                            Map<String, String> llmProfile, String query) {
        try {
            OverAllState initialState = buildInitialState(userId, threadId, llmProfile, query);
            registerStrategies(initialState);

            RunnableConfig config = RunnableConfig.builder().threadId(threadId).build();

            log.info("StateGraph invoke | user_id={} | thread_id={}", userId, threadId);

            Optional<OverAllState> result = compiledGraph.invoke(initialState, config);

            if (result.isPresent()) {
                OverAllState state = result.get();

                if (isInterruptedAt(state, config, "high_risk_tools")) {
                    Map<String, Object> interruptedResult = convertStateToResult(state, threadId);
                    interruptedResult.put("interruptedAt", "high_risk_tools");
                    activeConfigs.put(threadId, config);
                    log.info("StateGraph interrupted | thread_id={} | at=high_risk_tools", threadId);
                    return interruptedResult;
                }

                activeConfigs.remove(threadId);
                log.info("StateGraph completed | thread_id={}", threadId);
                return convertStateToResult(state, threadId);
            }

            return Map.of("thread_id", threadId, "messages", List.of());

        } catch (Exception e) {
            activeConfigs.remove(threadId);
            log.error("StateGraph error | thread_id={} | error={}", threadId, e.getMessage(), e);
            throw new RuntimeException("Workflow execution failed", e);
        }
    }

    public Map<String, Object> resumeWorkflow(String userId, String threadId,
                                               Map<String, String> llmProfile, String action) {
        RunnableConfig config = activeConfigs.get(threadId);
        if (config == null) {
            throw new IllegalStateException("No active workflow found for thread: " + threadId);
        }

        if ("reject".equalsIgnoreCase(action)) {
            activeConfigs.remove(threadId);
            return buildRejectResult(threadId);
        }

        try {
            OverAllState.HumanFeedback feedback = new OverAllState.HumanFeedback(Map.of(), "");

            log.info("StateGraph resume | thread_id={} | action={}", threadId, action);

            Optional<OverAllState> result = compiledGraph.resume(feedback, config);

            if (result.isPresent()) {
                activeConfigs.remove(threadId);
                log.info("StateGraph resume completed | thread_id={}", threadId);
                return convertStateToResult(result.get(), threadId);
            }

            return Map.of("thread_id", threadId, "messages", List.of());

        } catch (Exception e) {
            activeConfigs.remove(threadId);
            log.error("StateGraph resume error | thread_id={} | error={}", threadId, e.getMessage(), e);
            throw new RuntimeException("Resume execution failed", e);
        }
    }

    public String generateThreadId() {
        return UUID.randomUUID().toString();
    }

    private OverAllState buildInitialState(String userId, String threadId,
                                            Map<String, String> llmProfile, String query) {
        List<AgentState.Message> messages = new ArrayList<>();
        if (query != null && !query.isBlank()) {
            messages.add(AgentState.Message.builder()
                    .type(AgentState.Message.MessageType.HUMAN)
                    .content(query)
                    .build());
        }

        OverAllState state = new OverAllState();
        state.input(Map.of(
                "messages", messages,
                "userId", userId,
                "memoryContext", "",
                "currentAgent", "",
                "orchestratorContext", "",
                "llmProfile", llmProfile,
                "threadId", threadId,
                "orchestratorDecision", "",
                "query", query != null ? query : "",
                "activeSkill", ""
        ));
        return state;
    }

    private void registerStrategies(OverAllState state) {
        ReplaceStrategy replace = new ReplaceStrategy();
        state.registerKeyAndStrategy("messages", replace);
        state.registerKeyAndStrategy("userId", replace);
        state.registerKeyAndStrategy("memoryContext", replace);
        state.registerKeyAndStrategy("currentAgent", replace);
        state.registerKeyAndStrategy("orchestratorContext", replace);
        state.registerKeyAndStrategy("llmProfile", replace);
        state.registerKeyAndStrategy("threadId", replace);
        state.registerKeyAndStrategy("orchestratorDecision", replace);
        state.registerKeyAndStrategy("query", replace);
        state.registerKeyAndStrategy("activeSkill", replace);
    }

    private boolean isInterruptedAt(OverAllState state, RunnableConfig config, String nodeName) {
        StateSnapshot snapshot = compiledGraph.getState(config);
        return snapshot != null && snapshot.getNext() != null && snapshot.getNext().equals(nodeName);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> convertStateToResult(OverAllState state, String threadId) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("messages", state.value("messages").orElse(List.of()));
        result.put("memoryContext", state.value("memoryContext").orElse(""));
        result.put("currentAgent", state.value("currentAgent").orElse(""));
        result.put("orchestratorContext", state.value("orchestratorContext").orElse(""));
        result.put("thread_id", threadId);
        result.put("activeSkill", state.value("activeSkill").orElse(""));
        return result;
    }

    private Map<String, Object> buildRejectResult(String threadId) {
        List<AgentState.Message> rejectionMessages = new ArrayList<>();
        rejectionMessages.add(AgentState.Message.builder()
                .type(AgentState.Message.MessageType.HUMAN)
                .content("审批已拒绝：不发送该邮件。")
                .build());
        rejectionMessages.add(AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content("邮件发送请求已被用户拒绝。")
                .build());
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("messages", rejectionMessages);
        result.put("currentAgent", "reporter");
        result.put("thread_id", threadId);
        return result;
    }
}
