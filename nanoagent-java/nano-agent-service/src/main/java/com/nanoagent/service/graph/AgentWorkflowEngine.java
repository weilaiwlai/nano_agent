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

            graph.addNode("retrieve_memory",
                    AsyncNodeAction.node_async(new RetrieveMemoryNode(memoryManager, properties)));
            graph.addNode("supervisor",
                    AsyncNodeAction.node_async(new SupervisorNode(properties, llmClientConfig)));
            graph.addNode("knowledge_worker",
                    AsyncNodeAction.node_async(new KnowledgeWorkerNode(properties, llmClientConfig)));
            graph.addNode("tools",
                    AsyncNodeAction.node_async(new ToolsNode(mcpToolClient)));
            graph.addNode("reporter",
                    AsyncNodeAction.node_async(new ReporterNode(properties, llmClientConfig)));
            graph.addNode("permission_tools",
                    AsyncNodeAction.node_async(new PermissionToolsNode(mcpToolClient)));
            graph.addNode("assistant",
                    AsyncNodeAction.node_async(new AssistantNode(properties, llmClientConfig)));
            graph.addNode("skills_tools",
                    AsyncNodeAction.node_async(new SkillsToolsNode(properties, llmClientConfig, new SkillRegistry())));
            graph.addNode("skill_tools_executor",
                    AsyncNodeAction.node_async(new SkillToolsExecutorNode(mcpToolClient)));

            graph.addEdge(StateGraph.START, "retrieve_memory");
            graph.addEdge("retrieve_memory", "supervisor");

            graph.addConditionalEdges("supervisor",
                    AsyncEdgeAction.edge_async(new SupervisorRouter()),
                    Map.of("KNOWLEDGE_WORKER", "knowledge_worker",
                           "REPORTER", "reporter",
                           "ASSISTANT", "assistant",
                           "FINISH", StateGraph.END));

            graph.addConditionalEdges("knowledge_worker",
                    AsyncEdgeAction.edge_async(new RouteAfterKnowledgeWorker()),
                    Map.of("TOOLS", "tools", "FINISH", StateGraph.END));
            graph.addEdge("tools", "knowledge_worker");

            graph.addConditionalEdges("reporter",
                    AsyncEdgeAction.edge_async(new RouteAfterReporter()),
                    Map.of("PERMISSION_TOOLS", "permission_tools", "FINISH", StateGraph.END));
            graph.addEdge("permission_tools", "reporter");

            graph.addConditionalEdges("assistant",
                    AsyncEdgeAction.edge_async(new RouteAfterAssistant()),
                    Map.of("SKILLS_TOOLS", "skills_tools", "FINISH", StateGraph.END));
            graph.addConditionalEdges("skills_tools",
                    AsyncEdgeAction.edge_async(new RouteAfterSkillsTools()),
                    Map.of("SKILL_TOOLS_EXECUTOR", "skill_tools_executor",
                           "ASSISTANT", "assistant"));
            graph.addEdge("skill_tools_executor", "skills_tools");

            CompileConfig compileConfig = CompileConfig.builder()
                    .interruptBefore("permission_tools")
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

                if (isInterruptedAt(state, config, "permission_tools")) {
                    Map<String, Object> interruptedResult = convertStateToResult(state, threadId);
                    interruptedResult.put("interruptedAt", "permission_tools");
                    activeConfigs.put(threadId, config);
                    log.info("StateGraph interrupted | thread_id={} | at=permission_tools", threadId);
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
                "sender", "",
                "llmProfile", llmProfile,
                "threadId", threadId,
                "supervisorDecision", "",
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
        state.registerKeyAndStrategy("sender", replace);
        state.registerKeyAndStrategy("llmProfile", replace);
        state.registerKeyAndStrategy("threadId", replace);
        state.registerKeyAndStrategy("supervisorDecision", replace);
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
        result.put("sender", state.value("sender").orElse(""));
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
        result.put("sender", "Reporter");
        result.put("thread_id", threadId);
        return result;
    }
}