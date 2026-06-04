package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import com.nanoagent.service.config.LlmClientConfig;
import com.nanoagent.service.config.NanoAgentProperties;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OrchestratorNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(OrchestratorNode.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;

    public OrchestratorNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        String userId = (String) state.value("userId").orElse("");
        List<AgentState.Message> history = sanitizeHistory(
                (List<AgentState.Message>) state.value("messages").orElse(List.of()));
        String memoryContext = (String) state.value("memoryContext").orElse("");

        log.info("Node start | orchestrator | user_id={} | history_len={}", userId, history.size());

        String orchestratorPrompt = Prompts.ORCHESTRATOR_PROMPT + "\n\n长期记忆上下文：\n" +
                (memoryContext.isBlank() ? "（无）" : memoryContext);

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        OpenAiChatModel chatModel = llmClientConfig.getOrCreateChatModel(
                llmProfile.get("api_key"), llmProfile.get("base_url"),
                llmProfile.get("model"), false);

        List<Message> promptMessages = new ArrayList<>();
        promptMessages.add(new SystemMessage(orchestratorPrompt));
        for (AgentState.Message msg : history) {
            promptMessages.add(convertToSpringMessage(msg));
        }

        String route;
        String taskSummary;
        try {
            ChatResponse response = chatModel.call(new Prompt(promptMessages));
            String responseText = response.getResult().getOutput().getText();
            log.info("Orchestrator raw response: {}", responseText);

            // 解析JSON响应
            JsonNode jsonNode = objectMapper.readTree(responseText.trim());
            route = jsonNode.has("route") ? jsonNode.get("route").asText() : "FINISH";
            taskSummary = jsonNode.has("task_summary") ? jsonNode.get("task_summary").asText() : "";
        } catch (Exception e) {
            log.error("Node error | orchestrator | user_id={} | error={}", userId, e.getMessage());
            route = "FINISH";
            taskSummary = "";
        }

        // 标准化路由
        String normalizedRoute = normalizeRoute(route);
        log.info("Node end | orchestrator | user_id={} | route={} | task_summary={}",
                userId, normalizedRoute, taskSummary);

        List<AgentState.Message> newMessages = new ArrayList<>(history);
        newMessages.add(AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content(normalizedRoute)
                .build());

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("orchestratorDecision", normalizedRoute);
        result.put("orchestratorContext", taskSummary);
        result.put("currentAgent", normalizedRoute.toLowerCase());
        return result;
    }

    private String normalizeRoute(String route) {
        if (route == null) return "FINISH";
        String lower = route.trim().toLowerCase();
        if (lower.contains("data_analyst") || lower.contains("dataanalyst") || lower.contains("knowledge_worker")) {
            return "data_analyst";
        }
        if (lower.contains("reporter")) {
            return "reporter";
        }
        if (lower.contains("assistant")) {
            return "assistant";
        }
        if (lower.contains("finish")) {
            return "FINISH";
        }
        return "FINISH";
    }

    private List<AgentState.Message> sanitizeHistory(List<AgentState.Message> messages) {
        return MessageSanitizer.sanitizeForModel(messages, properties.getMaxModelHistoryMessages());
    }

    private Message convertToSpringMessage(AgentState.Message msg) {
        return switch (msg.getType()) {
            case HUMAN -> new UserMessage(msg.getContent() != null ? msg.getContent() : "");
            case AI -> new AssistantMessage(msg.getContent() != null ? msg.getContent() : "");
            case SYSTEM -> new SystemMessage(msg.getContent() != null ? msg.getContent() : "");
            default -> new UserMessage(msg.getContent() != null ? msg.getContent() : "");
        };
    }
}
