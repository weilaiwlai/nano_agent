package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import com.nanoagent.service.config.LlmClientConfig;
import com.nanoagent.service.config.NanoAgentProperties;
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

public class SupervisorNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(SupervisorNode.class);

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;

    public SupervisorNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig) {
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

        log.info("Node start | supervisor_node | user_id={} | history_len={}", userId, history.size());

        String supervisorPrompt = Prompts.SUPERVISOR_ROUTER_PROMPT + "\n\n长期记忆上下文：\n" +
                (memoryContext.isBlank() ? "（无）" : memoryContext);

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        OpenAiChatModel chatModel = llmClientConfig.getOrCreateChatModel(
                llmProfile.get("api_key"), llmProfile.get("base_url"),
                llmProfile.get("model"), false);

        List<Message> promptMessages = new ArrayList<>();
        promptMessages.add(new SystemMessage(supervisorPrompt));
        for (AgentState.Message msg : history) {
            promptMessages.add(convertToSpringMessage(msg));
        }

        SupervisorDecision decision;
        try {
            ChatResponse response = chatModel.call(new Prompt(promptMessages));
            String decisionText = response.getResult().getOutput().getText();
            decision = SupervisorDecision.fromText(decisionText);
        } catch (Exception e) {
            log.error("Node error | supervisor_node | user_id={} | error={}", userId, e.getMessage());
            decision = SupervisorDecision.FINISH;
        }

        log.info("Node end | supervisor_node | user_id={} | decision={}", userId, decision);

        List<AgentState.Message> newMessages = new ArrayList<>(history);
        newMessages.add(AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content(decision.name())
                .build());

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("supervisorDecision", decision.name());
        result.put("sender", "Supervisor");
        return result;
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