package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import com.nanoagent.service.config.LlmClientConfig;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.graph.skills.AgentSkill;
import com.nanoagent.service.graph.skills.SkillRegistry;
import com.nanoagent.service.graph.skills.SkillTools;
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
import java.util.stream.Collectors;

public class AssistantNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(AssistantNode.class);

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;
    private final SkillRegistry skillRegistry;

    public AssistantNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
        this.skillRegistry = new SkillRegistry();
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        String userId = (String) state.value("userId").orElse("");
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        List<AgentState.Message> history = sanitizeHistory(trimOrchestratorDecision(messages));
        String memoryContext = (String) state.value("memoryContext").orElse("");

        log.info("Node start | assistant | user_id={} | history_len={}", userId, history.size());

        skillRegistry.refresh();
        List<Map<String, String>> skills = skillRegistry.listSkills();
        if (skills.isEmpty()) {
            log.info("No skills available");
        } else {
            log.info("Available skills: {}",
                    skills.stream().map(s -> s.get("name")).collect(Collectors.joining(", ")));
        }

        String skillListStr = skills.stream()
                .map(s -> "- " + s.get("name") + ": " + s.get("description"))
                .collect(Collectors.joining("\n"));

        String systemPrompt = Prompts.ASSISTANT_PROMPT
                + "\n\n可用的专家技能团队：\n" + skillListStr + "\n\n"
                + "长期记忆上下文：\n" + (memoryContext.isBlank() ? "（无）" : memoryContext);

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        OpenAiChatModel chatModel = llmClientConfig.getOrCreateChatModel(
                llmProfile.get("api_key"), llmProfile.get("base_url"),
                llmProfile.get("model"), true);

        List<Message> promptMessages = new ArrayList<>();
        promptMessages.add(new SystemMessage(systemPrompt));
        for (AgentState.Message msg : history) {
            promptMessages.add(convertToSpringMessage(msg));
        }

        String content;
        try {
            ChatResponse response = chatModel.call(new Prompt(promptMessages));
            content = response.getResult().getOutput().getText();
        } catch (Exception e) {
            log.error("Node error | assistant | user_id={} | error={}", userId, e.getMessage());
            content = "助手节点处理失败，请稍后重试。";
        }

        String activeSkillName = null;
        String trimmedContent = content != null ? content.trim() : "";

        AgentSkill matchedSkill = skillRegistry.getSkill(trimmedContent);
        if (matchedSkill != null) {
            SkillTools.setActivePath(matchedSkill.getRootPath());
            activeSkillName = matchedSkill.getName();
            log.info("Skill activated: {}", activeSkillName);
        }

        List<AgentState.Message> newMessages = new ArrayList<>(messages);
        newMessages.add(AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content(content)
                .build());

        log.info("Node end | assistant | user_id={} | active_skill={}", userId, activeSkillName);

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("currentAgent", "assistant");
        result.put("activeSkill", activeSkillName != null ? activeSkillName : "");
        return result;
    }

    private List<AgentState.Message> sanitizeHistory(List<AgentState.Message> messages) {
        return MessageSanitizer.sanitizeForModel(messages, properties.getMaxModelHistoryMessages());
    }

    private List<AgentState.Message> trimOrchestratorDecision(List<AgentState.Message> messages) {
        if (messages == null || messages.isEmpty()) return messages;
        AgentState.Message last = messages.get(messages.size() - 1);
        if (last.getType() == AgentState.Message.MessageType.AI) {
            OrchestratorDecision decision = OrchestratorDecision.fromText(last.getContent());
            if (decision == OrchestratorDecision.ASSISTANT || decision == OrchestratorDecision.DATA_ANALYST || decision == OrchestratorDecision.REPORTER) {
                List<AgentState.Message> trimmed = new ArrayList<>(messages);
                trimmed.remove(trimmed.size() - 1);
                return trimmed;
            }
        }
        return messages;
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
