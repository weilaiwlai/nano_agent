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
        List<AgentState.Message> history = sanitizeHistory(trimSupervisorDecision(messages));
        String memoryContext = (String) state.value("memoryContext").orElse("");

        log.info("Node start | assistant_node | user_id={} | history_len={}", userId, history.size());

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

        String systemPrompt = Prompts.ASSISTANT_PROMPT.formatted(properties.getEmailDraftTargetChars())
                + "\n你是一个智能助手，拥有专业的技能团队来帮助你解决问题。\n\n"
                + "可用的专家技能团队：\n" + skillListStr + "\n\n"
                + "重要规则：\n"
                + "1. 当用户的问题适合使用特定技能时，必须只返回要激活的技能的确切名称，不要包含任何其他文字。\n"
                + "2. 例如：如果需要旅行规划技能，只返回'travel-planning'，不要返回'生成的是 travel-planning'或类似文本。\n"
                + "3. 如果不需要特定技能，请直接回答用户的问题。\n\n"
                + "当用户提出'发邮件'诉求时，先生成可审阅草稿，不要直接执行发送\n\n"
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
            log.error("Node error | assistant_node | user_id={} | error={}", userId, e.getMessage());
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

        log.info("Node end | assistant_node | user_id={} | active_skill={}", userId, activeSkillName);

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("sender", "Assistant");
        result.put("activeSkill", activeSkillName != null ? activeSkillName : "");
        return result;
    }

    private List<AgentState.Message> sanitizeHistory(List<AgentState.Message> messages) {
        return MessageSanitizer.sanitizeForModel(messages, properties.getMaxModelHistoryMessages());
    }

    private List<AgentState.Message> trimSupervisorDecision(List<AgentState.Message> messages) {
        if (messages == null || messages.isEmpty()) return messages;
        AgentState.Message last = messages.get(messages.size() - 1);
        if (last.getType() == AgentState.Message.MessageType.AI) {
            SupervisorDecision decision = SupervisorDecision.fromText(last.getContent());
            if (decision == SupervisorDecision.ASSISTANT || decision == SupervisorDecision.KNOWLEDGE_WORKER || decision == SupervisorDecision.REPORTER) {
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