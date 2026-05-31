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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ReporterNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(ReporterNode.class);
    private static final Pattern EMAIL_PATTERN = Pattern.compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}");

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;

    public ReporterNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        String userId = (String) state.value("userId").orElse("");
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        List<AgentState.Message> history = trimSupervisorDecision(messages);
        String latestQuery = latestUserQuery(history);

        log.info("Node start | reporter_node | user_id={} | history_len={}", userId, history.size());

        if (hasRecentSendReportResult(history)) {
            for (int i = history.size() - 1; i >= 0; i--) {
                AgentState.Message msg = history.get(i);
                if (msg.getType() == AgentState.Message.MessageType.TOOL) {
                    String name = msg.getName() != null ? msg.getName().trim().toLowerCase() : "";
                    if ("tool_send_report".equals(name) || "send_report".equals(name)) {
                        String summary = buildReporterSuccessMessage(msg.getContent());
                        List<AgentState.Message> newMessages = new ArrayList<>(messages);
                        newMessages.add(AgentState.Message.builder()
                                .type(AgentState.Message.MessageType.AI)
                                .content(summary)
                                .build());
                        log.info("Node end | reporter_node | user_id={} | mode=post_send_summary", userId);
                        Map<String, Object> result = new HashMap<>();
                        result.put("messages", newMessages);
                        result.put("sender", "Reporter");
                        return result;
                    }
                }
            }
            List<AgentState.Message> newMessages = new ArrayList<>(messages);
            newMessages.add(AgentState.Message.builder()
                    .type(AgentState.Message.MessageType.AI)
                    .content("邮件发送流程已结束。")
                    .build());
            Map<String, Object> result = new HashMap<>();
            result.put("messages", newMessages);
            result.put("sender", "Reporter");
            return result;
        }

        try {
            boolean executeIntent = checkExplicitSendIntent(history, state);
            if (!executeIntent) {
                List<AgentState.Message> newMessages = new ArrayList<>(messages);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.AI)
                        .content("我已将本轮需求判定为'内容起草/普通对话'，不会直接发送邮件。如果你确认要发送，请明确回复：确认发送到 xxx@xxx.com。")
                        .build());
                log.info("Node end | reporter_node | user_id={} | reason=not_explicit_execute_intent", userId);
                Map<String, Object> result = new HashMap<>();
                result.put("messages", newMessages);
                result.put("sender", "Reporter");
                return result;
            }

            String email = extractFirstEmail(latestQuery);
            String content = extractReportContent(latestQuery, history);

            if (email == null || email.isBlank()) {
                List<AgentState.Message> newMessages = new ArrayList<>(messages);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.AI)
                        .content("我还没有拿到收件邮箱，请补充'发送到 xxx@xxx.com'。")
                        .build());
                Map<String, Object> result = new HashMap<>();
                result.put("messages", newMessages);
                result.put("sender", "Reporter");
                return result;
            }

            if (content == null || content.isBlank()) {
                List<AgentState.Message> newMessages = new ArrayList<>(messages);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.AI)
                        .content("当前没有可发送的正文。请先让我生成邮件草稿，然后再回复'确认发送到 xxx@xxx.com'。")
                        .build());
                Map<String, Object> result = new HashMap<>();
                result.put("messages", newMessages);
                result.put("sender", "Reporter");
                return result;
            }

            int softLimit = properties.getReportContentSoftLimit();
            String truncatedContent = content.length() > softLimit ? content.substring(0, softLimit) : content;

            Map<String, Object> toolArgs = new HashMap<>();
            toolArgs.put("email", email);
            toolArgs.put("content", truncatedContent);

            AgentState.ToolCall toolCall = AgentState.ToolCall.builder()
                    .id("call_" + System.currentTimeMillis())
                    .name("tool_send_report")
                    .args(toolArgs)
                    .build();

            List<AgentState.Message> newMessages = new ArrayList<>(messages);
            newMessages.add(AgentState.Message.builder()
                    .type(AgentState.Message.MessageType.AI)
                    .content("准备发送邮件到 " + maskEmail(email))
                    .toolCalls(List.of(toolCall))
                    .build());

            log.info("Node end | reporter_node | user_id={} | mode=prepare_send | email_masked={}",
                    userId, maskEmail(email));

            Map<String, Object> result = new HashMap<>();
            result.put("messages", newMessages);
            result.put("sender", "Reporter");
            return result;
        } catch (Exception e) {
            log.error("Node error | reporter_node | user_id={} | error={}", userId, e.getMessage());
            List<AgentState.Message> newMessages = new ArrayList<>(messages);
            newMessages.add(AgentState.Message.builder()
                    .type(AgentState.Message.MessageType.AI)
                    .content("报告执行节点处理失败，请稍后重试。")
                    .build());
            Map<String, Object> result = new HashMap<>();
            result.put("messages", newMessages);
            result.put("sender", "Reporter");
            return result;
        }
    }

    private boolean checkExplicitSendIntent(List<AgentState.Message> history, OverAllState state) {
        if (history.isEmpty()) return false;
        String latestQuery = latestUserQuery(history);
        if (latestQuery == null || latestQuery.isBlank()) return false;

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        try {
            OpenAiChatModel chatModel = llmClientConfig.getOrCreateChatModel(
                    llmProfile.get("api_key"), llmProfile.get("base_url"),
                    llmProfile.get("model"), false);

            List<Message> promptMessages = List.of(
                    new SystemMessage(Prompts.REPORT_EXECUTION_GUARD_PROMPT),
                    new UserMessage(latestQuery)
            );
            ChatResponse response = chatModel.call(new Prompt(promptMessages));
            String decision = response.getResult().getOutput().getText().trim().toUpperCase();
            return decision.contains("EXECUTE");
        } catch (Exception e) {
            log.warn("Send intent check failed | error={}", e.getMessage());
            return false;
        }
    }

    private String extractFirstEmail(String query) {
        if (query == null) return null;
        Matcher matcher = EMAIL_PATTERN.matcher(query);
        return matcher.find() ? matcher.group() : null;
    }

    private String extractReportContent(String query, List<AgentState.Message> history) {
        String cleaned = query != null ? cleanContentQuery(query) : null;
        if (cleaned != null && !cleaned.isBlank()) {
            return cleaned;
        }
        for (int i = history.size() - 1; i >= 0; i--) {
            AgentState.Message msg = history.get(i);
            if (msg.getType() == AgentState.Message.MessageType.AI) {
                String content = msg.getContent();
                if (content != null && !content.isBlank() && !isRoutingDecision(content)) {
                    return content;
                }
            }
        }
        return null;
    }

    private String cleanContentQuery(String query) {
        if (query == null) return null;

        Pattern contentMarker1 = Pattern.compile("(?:内容|正文)\\s*[：:]\\s*(.+)$", Pattern.DOTALL);
        Matcher m1 = contentMarker1.matcher(query);
        if (m1.find()) {
            String extracted = m1.group(1).trim();
            if (!extracted.isBlank()) return extracted;
        }

        Pattern contentMarker2 = Pattern.compile("(?:发送|发给|发到).{0,30}(?:内容|正文)\\s*(?:是|为)\\s*(.+)$", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
        Matcher m2 = contentMarker2.matcher(query);
        if (m2.find()) {
            String extracted = m2.group(1).trim();
            if (!extracted.isBlank()) return extracted;
        }

        String[] markers = {"确认发送", "发送到", "send to", "发送邮件"};
        for (String marker : markers) {
            int idx = query.toLowerCase().indexOf(marker.toLowerCase());
            if (idx >= 0) {
                String after = query.substring(idx + marker.length()).trim();
                Matcher m = EMAIL_PATTERN.matcher(after);
                if (m.find()) {
                    String content = after.substring(m.end()).trim();
                    if (!content.isBlank()) return content;
                }
            }
        }
        return null;
    }

    private boolean isRoutingDecision(String content) {
        if (content == null) return false;
        SupervisorDecision decision = SupervisorDecision.fromText(content);
        return decision != SupervisorDecision.FINISH || content.contains("KnowledgeWorker") || content.contains("Reporter") || content.contains("Assistant");
    }

    private String buildReporterSuccessMessage(String toolResult) {
        if (toolResult == null || toolResult.isBlank()) {
            return "邮件已成功发送！";
        }
        if (toolResult.contains("\"status\":\"success\"") || toolResult.contains("\"status\": \"success\"")) {
            return "邮件已成功发送！";
        }
        return "邮件发送完成，结果如下：\n" + toolResult;
    }

    private boolean hasRecentSendReportResult(List<AgentState.Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            AgentState.Message msg = messages.get(i);
            if (msg.getType() == AgentState.Message.MessageType.TOOL) {
                String name = msg.getName() != null ? msg.getName().trim().toLowerCase() : "";
                if ("tool_send_report".equals(name) || "send_report".equals(name)) {
                    return true;
                }
            }
        }
        return false;
    }

    private String latestUserQuery(List<AgentState.Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i).getType() == AgentState.Message.MessageType.HUMAN) {
                return messages.get(i).getContent() != null ? messages.get(i).getContent().trim() : "";
            }
        }
        return "";
    }

    private List<AgentState.Message> trimSupervisorDecision(List<AgentState.Message> messages) {
        if (messages == null || messages.isEmpty()) return messages;
        AgentState.Message last = messages.get(messages.size() - 1);
        if (last.getType() == AgentState.Message.MessageType.AI) {
            SupervisorDecision decision = SupervisorDecision.fromText(last.getContent());
            if (decision == SupervisorDecision.REPORTER || decision == SupervisorDecision.KNOWLEDGE_WORKER || decision == SupervisorDecision.ASSISTANT) {
                List<AgentState.Message> trimmed = new ArrayList<>(messages);
                trimmed.remove(trimmed.size() - 1);
                return trimmed;
            }
        }
        return messages;
    }

    private String maskEmail(String email) {
        if (email == null || !email.contains("@")) return "***";
        String[] parts = email.split("@", 2);
        String local = parts[0];
        int keep = local.length() < 4 ? 1 : 2;
        return local.substring(0, keep) + "***@" + parts[1];
    }
}