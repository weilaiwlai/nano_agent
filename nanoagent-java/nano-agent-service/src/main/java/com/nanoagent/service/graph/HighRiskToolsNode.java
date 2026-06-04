package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class HighRiskToolsNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(HighRiskToolsNode.class);

    private final McpToolClient mcpToolClient;

    public HighRiskToolsNode(McpToolClient mcpToolClient) {
        this.mcpToolClient = mcpToolClient;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        if (messages.isEmpty()) {
            Map<String, Object> result = new HashMap<>();
            result.put("messages", messages);
            return result;
        }

        AgentState.Message lastMessage = messages.get(messages.size() - 1);
        if (lastMessage.getType() != AgentState.Message.MessageType.AI || lastMessage.getToolCalls() == null || lastMessage.getToolCalls().isEmpty()) {
            return Map.of("messages", messages);
        }

        String userId = (String) state.value("userId").orElse("");
        String currentAgent = (String) state.value("currentAgent").orElse("data_analyst");
        List<AgentState.Message> newMessages = new ArrayList<>(messages);

        for (AgentState.ToolCall toolCall : lastMessage.getToolCalls()) {
            String toolName = toolCall.getName();
            Map<String, Object> args = toolCall.getArgs() instanceof Map ? (Map<String, Object>) toolCall.getArgs() : Map.of();

            log.info("High risk tool executing | tool={} | user_id={} | current_agent={}", toolName, userId, currentAgent);

            try {
                String result = executeHighRiskTool(toolName, args, userId);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.TOOL)
                        .content(result)
                        .name(toolName)
                        .toolCallId(toolCall.getId())
                        .build());
                log.info("High risk tool completed | tool={} | result_len={}", toolName, result.length());
            } catch (Exception e) {
                log.error("High risk tool error | tool={} | error={}", toolName, e.getMessage());
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.TOOL)
                        .content("{\"status\":\"error\",\"tool\":\"" + toolName + "\",\"message\":\"" + e.getMessage() + "\"}")
                        .name(toolName)
                        .toolCallId(toolCall.getId())
                        .build());
            }
        }

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        return result;
    }

    private String executeHighRiskTool(String toolName, Map<String, Object> args, String userId) {
        return switch (toolName) {
            case "tool_query_database", "query_database" -> {
                String sql = (String) args.getOrDefault("sql", "");
                yield mcpToolClient.queryDatabaseTool(sql);
            }
            case "tool_send_report", "send_report" -> {
                String email = (String) args.getOrDefault("email", "");
                String content = (String) args.getOrDefault("content", "");
                yield mcpToolClient.sendReportTool(email, content);
            }
            default -> {
                log.warn("Unknown high risk tool requested: {}", toolName);
                yield "{\"status\":\"error\",\"tool\":\"" + toolName + "\",\"message\":\"Unknown high risk tool\"}";
            }
        };
    }
}
