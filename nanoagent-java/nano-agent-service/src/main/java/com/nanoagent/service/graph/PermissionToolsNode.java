package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PermissionToolsNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(PermissionToolsNode.class);

    private final McpToolClient mcpToolClient;

    public PermissionToolsNode(McpToolClient mcpToolClient) {
        this.mcpToolClient = mcpToolClient;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        if (messages.isEmpty()) {
            return Map.of("messages", messages);
        }

        AgentState.Message lastMessage = messages.get(messages.size() - 1);
        if (lastMessage.getType() != AgentState.Message.MessageType.AI || lastMessage.getToolCalls() == null || lastMessage.getToolCalls().isEmpty()) {
            return Map.of("messages", messages);
        }

        List<AgentState.Message> newMessages = new ArrayList<>(messages);

        for (AgentState.ToolCall toolCall : lastMessage.getToolCalls()) {
            String toolName = toolCall.getName();
            Map<String, Object> args = toolCall.getArgs() instanceof Map ? (Map<String, Object>) toolCall.getArgs() : Map.of();

            if ("tool_send_report".equals(toolName) || "send_report".equals(toolName)) {
                String email = (String) args.getOrDefault("email", "");
                String content = (String) args.getOrDefault("content", "");

                log.info("Permission tool executing | tool=send_report | email=***");

                try {
                    String result = mcpToolClient.sendReportTool(email, content);
                    newMessages.add(AgentState.Message.builder()
                            .type(AgentState.Message.MessageType.TOOL)
                            .content(result)
                            .name("tool_send_report")
                            .toolCallId(toolCall.getId())
                            .build());
                    log.info("Permission tool completed | tool=send_report");
                } catch (Exception e) {
                    log.error("Permission tool error | tool=send_report | error={}", e.getMessage());
                    newMessages.add(AgentState.Message.builder()
                            .type(AgentState.Message.MessageType.TOOL)
                            .content("{\"status\":\"error\",\"tool\":\"send_report\",\"message\":\"" + e.getMessage() + "\"}")
                            .name("tool_send_report")
                            .toolCallId(toolCall.getId())
                            .build());
                }
            } else {
                log.warn("Unknown permission tool: {}", toolName);
            }
        }

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("sender", "Reporter");
        return result;
    }
}