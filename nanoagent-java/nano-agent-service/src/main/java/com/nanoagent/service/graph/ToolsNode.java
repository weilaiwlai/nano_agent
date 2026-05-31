package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ToolsNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(ToolsNode.class);

    private final McpToolClient mcpToolClient;

    public ToolsNode(McpToolClient mcpToolClient) {
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
        List<AgentState.Message> newMessages = new ArrayList<>(messages);

        for (AgentState.ToolCall toolCall : lastMessage.getToolCalls()) {
            String toolName = toolCall.getName();
            Map<String, Object> args = toolCall.getArgs() instanceof Map ? (Map<String, Object>) toolCall.getArgs() : Map.of();

            log.info("Tool node executing | tool={} | user_id={}", toolName, userId);

            try {
                String result = executeTool(toolName, args, userId);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.TOOL)
                        .content(result)
                        .name(toolName)
                        .toolCallId(toolCall.getId())
                        .build());
                log.info("Tool node completed | tool={} | result_len={}", toolName, result.length());
            } catch (Exception e) {
                log.error("Tool node error | tool={} | error={}", toolName, e.getMessage());
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

    private String executeTool(String toolName, Map<String, Object> args, String userId) {
        return switch (toolName) {
            case "tool_query_database", "query_database" -> {
                String sql = (String) args.getOrDefault("sql", "");
                yield mcpToolClient.queryDatabaseTool(sql);
            }
            case "tool_get_current_time", "get_current_time" ->
                mcpToolClient.getCurrentTimeTool();
            case "tool_search", "search" -> {
                String query = (String) args.getOrDefault("query", "");
                yield mcpToolClient.searchTool(query);
            }
            case "tool_list_allowed_directories", "list_allowed_directories" ->
                mcpToolClient.listAllowedDirectoriesTool();
            case "tool_is_path_allowed", "is_path_allowed" -> {
                String path = (String) args.getOrDefault("path", "");
                yield mcpToolClient.callTool("is_path_allowed", Map.of("path", path));
            }
            case "tool_read_file", "read_file" -> {
                String path = (String) args.getOrDefault("path", "");
                yield mcpToolClient.readFileTool(path);
            }
            case "tool_write_file", "write_file" -> {
                String path = (String) args.getOrDefault("path", "");
                String content = (String) args.getOrDefault("content", "");
                yield mcpToolClient.writeFileTool(path, content);
            }
            case "tool_create_directory", "create_directory" -> {
                String path = (String) args.getOrDefault("path", "");
                yield mcpToolClient.callTool("create_directory", Map.of("path", path));
            }
            case "tool_move_file", "move_file" -> {
                String src = (String) args.getOrDefault("src", "");
                String dst = (String) args.getOrDefault("dst", "");
                yield mcpToolClient.callTool("move_file", Map.of("src", src, "dst", dst));
            }
            case "tool_edit_file", "edit_file" -> {
                Object edits = args.get("edits");
                yield mcpToolClient.callTool("edit_file", Map.of("path", args.getOrDefault("path", ""), "edits", edits != null ? edits : List.of()));
            }
            case "tool_upsert_user_setting", "upsert_user_setting" -> {
                String settingKey = (String) args.getOrDefault("setting_key", "");
                String settingValue = (String) args.getOrDefault("setting_value", "");
                yield mcpToolClient.upsertUserSettingTool(userId, settingKey, settingValue);
            }
            default -> {
                log.warn("Unknown tool requested: {}", toolName);
                yield "{\"status\":\"error\",\"tool\":\"" + toolName + "\",\"message\":\"Unknown tool\"}";
            }
        };
    }
}