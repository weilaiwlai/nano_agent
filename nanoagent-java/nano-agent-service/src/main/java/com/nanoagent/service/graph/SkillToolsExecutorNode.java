package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import com.nanoagent.service.graph.skills.SkillTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SkillToolsExecutorNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(SkillToolsExecutorNode.class);

    private final McpToolClient mcpToolClient;

    public SkillToolsExecutorNode(McpToolClient mcpToolClient) {
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

            log.info("Skill tool executing | tool={}", toolName);

            try {
                String result = executeSkillTool(toolName, args);
                newMessages.add(AgentState.Message.builder()
                        .type(AgentState.Message.MessageType.TOOL)
                        .content(result)
                        .name(toolName)
                        .toolCallId(toolCall.getId())
                        .build());
                log.info("Skill tool completed | tool={} | result_len={}", toolName, result.length());
            } catch (Exception e) {
                log.error("Skill tool error | tool={} | error={}", toolName, e.getMessage());
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

    private String executeSkillTool(String toolName, Map<String, Object> args) {
        return switch (toolName) {
            case "run_skill_script" -> {
                String scriptName = (String) args.getOrDefault("script_name", "");
                List<String> scriptArgs = new ArrayList<>();
                Object rawArgs = args.get("args");
                if (rawArgs instanceof List<?> list) {
                    for (Object item : list) {
                        if (item != null) scriptArgs.add(item.toString());
                    }
                }
                yield SkillTools.runSkillScript(scriptName, scriptArgs);
            }
            case "read_reference" -> {
                String filename = (String) args.getOrDefault("filename", "");
                yield SkillTools.readReference(filename);
            }
            default -> {
                log.warn("Unknown skill tool: {}", toolName);
                yield "{\"status\":\"error\",\"tool\":\"" + toolName + "\",\"message\":\"Unknown skill tool\"}";
            }
        };
    }
}