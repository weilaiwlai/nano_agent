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
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SkillsToolsNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(SkillsToolsNode.class);

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;
    private final SkillRegistry skillRegistry;

    public SkillsToolsNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig, SkillRegistry skillRegistry) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
        this.skillRegistry = skillRegistry;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        if (messages.isEmpty()) {
            return Map.of("messages", messages);
        }

        String activeSkillName = (String) state.value("activeSkill").orElse("");
        AgentSkill skill = skillRegistry.getSkill(activeSkillName);

        log.info("Skills tools node start | active_skill={}", activeSkillName);

        String systemText = "You are a helpful AI assistant.";
        if (skill != null) {
            SkillTools.setActivePath(skill.getRootPath());
            systemText += "\n\n=== ACTIVE SKILL: " + skill.getName() + " ===\n" + skill.getInstructions();

            java.nio.file.Path refDir = skill.getRootPath().resolve("references");
            if (java.nio.file.Files.exists(refDir) && java.nio.file.Files.isDirectory(refDir)) {
                try {
                    List<String> refFiles = new ArrayList<>();
                    java.nio.file.Files.list(refDir)
                            .filter(f -> java.nio.file.Files.isRegularFile(f))
                            .filter(f -> {
                                String name = f.getFileName().toString();
                                return !name.startsWith(".");
                            })
                            .forEach(f -> refFiles.add(f.getFileName().toString()));
                    if (!refFiles.isEmpty()) {
                        systemText += "\n\n=== AVAILABLE REFERENCES (Knowledge Base) ===\n";
                        systemText += "You have access to the following files in the 'references' folder:\n";
                        for (String f : refFiles) {
                            systemText += "- " + f + "\n";
                        }
                        systemText += "Use the `read_reference` tool to read their content if needed.\n";
                    }
                } catch (Exception ignored) {}
            }

            log.debug("Injecting instructions for {}", skill.getName());
        } else {
            SkillTools.setActivePath(null);
        }

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        String content = "";
        List<AgentState.ToolCall> toolCalls = null;

        try {
            OpenAiApi openAiApi = OpenAiApi.builder()
                    .apiKey(llmProfile.get("api_key"))
                    .baseUrl(llmProfile.get("base_url"))
                    .build();

            OpenAiChatOptions options = OpenAiChatOptions.builder()
                    .model(llmProfile.get("model"))
                    .temperature(0.0)
                    .tools(getSkillToolDefinitions())
                    .build();

            OpenAiChatModel chatModel = OpenAiChatModel.builder()
                    .openAiApi(openAiApi)
                    .defaultOptions(options)
                    .build();

            List<Message> fullMessages = new ArrayList<>();
            fullMessages.add(new SystemMessage(systemText));
            for (AgentState.Message msg : messages) {
                fullMessages.add(convertToSpringMessage(msg));
            }

            ChatResponse response = chatModel.call(new Prompt(fullMessages));
            var output = response.getResult().getOutput();
            content = output.getText();

            if (output instanceof AssistantMessage) {
                AssistantMessage am = (AssistantMessage) output;
                var springToolCalls = am.getToolCalls();
                if (springToolCalls != null && !springToolCalls.isEmpty()) {
                    toolCalls = new ArrayList<>();
                    for (var tc : springToolCalls) {
                        AgentState.ToolCall toolCall = AgentState.ToolCall.builder()
                                .id(tc.id())
                                .name(tc.name())
                                .args(tc.arguments())
                                .build();
                        toolCalls.add(toolCall);
                    }
                    log.info("Skills agent requested {} tool calls", toolCalls.size());
                }
            }
        } catch (Exception e) {
            log.error("Skills tools node error | skill={} | error={}", activeSkillName, e.getMessage());
            content = "技能工具节点处理失败，请稍后重试。";
        }

        List<AgentState.Message> newMessages = new ArrayList<>(messages);
        AgentState.Message.MessageBuilder aiBuilder = AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content(content != null ? content : "");
        if (toolCalls != null && !toolCalls.isEmpty()) {
            aiBuilder.toolCalls(toolCalls);
        }
        newMessages.add(aiBuilder.build());

        log.info("Skills tools node end | active_skill={} | tool_calls={}",
                activeSkillName, toolCalls != null ? toolCalls.size() : 0);

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("sender", "Assistant");
        return result;
    }

    private List<OpenAiApi.FunctionTool> getSkillToolDefinitions() {
        List<OpenAiApi.FunctionTool> tools = new ArrayList<>();

        tools.add(buildToolDef("run_skill_script",
                "Execute a Python script in the 'scripts' folder of the active skill.",
                Map.of("type", "object", "properties",
                        Map.of("script_name", Map.of("type", "string", "description", "Filename (e.g., 'magic.py')"),
                                "args", Map.of("type", "array", "items", Map.of("type", "string"),
                                        "description", "Arguments"))),
                List.of("script_name")));

        tools.add(buildToolDef("read_reference",
                "Read the content of a reference file located in the 'references' folder of the active skill.",
                Map.of("type", "object", "properties",
                        Map.of("filename", Map.of("type", "string",
                                "description", "The name of the file to read (e.g., 'guidelines.md')"))),
                List.of("filename")));

        return tools;
    }

    private OpenAiApi.FunctionTool buildToolDef(String name, String description,
                                                  Map<String, Object> parameters, List<String> required) {
        Map<String, Object> functionDef = new HashMap<>();
        functionDef.put("name", name);
        functionDef.put("description", description);
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("required", required);
        functionDef.put("parameters", params);

        return new OpenAiApi.FunctionTool(new OpenAiApi.FunctionTool.Function(
                name, description, objectMapperValue(params)));
    }

    private String objectMapperValue(Map<String, Object> params) {
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper().writeValueAsString(params);
        } catch (Exception e) {
            return "{}";
        }
    }

    private Message convertToSpringMessage(AgentState.Message msg) {
        return switch (msg.getType()) {
            case HUMAN -> new UserMessage(msg.getContent() != null ? msg.getContent() : "");
            case AI -> msg.getToolCalls() != null && !msg.getToolCalls().isEmpty()
                    ? new AssistantMessage(msg.getContent() != null ? msg.getContent() : "", Map.of(), List.of())
                    : new AssistantMessage(msg.getContent() != null ? msg.getContent() : "");
            case SYSTEM -> new SystemMessage(msg.getContent() != null ? msg.getContent() : "");
            case TOOL -> new AssistantMessage(msg.getContent() != null ? msg.getContent() : "");
            default -> new UserMessage(msg.getContent() != null ? msg.getContent() : "");
        };
    }
}