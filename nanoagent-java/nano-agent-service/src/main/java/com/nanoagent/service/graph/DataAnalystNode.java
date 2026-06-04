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
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DataAnalystNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(DataAnalystNode.class);

    private final NanoAgentProperties properties;
    private final LlmClientConfig llmClientConfig;

    public DataAnalystNode(NanoAgentProperties properties, LlmClientConfig llmClientConfig) {
        this.properties = properties;
        this.llmClientConfig = llmClientConfig;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        String userId = (String) state.value("userId").orElse("");
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        List<AgentState.Message> history = sanitizeHistory(trimOrchestratorDecision(messages));
        String memoryContext = (String) state.value("memoryContext").orElse("");
        String orchestratorContext = (String) state.value("orchestratorContext").orElse("");
        String latestQuery = latestUserQuery(history);

        log.info("Node start | data_analyst | user_id={} | history_len={}", userId, history.size());

        if (hasDatabaseIntent(latestQuery) && !hasSqlSnippet(latestQuery)) {
            String helpAnswer = buildDatabaseHelpAnswer();
            List<AgentState.Message> newMessages = new ArrayList<>(messages);
            newMessages.add(AgentState.Message.builder()
                    .type(AgentState.Message.MessageType.AI)
                    .content(helpAnswer)
                    .build());
            Map<String, Object> result = new HashMap<>();
            result.put("messages", newMessages);
            result.put("currentAgent", "data_analyst");
            return result;
        }

        String systemPrompt = String.format(Prompts.ANALYST_PROMPT, Prompts.BUSINESS_DB_SCHEMA)
                + "\n\n长期记忆上下文：\n" + (memoryContext.isBlank() ? "（无）" : memoryContext)
                + (orchestratorContext.isBlank() ? "" : "\n\n任务描述：\n" + orchestratorContext);

        Map<String, String> llmProfile = (Map<String, String>) state.value("llmProfile").orElse(Map.of());
        String content;
        List<AgentState.ToolCall> toolCalls = null;

        try {
            OpenAiApi openAiApi = OpenAiApi.builder()
                    .apiKey(llmProfile.get("api_key"))
                    .baseUrl(llmProfile.get("base_url"))
                    .build();

            OpenAiChatOptions options = OpenAiChatOptions.builder()
                    .model(llmProfile.get("model"))
                    .temperature(0.0)
                    .tools(getToolDefinitions())
                    .build();

            OpenAiChatModel chatModel = OpenAiChatModel.builder()
                    .openAiApi(openAiApi)
                    .defaultOptions(options)
                    .build();

            List<Message> promptMessages = new ArrayList<>();
            promptMessages.add(new SystemMessage(systemPrompt));
            for (AgentState.Message msg : history) {
                promptMessages.add(convertToSpringMessage(msg));
            }

            ChatResponse response = chatModel.call(new Prompt(promptMessages));
            content = response.getResult().getOutput().getText();

            toolCalls = extractToolCalls(response);
            if (toolCalls != null && !toolCalls.isEmpty()) {
                log.info("DataAnalyst requested {} tool calls", toolCalls.size());
            }
        } catch (Exception e) {
            log.error("Node error | data_analyst | user_id={} | error={}", userId, e.getMessage());
            content = "数据分析节点处理失败，请稍后重试。";
        }

        List<AgentState.Message> newMessages = new ArrayList<>(messages);
        AgentState.Message.MessageBuilder aiBuilder = AgentState.Message.builder()
                .type(AgentState.Message.MessageType.AI)
                .content(content != null ? content : "");
        if (toolCalls != null && !toolCalls.isEmpty()) {
            aiBuilder.toolCalls(toolCalls);
        }
        newMessages.add(aiBuilder.build());

        log.info("Node end | data_analyst | user_id={} | tool_calls={}",
                userId, toolCalls != null ? toolCalls.size() : 0);

        Map<String, Object> result = new HashMap<>();
        result.put("messages", newMessages);
        result.put("currentAgent", "data_analyst");
        return result;
    }

    private List<AgentState.ToolCall> extractToolCalls(ChatResponse response) {
        if (response == null || response.getResult() == null) return null;
        var output = response.getResult().getOutput();
        if (output instanceof AssistantMessage) {
            AssistantMessage am = (AssistantMessage) output;
            var springToolCalls = am.getToolCalls();
            if (springToolCalls != null && !springToolCalls.isEmpty()) {
                List<AgentState.ToolCall> toolCalls = new ArrayList<>();
                for (var tc : springToolCalls) {
                    AgentState.ToolCall toolCall = AgentState.ToolCall.builder()
                            .id(tc.id())
                            .name(tc.name())
                            .args(tc.arguments())
                            .build();
                    toolCalls.add(toolCall);
                }
                return toolCalls;
            }
        }
        return null;
    }

    private List<OpenAiApi.FunctionTool> getToolDefinitions() {
        List<OpenAiApi.FunctionTool> tools = new ArrayList<>();

        // data_analyst 只绑定数据库查询和时间工具
        tools.add(buildToolDef("tool_query_database", "执行SQL查询数据库",
                Map.of("type", "object", "properties",
                        Map.of("sql", Map.of("type", "string", "description", "要执行的SQL查询语句"))),
                List.of("sql")));

        tools.add(buildToolDef("tool_get_current_time", "获取当前时间",
                Map.of("type", "object", "properties", Map.of()),
                List.of()));

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

    private String latestUserQuery(List<AgentState.Message> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i).getType() == AgentState.Message.MessageType.HUMAN) {
                return messages.get(i).getContent() != null ? messages.get(i).getContent().trim() : "";
            }
        }
        return "";
    }

    private List<AgentState.Message> sanitizeHistory(List<AgentState.Message> messages) {
        return MessageSanitizer.sanitizeForModel(messages, properties.getMaxModelHistoryMessages());
    }

    private List<AgentState.Message> trimOrchestratorDecision(List<AgentState.Message> messages) {
        if (messages == null || messages.isEmpty()) return messages;
        AgentState.Message last = messages.get(messages.size() - 1);
        if (last.getType() == AgentState.Message.MessageType.AI) {
            OrchestratorDecision decision = OrchestratorDecision.fromText(last.getContent());
            if (decision == OrchestratorDecision.DATA_ANALYST || decision == OrchestratorDecision.REPORTER || decision == OrchestratorDecision.ASSISTANT) {
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
            case AI -> msg.getToolCalls() != null && !msg.getToolCalls().isEmpty()
                    ? new AssistantMessage(msg.getContent() != null ? msg.getContent() : "", Map.of(), List.of())
                    : new AssistantMessage(msg.getContent() != null ? msg.getContent() : "");
            case SYSTEM -> new SystemMessage(msg.getContent() != null ? msg.getContent() : "");
            default -> new UserMessage(msg.getContent() != null ? msg.getContent() : "");
        };
    }

    private boolean hasDatabaseIntent(String query) {
        if (query == null) return false;
        String lower = query.toLowerCase();
        String[] keywords = {"数据库", "查库", "sql", "postgres", "postgre", "table", "表", "字段", "schema", "库里", "db"};
        for (String kw : keywords) {
            if (lower.contains(kw)) return true;
        }
        return false;
    }

    private boolean hasSqlSnippet(String query) {
        if (query == null) return false;
        String lower = query.toLowerCase();
        String[] patterns = {"select ", "with ", "from ", "where ", "join ", "group by", "order by", "limit ", "count("};
        for (String p : patterns) {
            if (lower.contains(p)) return true;
        }
        return false;
    }

    private String buildDatabaseHelpAnswer() {
        return """
                检测到数据库查询意图，但未提供具体 SQL 语句。

                可用的查询示例：
                ```sql
                SELECT * FROM information_schema.tables WHERE table_schema = 'public';
                SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'your_table';
                SELECT * FROM your_table LIMIT 10;
                ```

                请提供具体的 SQL 查询语句，我将为你执行。""";
    }
}
