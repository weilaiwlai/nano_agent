package com.nanoagent.service.graph;

import java.util.ArrayList;
import java.util.List;

public class AgentState {

    private List<Message> messages = new ArrayList<>();
    private String userId;
    private String memoryContext;
    private String currentAgent;           // 当前活跃的 agent："data_analyst" | "reporter" | "assistant"
    private String orchestratorContext;    // orchestrator 输出的任务描述，传递给下游 Worker

    public AgentState() {}

    public AgentState(List<Message> messages, String userId, String memoryContext,
                      String currentAgent, String orchestratorContext) {
        this.messages = messages != null ? messages : new ArrayList<>();
        this.userId = userId;
        this.memoryContext = memoryContext;
        this.currentAgent = currentAgent;
        this.orchestratorContext = orchestratorContext;
    }

    public List<Message> getMessages() { return messages; }
    public void setMessages(List<Message> messages) { this.messages = messages; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getMemoryContext() { return memoryContext; }
    public void setMemoryContext(String memoryContext) { this.memoryContext = memoryContext; }
    public String getCurrentAgent() { return currentAgent; }
    public void setCurrentAgent(String currentAgent) { this.currentAgent = currentAgent; }
    public String getOrchestratorContext() { return orchestratorContext; }
    public void setOrchestratorContext(String orchestratorContext) { this.orchestratorContext = orchestratorContext; }

    public static AgentStateBuilder builder() {
        return new AgentStateBuilder();
    }

    public static class AgentStateBuilder {
        private List<Message> messages = new ArrayList<>();
        private String userId;
        private String memoryContext;
        private String currentAgent;
        private String orchestratorContext;

        public AgentStateBuilder messages(List<Message> messages) { this.messages = messages; return this; }
        public AgentStateBuilder userId(String userId) { this.userId = userId; return this; }
        public AgentStateBuilder memoryContext(String memoryContext) { this.memoryContext = memoryContext; return this; }
        public AgentStateBuilder currentAgent(String currentAgent) { this.currentAgent = currentAgent; return this; }
        public AgentStateBuilder orchestratorContext(String orchestratorContext) { this.orchestratorContext = orchestratorContext; return this; }

        public AgentState build() {
            return new AgentState(messages, userId, memoryContext, currentAgent, orchestratorContext);
        }
    }

    public static class Message {
        private MessageType type;
        private String content;
        private String name;
        private List<ToolCall> toolCalls;
        private String toolCallId;

        public enum MessageType {
            HUMAN, AI, SYSTEM, TOOL
        }

        public Message() {}

        public Message(MessageType type, String content, String name, List<ToolCall> toolCalls, String toolCallId) {
            this.type = type;
            this.content = content;
            this.name = name;
            this.toolCalls = toolCalls;
            this.toolCallId = toolCallId;
        }

        public MessageType getType() { return type; }
        public void setType(MessageType type) { this.type = type; }
        public String getContent() { return content; }
        public void setContent(String content) { this.content = content; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public List<ToolCall> getToolCalls() { return toolCalls; }
        public void setToolCalls(List<ToolCall> toolCalls) { this.toolCalls = toolCalls; }
        public String getToolCallId() { return toolCallId; }
        public void setToolCallId(String toolCallId) { this.toolCallId = toolCallId; }

        public static MessageBuilder builder() {
            return new MessageBuilder();
        }

        public static class MessageBuilder {
            private MessageType type;
            private String content;
            private String name;
            private List<ToolCall> toolCalls;
            private String toolCallId;

            public MessageBuilder type(MessageType type) { this.type = type; return this; }
            public MessageBuilder content(String content) { this.content = content; return this; }
            public MessageBuilder name(String name) { this.name = name; return this; }
            public MessageBuilder toolCalls(List<ToolCall> toolCalls) { this.toolCalls = toolCalls; return this; }
            public MessageBuilder toolCallId(String toolCallId) { this.toolCallId = toolCallId; return this; }

            public Message build() {
                return new Message(type, content, name, toolCalls, toolCallId);
            }
        }
    }

    public static class ToolCall {
        private String id;
        private String name;
        private Object args;

        public ToolCall() {}

        public ToolCall(String id, String name, Object args) {
            this.id = id;
            this.name = name;
            this.args = args;
        }

        public String getId() { return id; }
        public void setId(String id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public Object getArgs() { return args; }
        public void setArgs(Object args) { this.args = args; }

        public static ToolCallBuilder builder() {
            return new ToolCallBuilder();
        }

        public static class ToolCallBuilder {
            private String id;
            private String name;
            private Object args;

            public ToolCallBuilder id(String id) { this.id = id; return this; }
            public ToolCallBuilder name(String name) { this.name = name; return this; }
            public ToolCallBuilder args(Object args) { this.args = args; return this; }

            public ToolCall build() {
                return new ToolCall(id, name, args);
            }
        }
    }
}