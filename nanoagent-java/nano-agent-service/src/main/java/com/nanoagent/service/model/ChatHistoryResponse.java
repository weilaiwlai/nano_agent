package com.nanoagent.service.model;

import java.util.List;

public class ChatHistoryResponse {
    private String threadId;
    private String userId;
    private List<ChatHistoryItem> messages;

    public ChatHistoryResponse() {}

    public ChatHistoryResponse(String threadId, String userId, List<ChatHistoryItem> messages) {
        this.threadId = threadId;
        this.userId = userId;
        this.messages = messages;
    }

    public String getThreadId() { return threadId; }
    public void setThreadId(String threadId) { this.threadId = threadId; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public List<ChatHistoryItem> getMessages() { return messages; }
    public void setMessages(List<ChatHistoryItem> messages) { this.messages = messages; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String threadId;
        private String userId;
        private List<ChatHistoryItem> messages;

        public Builder threadId(String threadId) { this.threadId = threadId; return this; }
        public Builder userId(String userId) { this.userId = userId; return this; }
        public Builder messages(List<ChatHistoryItem> messages) { this.messages = messages; return this; }
        public ChatHistoryResponse build() { return new ChatHistoryResponse(threadId, userId, messages); }
    }
}