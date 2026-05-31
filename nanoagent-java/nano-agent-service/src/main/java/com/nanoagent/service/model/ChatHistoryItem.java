package com.nanoagent.service.model;

public class ChatHistoryItem {
    private String role;
    private String content;
    private String timestamp;

    public ChatHistoryItem() {}

    public ChatHistoryItem(String role, String content, String timestamp) {
        this.role = role;
        this.content = content;
        this.timestamp = timestamp;
    }

    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String role;
        private String content;
        private String timestamp;

        public Builder role(String role) { this.role = role; return this; }
        public Builder content(String content) { this.content = content; return this; }
        public Builder timestamp(String timestamp) { this.timestamp = timestamp; return this; }
        public ChatHistoryItem build() { return new ChatHistoryItem(role, content, timestamp); }
    }
}