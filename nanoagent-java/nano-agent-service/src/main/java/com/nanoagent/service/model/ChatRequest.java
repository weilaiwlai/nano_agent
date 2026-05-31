package com.nanoagent.service.model;

import jakarta.validation.constraints.NotBlank;

public class ChatRequest {
    @NotBlank(message = "user_id 不能为空")
    private String userId;

    @NotBlank(message = "query 不能为空")
    private String query;

    private String sessionId;
    private String threadId;

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getThreadId() { return threadId; }
    public void setThreadId(String threadId) { this.threadId = threadId; }
}