package com.nanoagent.service.model;

import jakarta.validation.constraints.NotBlank;

public class ChatResumeRequest {
    @NotBlank(message = "user_id 不能为空")
    private String userId;

    @NotBlank(message = "action 不能为空")
    private String action;

    private String sessionId;
    private String threadId;

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getThreadId() { return threadId; }
    public void setThreadId(String threadId) { this.threadId = threadId; }
}