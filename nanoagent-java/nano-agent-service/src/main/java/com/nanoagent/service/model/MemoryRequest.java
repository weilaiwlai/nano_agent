package com.nanoagent.service.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class MemoryRequest {
    @NotBlank(message = "user_id 不能为空")
    private String userId;

    @NotBlank(message = "preference_text 不能为空")
    private String preferenceText;

    private String sessionId;

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getPreferenceText() { return preferenceText; }
    public void setPreferenceText(String preferenceText) { this.preferenceText = preferenceText; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
}