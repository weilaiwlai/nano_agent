package com.nanoagent.service.model;

public class LlmSessionDeleteResponse {
    private String sessionId;
    private String status;
    private String message;

    public LlmSessionDeleteResponse() {}

    public LlmSessionDeleteResponse(String sessionId, String status, String message) {
        this.sessionId = sessionId;
        this.status = status;
        this.message = message;
    }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String sessionId;
        private String status;
        private String message;

        public Builder sessionId(String sessionId) { this.sessionId = sessionId; return this; }
        public Builder status(String status) { this.status = status; return this; }
        public Builder message(String message) { this.message = message; return this; }
        public LlmSessionDeleteResponse build() { return new LlmSessionDeleteResponse(sessionId, status, message); }
    }
}