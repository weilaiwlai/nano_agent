package com.nanoagent.service.model;

public class MemoryResponse {
    private String userId;
    private String status;
    private String message;
    private String memoryId;

    public MemoryResponse() {}

    public MemoryResponse(String userId, String status, String message, String memoryId) {
        this.userId = userId;
        this.status = status;
        this.message = message;
        this.memoryId = memoryId;
    }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public String getMemoryId() { return memoryId; }
    public void setMemoryId(String memoryId) { this.memoryId = memoryId; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String userId;
        private String status;
        private String message;
        private String memoryId;

        public Builder userId(String userId) { this.userId = userId; return this; }
        public Builder status(String status) { this.status = status; return this; }
        public Builder message(String message) { this.message = message; return this; }
        public Builder memoryId(String memoryId) { this.memoryId = memoryId; return this; }
        public MemoryResponse build() { return new MemoryResponse(userId, status, message, memoryId); }
    }
}