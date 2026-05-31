package com.nanoagent.service.model;

public class MemoryDeleteResponse {
    private String userId;
    private String memoryId;
    private String status;
    private String message;

    public MemoryDeleteResponse() {}

    public MemoryDeleteResponse(String userId, String memoryId, String status, String message) {
        this.userId = userId;
        this.memoryId = memoryId;
        this.status = status;
        this.message = message;
    }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getMemoryId() { return memoryId; }
    public void setMemoryId(String memoryId) { this.memoryId = memoryId; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String userId;
        private String memoryId;
        private String status;
        private String message;

        public Builder userId(String userId) { this.userId = userId; return this; }
        public Builder memoryId(String memoryId) { this.memoryId = memoryId; return this; }
        public Builder status(String status) { this.status = status; return this; }
        public Builder message(String message) { this.message = message; return this; }
        public MemoryDeleteResponse build() { return new MemoryDeleteResponse(userId, memoryId, status, message); }
    }
}