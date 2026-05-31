package com.nanoagent.service.model;

import java.util.List;

public class MemoryListResponse {
    private String userId;
    private List<MemoryItem> items;

    public MemoryListResponse() {}

    public MemoryListResponse(String userId, List<MemoryItem> items) {
        this.userId = userId;
        this.items = items;
    }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public List<MemoryItem> getItems() { return items; }
    public void setItems(List<MemoryItem> items) { this.items = items; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String userId;
        private List<MemoryItem> items;

        public Builder userId(String userId) { this.userId = userId; return this; }
        public Builder items(List<MemoryItem> items) { this.items = items; return this; }
        public MemoryListResponse build() { return new MemoryListResponse(userId, items); }
    }
}