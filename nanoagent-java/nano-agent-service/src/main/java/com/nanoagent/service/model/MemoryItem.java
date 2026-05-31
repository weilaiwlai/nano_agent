package com.nanoagent.service.model;

public class MemoryItem {
    private String memoryId;
    private String preferenceText;
    private String timestamp;

    public MemoryItem() {}

    public MemoryItem(String memoryId, String preferenceText, String timestamp) {
        this.memoryId = memoryId;
        this.preferenceText = preferenceText;
        this.timestamp = timestamp;
    }

    public String getMemoryId() { return memoryId; }
    public void setMemoryId(String memoryId) { this.memoryId = memoryId; }
    public String getPreferenceText() { return preferenceText; }
    public void setPreferenceText(String preferenceText) { this.preferenceText = preferenceText; }
    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String memoryId;
        private String preferenceText;
        private String timestamp;

        public Builder memoryId(String memoryId) { this.memoryId = memoryId; return this; }
        public Builder preferenceText(String preferenceText) { this.preferenceText = preferenceText; return this; }
        public Builder timestamp(String timestamp) { this.timestamp = timestamp; return this; }
        public MemoryItem build() { return new MemoryItem(memoryId, preferenceText, timestamp); }
    }
}