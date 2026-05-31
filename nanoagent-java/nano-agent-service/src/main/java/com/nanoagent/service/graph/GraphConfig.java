package com.nanoagent.service.graph;

import java.util.Map;

public class GraphConfig {

    private String userId;
    private String threadId;
    private String query;
    private Map<String, String> llmProfile;
    private Map<String, Object> metadata;

    public GraphConfig() {}

    public GraphConfig(String userId, String threadId, String query, Map<String, String> llmProfile, Map<String, Object> metadata) {
        this.userId = userId;
        this.threadId = threadId;
        this.query = query;
        this.llmProfile = llmProfile;
        this.metadata = metadata;
    }

    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getThreadId() { return threadId; }
    public void setThreadId(String threadId) { this.threadId = threadId; }
    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
    public Map<String, String> getLlmProfile() { return llmProfile; }
    public void setLlmProfile(Map<String, String> llmProfile) { this.llmProfile = llmProfile; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }

    public static GraphConfigBuilder builder() {
        return new GraphConfigBuilder();
    }

    public static class GraphConfigBuilder {
        private String userId;
        private String threadId;
        private String query;
        private Map<String, String> llmProfile;
        private Map<String, Object> metadata;

        public GraphConfigBuilder userId(String userId) { this.userId = userId; return this; }
        public GraphConfigBuilder threadId(String threadId) { this.threadId = threadId; return this; }
        public GraphConfigBuilder query(String query) { this.query = query; return this; }
        public GraphConfigBuilder llmProfile(Map<String, String> llmProfile) { this.llmProfile = llmProfile; return this; }
        public GraphConfigBuilder metadata(Map<String, Object> metadata) { this.metadata = metadata; return this; }

        public GraphConfig build() {
            return new GraphConfig(userId, threadId, query, llmProfile, metadata);
        }
    }
}