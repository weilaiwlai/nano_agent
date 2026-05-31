package com.nanoagent.service.model;

public class LlmSessionCreateResponse {
    private String sessionId;
    private String provider;
    private int expiresIn;
    private String model;
    private String baseUrl;
    private String embeddingModel;

    public LlmSessionCreateResponse() {}

    public LlmSessionCreateResponse(String sessionId, String provider, int expiresIn, String model, String baseUrl, String embeddingModel) {
        this.sessionId = sessionId;
        this.provider = provider;
        this.expiresIn = expiresIn;
        this.model = model;
        this.baseUrl = baseUrl;
        this.embeddingModel = embeddingModel;
    }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public int getExpiresIn() { return expiresIn; }
    public void setExpiresIn(int expiresIn) { this.expiresIn = expiresIn; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public String getBaseUrl() { return baseUrl; }
    public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
    public String getEmbeddingModel() { return embeddingModel; }
    public void setEmbeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String sessionId;
        private String provider;
        private int expiresIn;
        private String model;
        private String baseUrl;
        private String embeddingModel;

        public Builder sessionId(String sessionId) { this.sessionId = sessionId; return this; }
        public Builder provider(String provider) { this.provider = provider; return this; }
        public Builder expiresIn(int expiresIn) { this.expiresIn = expiresIn; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder embeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; return this; }
        public LlmSessionCreateResponse build() { return new LlmSessionCreateResponse(sessionId, provider, expiresIn, model, baseUrl, embeddingModel); }
    }
}