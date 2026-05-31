package com.nanoagent.service.model;

import java.util.List;

public class LlmSessionValidateResponse {
    private String provider;
    private String model;
    private String baseUrl;
    private String embeddingModel;
    private boolean chatOk;
    private boolean embeddingOk;
    private List<String> errors;

    public LlmSessionValidateResponse() {}

    public LlmSessionValidateResponse(String provider, String model, String baseUrl, String embeddingModel, boolean chatOk, boolean embeddingOk, List<String> errors) {
        this.provider = provider;
        this.model = model;
        this.baseUrl = baseUrl;
        this.embeddingModel = embeddingModel;
        this.chatOk = chatOk;
        this.embeddingOk = embeddingOk;
        this.errors = errors;
    }

    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public String getBaseUrl() { return baseUrl; }
    public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
    public String getEmbeddingModel() { return embeddingModel; }
    public void setEmbeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; }
    public boolean isChatOk() { return chatOk; }
    public void setChatOk(boolean chatOk) { this.chatOk = chatOk; }
    public boolean isEmbeddingOk() { return embeddingOk; }
    public void setEmbeddingOk(boolean embeddingOk) { this.embeddingOk = embeddingOk; }
    public List<String> getErrors() { return errors; }
    public void setErrors(List<String> errors) { this.errors = errors; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String provider;
        private String model;
        private String baseUrl;
        private String embeddingModel;
        private boolean chatOk;
        private boolean embeddingOk;
        private List<String> errors;

        public Builder provider(String provider) { this.provider = provider; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder embeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; return this; }
        public Builder chatOk(boolean chatOk) { this.chatOk = chatOk; return this; }
        public Builder embeddingOk(boolean embeddingOk) { this.embeddingOk = embeddingOk; return this; }
        public Builder errors(List<String> errors) { this.errors = errors; return this; }
        public LlmSessionValidateResponse build() { return new LlmSessionValidateResponse(provider, model, baseUrl, embeddingModel, chatOk, embeddingOk, errors); }
    }
}