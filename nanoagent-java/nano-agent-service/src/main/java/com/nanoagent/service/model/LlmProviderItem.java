package com.nanoagent.service.model;

public class LlmProviderItem {
    private String provider;
    private String label;
    private boolean requiresBaseUrl;
    private String defaultBaseUrl;
    private String defaultEmbeddingModel;

    public LlmProviderItem() {}

    public LlmProviderItem(String provider, String label, boolean requiresBaseUrl, String defaultBaseUrl, String defaultEmbeddingModel) {
        this.provider = provider;
        this.label = label;
        this.requiresBaseUrl = requiresBaseUrl;
        this.defaultBaseUrl = defaultBaseUrl;
        this.defaultEmbeddingModel = defaultEmbeddingModel;
    }

    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }
    public boolean isRequiresBaseUrl() { return requiresBaseUrl; }
    public void setRequiresBaseUrl(boolean requiresBaseUrl) { this.requiresBaseUrl = requiresBaseUrl; }
    public String getDefaultBaseUrl() { return defaultBaseUrl; }
    public void setDefaultBaseUrl(String defaultBaseUrl) { this.defaultBaseUrl = defaultBaseUrl; }
    public String getDefaultEmbeddingModel() { return defaultEmbeddingModel; }
    public void setDefaultEmbeddingModel(String defaultEmbeddingModel) { this.defaultEmbeddingModel = defaultEmbeddingModel; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String provider;
        private String label;
        private boolean requiresBaseUrl;
        private String defaultBaseUrl;
        private String defaultEmbeddingModel;

        public Builder provider(String provider) { this.provider = provider; return this; }
        public Builder label(String label) { this.label = label; return this; }
        public Builder requiresBaseUrl(boolean requiresBaseUrl) { this.requiresBaseUrl = requiresBaseUrl; return this; }
        public Builder defaultBaseUrl(String defaultBaseUrl) { this.defaultBaseUrl = defaultBaseUrl; return this; }
        public Builder defaultEmbeddingModel(String defaultEmbeddingModel) { this.defaultEmbeddingModel = defaultEmbeddingModel; return this; }
        public LlmProviderItem build() { return new LlmProviderItem(provider, label, requiresBaseUrl, defaultBaseUrl, defaultEmbeddingModel); }
    }
}