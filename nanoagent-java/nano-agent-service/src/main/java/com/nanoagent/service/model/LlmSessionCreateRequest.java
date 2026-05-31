package com.nanoagent.service.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class LlmSessionCreateRequest {
    private String provider;

    @NotBlank(message = "api_key 不能为空")
    @Size(min = 10, message = "api_key 长度至少为10")
    private String apiKey;

    @NotBlank(message = "model 不能为空")
    private String model;

    private String baseUrl;
    private String embeddingModel;
    private Integer ttlSeconds;

    public String getProvider() { return provider; }
    public void setProvider(String provider) { this.provider = provider; }
    public String getApiKey() { return apiKey; }
    public void setApiKey(String apiKey) { this.apiKey = apiKey; }
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public String getBaseUrl() { return baseUrl; }
    public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
    public String getEmbeddingModel() { return embeddingModel; }
    public void setEmbeddingModel(String embeddingModel) { this.embeddingModel = embeddingModel; }
    public Integer getTtlSeconds() { return ttlSeconds; }
    public void setTtlSeconds(Integer ttlSeconds) { this.ttlSeconds = ttlSeconds; }
}