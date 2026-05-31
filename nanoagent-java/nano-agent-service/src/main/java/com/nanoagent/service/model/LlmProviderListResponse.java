package com.nanoagent.service.model;

import java.util.List;

public class LlmProviderListResponse {
    private List<LlmProviderItem> items;

    public LlmProviderListResponse() {}

    public LlmProviderListResponse(List<LlmProviderItem> items) {
        this.items = items;
    }

    public List<LlmProviderItem> getItems() { return items; }
    public void setItems(List<LlmProviderItem> items) { this.items = items; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private List<LlmProviderItem> items;

        public Builder items(List<LlmProviderItem> items) { this.items = items; return this; }
        public LlmProviderListResponse build() { return new LlmProviderListResponse(items); }
    }
}