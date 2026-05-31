package com.nanoagent.service.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Configuration
public class LlmClientConfig {

    private final Map<String, OpenAiChatModel> chatModelCache = new ConcurrentHashMap<>();
    private final Map<String, OpenAiChatModel> nonStreamChatModelCache = new ConcurrentHashMap<>();

    public OpenAiChatModel getOrCreateChatModel(String apiKey, String baseUrl, String model, boolean streaming) {
        String cacheKey = model + "|" + baseUrl + "|" + apiKey + "|" + streaming;
        Map<String, OpenAiChatModel> cache = streaming ? chatModelCache : nonStreamChatModelCache;

        return cache.computeIfAbsent(cacheKey, key -> {
            OpenAiApi openAiApi = OpenAiApi.builder()
                    .apiKey(apiKey)
                    .baseUrl(baseUrl)
                    .build();

            OpenAiChatOptions options = OpenAiChatOptions.builder()
                    .model(model)
                    .temperature(0.0)
                    .build();

            return OpenAiChatModel.builder()
                    .openAiApi(openAiApi)
                    .defaultOptions(options)
                    .build();
        });
    }

    public ChatClient createChatClient(OpenAiChatModel chatModel) {
        return ChatClient.builder(chatModel).build();
    }
}