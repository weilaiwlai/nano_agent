package com.nanoagent.service.memory;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanoagent.service.config.NanoAgentProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class EmbeddingService {

    private static final Logger log = LoggerFactory.getLogger(EmbeddingService.class);

    private final NanoAgentProperties properties;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public EmbeddingService(NanoAgentProperties properties) {
        this.properties = properties;
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    public List<Double> embed(String text, String apiKey, String baseUrl, String model) {
        String effectiveApiKey = apiKey != null ? apiKey : System.getenv("OPENAI_API_KEY");
        String effectiveBaseUrl = baseUrl != null ? baseUrl :
                System.getenv().getOrDefault("OPENAI_BASE_URL", "https://api.openai.com/v1");
        String effectiveModel = model != null ? model :
                properties.getEmbedding().getDefaultModel();

        if (effectiveApiKey == null || effectiveApiKey.isBlank()) {
            log.warn("No API key available for embedding");
            return List.of();
        }

        String url = effectiveBaseUrl;
        if (!url.endsWith("/")) url += "/";
        url += "embeddings";

        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("input", text);
            body.put("model", effectiveModel);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setBearerAuth(effectiveApiKey);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            Map result = restTemplate.postForObject(url, request, Map.class);

            if (result != null && result.containsKey("data")) {
                List<Map<String, Object>> data = (List<Map<String, Object>>) result.get("data");
                if (data != null && !data.isEmpty()) {
                    Object embedding = data.get(0).get("embedding");
                    if (embedding instanceof List) {
                        List<Number> numbers = (List<Number>) embedding;
                        List<Double> doubles = new ArrayList<>();
                        for (Number n : numbers) {
                            doubles.add(n.doubleValue());
                        }
                        return doubles;
                    }
                }
            }
        } catch (Exception e) {
            log.error("Embedding error: {}", e.getMessage());
        }
        return List.of();
    }
}