package com.nanoagent.service.memory;

import com.nanoagent.service.config.NanoAgentProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class UserMemoryManager {

    private static final Logger log = LoggerFactory.getLogger(UserMemoryManager.class);

    private final ChromaClient chromaClient;
    private final EmbeddingService embeddingService;
    private final NanoAgentProperties properties;
    private final Map<String, List<Map<String, Object>>> localFallback;

    public UserMemoryManager(NanoAgentProperties properties, EmbeddingService embeddingService) {
        this.properties = properties;
        this.chromaClient = new ChromaClient("http://localhost:8001");
        this.embeddingService = embeddingService;
        this.localFallback = new ConcurrentHashMap<>();
    }

    public String savePreference(String userId, String preferenceText, Map<String, String> llmProfile) {
        String memoryId = UUID.randomUUID().toString();
        String collectionName = collectionName(userId);
        ensureCollection(collectionName);

        List<Double> embedding;
        try {
            embedding = embeddingService.embed(preferenceText, null, null, null);
        } catch (Exception e) {
            log.warn("Embedding failed, using local store: {}", e.getMessage());
            embedding = List.of();
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("user_id", userId);
        metadata.put("timestamp", Instant.now().toString());
        metadata.put("text_hash", sha256Hex(preferenceText));

        boolean success = chromaClient.addDocuments(
                collectionName,
                List.of(memoryId),
                List.of(preferenceText),
                embedding.isEmpty() ? null : List.of(embedding),
                metadata);

        if (!success) {
            log.warn("Chroma add failed, falling back to local store");
            localFallback.computeIfAbsent(userId, k -> new ArrayList<>())
                    .add(Map.of("memory_id", memoryId, "preference_text", preferenceText,
                            "timestamp", Instant.now().toString()));
        }

        return memoryId;
    }

    public String retrieveRelevantMemories(String userId, String query) {
        if (query == null || query.isBlank()) return "";

        String collectionName = collectionName(userId);

        List<Map<String, Object>> localItems = localFallback.getOrDefault(userId, Collections.emptyList());
        if (!localItems.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, Object> item : localItems) {
                sb.append("- ").append(item.getOrDefault("preference_text", "")).append("\n");
            }
            return sb.toString().trim();
        }

        try {
            if (!chromaClient.collectionExists(collectionName)) {
                return "";
            }

            List<Double> queryEmbedding;
            try {
                queryEmbedding = embeddingService.embed(query, null, null, null);
            } catch (Exception e) {
                log.warn("Query embedding failed: {}", e.getMessage());
                return "";
            }

            List<Map<String, Object>> results = chromaClient.query(collectionName, queryEmbedding, 5, null);

            if (results.isEmpty()) {
                return "";
            }

            StringBuilder sb = new StringBuilder();
            for (Map<String, Object> item : results) {
                String doc = String.valueOf(item.getOrDefault("document", ""));
                Double distance = (Double) item.getOrDefault("distance", 0.0);
                if (distance != null && distance <= 1.5) {
                    sb.append("- ").append(doc).append("\n");
                }
            }
            return sb.toString().trim();
        } catch (Exception e) {
            log.error("Error retrieving memories for user {}: {}", userId, e.getMessage());
            return "";
        }
    }

    public List<Map<String, Object>> listMemories(String userId, int limit) {
        String collectionName = collectionName(userId);
        List<Map<String, Object>> localItems = localFallback.getOrDefault(userId, Collections.emptyList());
        if (!localItems.isEmpty()) {
            if (localItems.size() > limit) {
                return localItems.subList(0, limit);
            }
            return new ArrayList<>(localItems);
        }

        try {
            return chromaClient.getDocuments(collectionName, limit, 0);
        } catch (Exception e) {
            return List.of();
        }
    }

    public boolean deleteMemory(String userId, String memoryId) {
        localFallback.getOrDefault(userId, Collections.emptyList())
                .removeIf(item -> memoryId.equals(item.get("memory_id")));

        try {
            return chromaClient.deleteDocuments(collectionName(userId), List.of(memoryId));
        } catch (Exception e) {
            return false;
        }
    }

    private void ensureCollection(String name) {
        if (!chromaClient.collectionExists(name)) {
            chromaClient.createCollection(name, Map.of("hnsw:space", "cosine"));
        }
    }

    private String collectionName(String userId) {
        return "nanoagent_user_memory_" + sanitizeName(userId);
    }

    private String sanitizeName(String userId) {
        return userId.toLowerCase().replaceAll("[^a-z0-9_-]", "_");
    }

    private String sha256Hex(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(text.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (Exception e) {
            return Integer.toHexString(text.hashCode());
        }
    }
}