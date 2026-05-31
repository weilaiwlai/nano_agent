package com.nanoagent.mcp.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    private final ObjectMapper objectMapper;
    private final Map<String, List<RagDocument>> documentStore;
    private final Map<String, Map<String, Double>> bm25Idf;

    public RagService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        this.documentStore = new ConcurrentHashMap<>();
        this.bm25Idf = new ConcurrentHashMap<>();
    }

    public Mono<String> ingest(String collection, List<String> documents, List<Map<String, Object>> metadatas) {
        return Mono.fromCallable(() -> {
            List<RagDocument> docs = new ArrayList<>();
            for (int i = 0; i < documents.size(); i++) {
                String id = UUID.randomUUID().toString();
                Map<String, Object> meta = (metadatas != null && i < metadatas.size()) ? metadatas.get(i) : Map.of();
                docs.add(new RagDocument(id, documents.get(i), meta));
            }
            documentStore.computeIfAbsent(collection, k -> new ArrayList<>()).addAll(docs);
            rebuildBm25Index(collection);
            return jsonResponseSync(Map.of("status", "success", "ingested", docs.size()));
        }).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
    }

    public Mono<String> hybridSearch(String collection, String query, int topK) {
        return Mono.fromCallable(() -> {
            List<RagDocument> docs = documentStore.getOrDefault(collection, List.of());
            if (docs.isEmpty()) {
                return jsonResponseSync(Map.of("status", "ok", "results", List.of(), "count", 0));
            }

            List<ScoredDocument> semanticResults;
            try {
                semanticResults = semanticSearch(docs, query, topK * 2);
            } catch (Exception e) {
                log.warn("Semantic search failed, using BM25 only: {}", e.getMessage());
                semanticResults = List.of();
            }

            List<ScoredDocument> bm25Results = bm25Search(collection, query, topK * 2);

            List<ScoredDocument> fused = rrfFusion(List.of(semanticResults, bm25Results), 60);

            String cohereApiKey = System.getenv("COHERE_API_KEY");
            List<ScoredDocument> finalResults;
            if (cohereApiKey != null && !cohereApiKey.isBlank() && !fused.isEmpty()) {
                finalResults = rerankWithCohere(fused, query, cohereApiKey, topK);
            } else {
                finalResults = fused.subList(0, Math.min(topK, fused.size()));
            }

            List<Map<String, Object>> response = new ArrayList<>();
            for (ScoredDocument sd : finalResults) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("id", sd.doc.id);
                item.put("content", sd.doc.content);
                item.put("score", sd.score);
                item.put("metadata", sd.doc.metadata);
                response.add(item);
            }

            return jsonResponseSync(Map.of("status", "ok", "results", response, "count", response.size(),
                    "recall_sources", Map.of("semantic", semanticResults.size(), "bm25", bm25Results.size(),
                            "fused", fused.size(), "final", response.size())));
        }).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
    }

    private List<ScoredDocument> semanticSearch(List<RagDocument> docs, String query, int topK) {
        String apiKey = System.getenv("OPENAI_API_KEY");
        String baseUrl = System.getenv().getOrDefault("OPENAI_BASE_URL", "https://api.openai.com/v1");
        if (apiKey == null || apiKey.isBlank()) return List.of();

        try {
            List<Double> queryVec = getEmbedding(query, apiKey, baseUrl);
            if (queryVec.isEmpty()) return List.of();

            List<ScoredDocument> results = new ArrayList<>();
            for (RagDocument doc : docs) {
                String cacheKey = doc.id;
                List<Double> docVec = getEmbedding(doc.content, apiKey, baseUrl);
                if (!docVec.isEmpty()) {
                    double similarity = cosineSimilarity(queryVec, docVec);
                    results.add(new ScoredDocument(doc, similarity));
                }
            }
            results.sort(Comparator.comparingDouble((ScoredDocument s) -> s.score).reversed());
            return results.subList(0, Math.min(topK, results.size()));
        } catch (Exception e) {
            return List.of();
        }
    }

    private List<Double> getEmbedding(String text, String apiKey, String baseUrl) {
        try {
            WebClient client = WebClient.builder()
                    .baseUrl(baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl)
                    .defaultHeader("Authorization", "Bearer " + apiKey)
                    .defaultHeader("Content-Type", "application/json")
                    .build();

            Map<String, Object> body = Map.of("input", text, "model", "text-embedding-v3");
            Map response = client.post().uri("/embeddings").bodyValue(body)
                    .retrieve().bodyToMono(Map.class).block();

            if (response != null && response.containsKey("data")) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> data = (List<Map<String, Object>>) response.get("data");
                if (!data.isEmpty()) {
                    @SuppressWarnings("unchecked")
                    List<Number> embedding = (List<Number>) data.get(0).get("embedding");
                    List<Double> result = new ArrayList<>();
                    for (Number n : embedding) result.add(n.doubleValue());
                    return result;
                }
            }
        } catch (Exception ignored) {}
        return List.of();
    }

    private double cosineSimilarity(List<Double> a, List<Double> b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < Math.min(a.size(), b.size()); i++) {
            dot += a.get(i) * b.get(i);
            normA += a.get(i) * a.get(i);
            normB += b.get(i) * b.get(i);
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
    }

    private List<ScoredDocument> bm25Search(String collection, String query, int topK) {
        List<RagDocument> docs = documentStore.getOrDefault(collection, List.of());
        if (docs.isEmpty()) return List.of();

        Map<String, Double> idf = bm25Idf.getOrDefault(collection, Map.of());
        String[] queryTerms = tokenize(query);
        double k1 = 1.2, b = 0.75;
        double avgdl = docs.stream().mapToInt(d -> tokenize(d.content).length).average().orElse(1.0);

        List<ScoredDocument> results = new ArrayList<>();
        for (RagDocument doc : docs) {
            String[] docTerms = tokenize(doc.content);
            int docLen = docTerms.length;
            Map<String, Long> tf = new HashMap<>();
            for (String t : docTerms) tf.merge(t, 1L, Long::sum);

            double score = 0;
            for (String qt : queryTerms) {
                double idfVal = idf.getOrDefault(qt, 0.0);
                long tfVal = tf.getOrDefault(qt, 0L);
                double numerator = tfVal * (k1 + 1);
                double denominator = tfVal + k1 * (1 - b + b * (docLen / avgdl));
                score += idfVal * numerator / denominator;
            }
            results.add(new ScoredDocument(doc, score));
        }
        results.sort(Comparator.comparingDouble((ScoredDocument s) -> s.score).reversed());
        return results.subList(0, Math.min(topK, results.size()));
    }

    private List<ScoredDocument> rrfFusion(List<List<ScoredDocument>> rankLists, int k) {
        Map<String, Double> fusedScores = new LinkedHashMap<>();
        Map<String, ScoredDocument> docMap = new HashMap<>();

        for (List<ScoredDocument> rankList : rankLists) {
            for (int i = 0; i < rankList.size(); i++) {
                ScoredDocument sd = rankList.get(i);
                String id = sd.doc.id;
                fusedScores.merge(id, 1.0 / (k + i + 1), Double::sum);
                docMap.putIfAbsent(id, sd);
            }
        }

        List<ScoredDocument> fused = new ArrayList<>();
        for (Map.Entry<String, Double> entry : fusedScores.entrySet()) {
            ScoredDocument sd = docMap.get(entry.getKey());
            fused.add(new ScoredDocument(sd.doc, entry.getValue()));
        }
        fused.sort(Comparator.comparingDouble((ScoredDocument s) -> s.score).reversed());
        return fused;
    }

    private List<ScoredDocument> rerankWithCohere(List<ScoredDocument> documents, String query, String apiKey, int topK) {
        try {
            List<String> docTexts = documents.stream().map(d -> d.doc.content).toList();

            WebClient client = WebClient.builder()
                    .baseUrl("https://api.cohere.com/v1")
                    .defaultHeader("Authorization", "Bearer " + apiKey)
                    .defaultHeader("Content-Type", "application/json")
                    .build();

            Map<String, Object> body = Map.of("model", "rerank-multilingual-v3.0",
                    "query", query, "documents", docTexts, "top_n", topK);

            Map response = client.post().uri("/rerank").bodyValue(body)
                    .retrieve().bodyToMono(Map.class).block();

            if (response != null && response.containsKey("results")) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                List<ScoredDocument> reranked = new ArrayList<>();
                for (Map<String, Object> r : results) {
                    int index = ((Number) r.get("index")).intValue();
                    double relevanceScore = ((Number) r.get("relevance_score")).doubleValue();
                    if (index < documents.size()) {
                        reranked.add(new ScoredDocument(documents.get(index).doc, relevanceScore));
                    }
                }
                return reranked;
            }
        } catch (Exception e) {
            log.warn("Cohere rerank failed, using RRF results: {}", e.getMessage());
        }
        return documents.subList(0, Math.min(topK, documents.size()));
    }

    private void rebuildBm25Index(String collection) {
        List<RagDocument> docs = documentStore.getOrDefault(collection, List.of());
        if (docs.isEmpty()) return;

        Map<String, Double> idf = new HashMap<>();
        long totalDocs = docs.size();
        Map<String, Long> docFreq = new HashMap<>();

        for (RagDocument doc : docs) {
            Map<String, Boolean> seen = new HashMap<>();
            for (String term : tokenize(doc.content)) {
                if (seen.putIfAbsent(term, true) == null) {
                    docFreq.merge(term, 1L, Long::sum);
                }
            }
        }

        for (Map.Entry<String, Long> entry : docFreq.entrySet()) {
            idf.put(entry.getKey(), Math.log((totalDocs - entry.getValue() + 0.5) / (entry.getValue() + 0.5) + 1));
        }
        bm25Idf.put(collection, idf);
        log.info("BM25 index rebuilt | collection={} | terms={}", collection, idf.size());
    }

    private String[] tokenize(String text) {
        return text.toLowerCase().replaceAll("[^a-z0-9\\u4e00-\\u9fa5]", " ").split("\\s+");
    }

    private static class RagDocument {
        final String id;
        final String content;
        final Map<String, Object> metadata;

        RagDocument(String id, String content, Map<String, Object> metadata) {
            this.id = id;
            this.content = content;
            this.metadata = metadata;
        }
    }

    private static class ScoredDocument {
        final RagDocument doc;
        final double score;

        ScoredDocument(RagDocument doc, double score) {
            this.doc = doc;
            this.score = score;
        }
    }

    private String jsonResponseSync(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException e) {
            return "{\"status\":\"error\",\"message\":\"Serialization error\"}";
        }
    }
}