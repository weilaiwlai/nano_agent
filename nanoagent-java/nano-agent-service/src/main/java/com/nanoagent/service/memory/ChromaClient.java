package com.nanoagent.service.memory;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ChromaClient {

    private static final Logger log = LoggerFactory.getLogger(ChromaClient.class);

    private final String baseUrl;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public ChromaClient(String baseUrl) {
        this.baseUrl = baseUrl != null && !baseUrl.isBlank() ? baseUrl : "http://localhost:8001";
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    public boolean createCollection(String name, Map<String, Object> metadata) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("name", name);
            if (metadata != null) {
                body.put("metadata", metadata);
            }
            restTemplate.postForObject(baseUrl + "/api/v1/collections", body, Map.class);
            log.info("Chroma collection created: {}", name);
            return true;
        } catch (Exception e) {
            log.warn("Chroma create collection error: {}", e.getMessage());
            return false;
        }
    }

    public boolean collectionExists(String name) {
        try {
            Map result = restTemplate.getForObject(baseUrl + "/api/v1/collections/" + name, Map.class);
            return result != null;
        } catch (Exception e) {
            return false;
        }
    }

    public boolean addDocuments(String collectionName, List<String> ids, List<String> documents,
                                 List<List<Double>> embeddings, Map<String, Object> metadatas) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("ids", ids);
            body.put("documents", documents);
            if (embeddings != null && !embeddings.isEmpty()) {
                body.put("embeddings", embeddings);
            }
            if (metadatas != null) {
                body.put("metadatas", metadatas);
            }

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            restTemplate.postForObject(
                    baseUrl + "/api/v1/collections/" + collectionName + "/add", request, Map.class);
            return true;
        } catch (Exception e) {
            log.error("Chroma add documents error: {}", e.getMessage());
            return false;
        }
    }

    public List<Map<String, Object>> query(String collectionName, List<Double> queryEmbedding,
                                            int nResults, Map<String, Object> where) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("query_embeddings", List.of(queryEmbedding));
            body.put("n_results", nResults);
            if (where != null) {
                body.put("where", where);
            }

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            Map result = restTemplate.postForObject(
                    baseUrl + "/api/v1/collections/" + collectionName + "/query", request, Map.class);

            List<Map<String, Object>> items = new ArrayList<>();
            if (result != null && result.containsKey("ids")) {
                List<List<String>> idsList = (List<List<String>>) result.get("ids");
                List<List<String>> docsList = (List<List<String>>) result.get("documents");
                List<List<Double>> distancesList = (List<List<Double>>) result.get("distances");

                if (idsList != null && !idsList.isEmpty()) {
                    List<String> ids = idsList.get(0);
                    List<String> docs = docsList != null && !docsList.isEmpty() ? docsList.get(0) : null;
                    List<Double> distances = distancesList != null && !distancesList.isEmpty() ? distancesList.get(0) : null;

                    for (int i = 0; i < ids.size(); i++) {
                        Map<String, Object> item = new LinkedHashMap<>();
                        item.put("id", ids.get(i));
                        item.put("document", docs != null && i < docs.size() ? docs.get(i) : "");
                        item.put("distance", distances != null && i < distances.size() ? distances.get(i) : 0.0);
                        items.add(item);
                    }
                }
            }
            return items;
        } catch (Exception e) {
            log.error("Chroma query error: {}", e.getMessage());
            return List.of();
        }
    }

    public boolean deleteDocuments(String collectionName, List<String> ids) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("ids", ids);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            restTemplate.postForObject(
                    baseUrl + "/api/v1/collections/" + collectionName + "/delete", request, Map.class);
            return true;
        } catch (Exception e) {
            log.error("Chroma delete error: {}", e.getMessage());
            return false;
        }
    }

    public List<Map<String, Object>> getDocuments(String collectionName, int limit, int offset) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("limit", limit);
            body.put("offset", offset);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            Map result = restTemplate.postForObject(
                    baseUrl + "/api/v1/collections/" + collectionName + "/get", request, Map.class);

            List<Map<String, Object>> items = new ArrayList<>();
            if (result != null && result.containsKey("ids")) {
                List<String> ids = (List<String>) result.get("ids");
                List<String> docs = (List<String>) result.get("documents");
                List<Map<String, Object>> metadatas = (List<Map<String, Object>>) result.get("metadatas");

                for (int i = 0; i < ids.size(); i++) {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("id", ids.get(i));
                    item.put("document", docs != null && i < docs.size() ? docs.get(i) : "");
                    item.put("metadata", metadatas != null && i < metadatas.size() ? metadatas.get(i) : Map.of());
                    items.add(item);
                }
            }
            return items;
        } catch (Exception e) {
            log.error("Chroma get documents error: {}", e.getMessage());
            return List.of();
        }
    }
}