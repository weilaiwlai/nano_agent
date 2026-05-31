package com.nanoagent.mcp.controller;

import com.nanoagent.mcp.service.RagService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/rag")
public class RagController {

    private static final Logger log = LoggerFactory.getLogger(RagController.class);

    private final RagService ragService;

    public RagController(RagService ragService) {
        this.ragService = ragService;
    }

    @PostMapping("/ingest")
    public Mono<String> ingest(@RequestBody Map<String, Object> request) {
        String collection = String.valueOf(request.getOrDefault("collection", "default"));
        @SuppressWarnings("unchecked")
        List<String> documents = (List<String>) request.getOrDefault("documents", List.of());
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> metadatas = (List<Map<String, Object>>) request.getOrDefault("metadatas", List.of());

        log.info("RAG ingest | collection={} | docs={}", collection, documents.size());
        return ragService.ingest(collection, documents, metadatas);
    }

    @PostMapping("/search")
    public Mono<String> search(@RequestBody Map<String, Object> request) {
        String collection = String.valueOf(request.getOrDefault("collection", "default"));
        String query = String.valueOf(request.getOrDefault("query", ""));
        int topK = request.containsKey("top_k") ? ((Number) request.get("top_k")).intValue() : 5;

        log.info("RAG search | collection={} | query_len={} | top_k={}", collection, query.length(), topK);
        return ragService.hybridSearch(collection, query, topK);
    }
}