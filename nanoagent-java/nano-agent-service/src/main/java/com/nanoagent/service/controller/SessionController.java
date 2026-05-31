package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.model.LlmSessionCreateRequest;
import com.nanoagent.service.model.LlmSessionValidateRequest;
import com.nanoagent.service.session.LlmSessionStore;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/session/llm")
public class SessionController {

    private static final Logger log = LoggerFactory.getLogger(SessionController.class);
    private static final List<Map<String, Object>> PROVIDERS = buildProviders();

    private final LlmSessionStore sessionStore;
    private final AuthService authService;

    public SessionController(LlmSessionStore sessionStore, AuthService authService) {
        this.sessionStore = sessionStore;
        this.authService = authService;
    }

    @GetMapping("/providers")
    public Map<String, Object> listProviders(HttpServletRequest request) {
        authService.authenticate(request);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("items", PROVIDERS);
        return result;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> createSession(@Valid @RequestBody LlmSessionCreateRequest request,
                                              HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String ownerId = authContext.requireSubject();

        Map<String, String> profile = new LinkedHashMap<>();
        profile.put("api_key", request.getApiKey());
        profile.put("model", request.getModel());
        if (request.getBaseUrl() != null && !request.getBaseUrl().isBlank()) {
            profile.put("base_url", request.getBaseUrl());
        }
        if (request.getEmbeddingModel() != null && !request.getEmbeddingModel().isBlank()) {
            profile.put("embedding_model", request.getEmbeddingModel());
        }
        if (request.getProvider() != null && !request.getProvider().isBlank()) {
            profile.put("provider", request.getProvider());
        }

        String sessionId = sessionStore.createSession(profile, ownerId);

        log.info("创建 LLM 会话 | owner_id={} | session_id={} | model={}", ownerId, sessionId, request.getModel());

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("session_id", sessionId);
        result.put("provider", request.getProvider() != null ? request.getProvider() : "other");
        result.put("expires_in", 3600);
        result.put("model", request.getModel());
        result.put("base_url", request.getBaseUrl() != null ? request.getBaseUrl() : "");
        result.put("embedding_model", request.getEmbeddingModel() != null ? request.getEmbeddingModel() : "");
        return result;
    }

    @DeleteMapping("/{sessionId}")
    public Map<String, Object> deleteSession(@PathVariable String sessionId, HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String ownerId = authContext.requireSubject();

        boolean deleted = sessionStore.deleteSession(sessionId.trim(), ownerId);
        if (!deleted) {
            throw new org.springframework.web.server.ResponseStatusException(
                    org.springframework.http.HttpStatus.NOT_FOUND, "会话不存在、已过期或不属于当前用户");
        }

        log.info("删除 LLM 会话 | owner_id={} | session_id={}", ownerId, sessionId);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("session_id", sessionId);
        result.put("status", "success");
        result.put("message", "LLM 会话已删除。");
        return result;
    }

    @PostMapping("/validate")
    public Map<String, Object> validateSession(@Valid @RequestBody LlmSessionValidateRequest request,
                                                HttpServletRequest httpRequest) {
        authService.authenticate(httpRequest);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("provider", request.getProvider() != null ? request.getProvider() : "other");
        result.put("model", request.getModel());
        result.put("base_url", request.getBaseUrl() != null ? request.getBaseUrl() : "");
        result.put("embedding_model", request.getEmbeddingModel() != null ? request.getEmbeddingModel() : "");
        result.put("chat_ok", true);
        result.put("embedding_ok", true);
        result.put("errors", List.of());

        return result;
    }

    private static List<Map<String, Object>> buildProviders() {
        List<Map<String, Object>> providers = new ArrayList<>();

        Map<String, Object> qwen = new LinkedHashMap<>();
        qwen.put("provider", "qwen");
        qwen.put("label", "Tongyi Qwen");
        qwen.put("requires_base_url", false);
        qwen.put("default_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1");
        qwen.put("default_embedding_model", "");
        providers.add(qwen);

        Map<String, Object> openai = new LinkedHashMap<>();
        openai.put("provider", "openai");
        openai.put("label", "OpenAI");
        openai.put("requires_base_url", false);
        openai.put("default_base_url", "https://api.openai.com/v1");
        openai.put("default_embedding_model", "");
        providers.add(openai);

        Map<String, Object> deepseek = new LinkedHashMap<>();
        deepseek.put("provider", "deepseek");
        deepseek.put("label", "DeepSeek");
        deepseek.put("requires_base_url", false);
        deepseek.put("default_base_url", "https://api.deepseek.com/v1");
        deepseek.put("default_embedding_model", "");
        providers.add(deepseek);

        Map<String, Object> groq = new LinkedHashMap<>();
        groq.put("provider", "groq");
        groq.put("label", "Groq");
        groq.put("requires_base_url", false);
        groq.put("default_base_url", "https://api.groq.com/openai/v1");
        groq.put("default_embedding_model", "");
        providers.add(groq);

        Map<String, Object> other = new LinkedHashMap<>();
        other.put("provider", "other");
        other.put("label", "Other");
        other.put("requires_base_url", true);
        other.put("default_base_url", null);
        other.put("default_embedding_model", "");
        providers.add(other);

        return providers;
    }
}