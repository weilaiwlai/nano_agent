package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.model.LlmProviderItem;
import com.nanoagent.service.model.LlmProviderListResponse;
import com.nanoagent.service.model.LlmSessionCreateRequest;
import com.nanoagent.service.model.LlmSessionCreateResponse;
import com.nanoagent.service.model.LlmSessionDeleteResponse;
import com.nanoagent.service.model.LlmSessionValidateRequest;
import com.nanoagent.service.model.LlmSessionValidateResponse;
import com.nanoagent.service.session.LlmSessionStore;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/session")
public class LlmSessionController {

    private static final Logger log = LoggerFactory.getLogger(LlmSessionController.class);

    private final LlmSessionStore sessionStore;
    private final AuthService authService;
    private final NanoAgentProperties properties;

    public LlmSessionController(LlmSessionStore sessionStore, AuthService authService, NanoAgentProperties properties) {
        this.sessionStore = sessionStore;
        this.authService = authService;
        this.properties = properties;
    }

    @GetMapping("/llm/providers")
    public LlmProviderListResponse listProviders() {
        List<LlmProviderItem> items = List.of(
                LlmProviderItem.builder()
                        .provider("qwen").label("Tongyi Qwen")
                        .requiresBaseUrl(false)
                        .defaultBaseUrl("https://dashscope.aliyuncs.com/compatible-mode/v1")
                        .defaultEmbeddingModel("")
                        .build(),
                LlmProviderItem.builder()
                        .provider("openai").label("OpenAI")
                        .requiresBaseUrl(false)
                        .defaultBaseUrl("https://api.openai.com/v1")
                        .defaultEmbeddingModel("")
                        .build(),
                LlmProviderItem.builder()
                        .provider("deepseek").label("DeepSeek")
                        .requiresBaseUrl(false)
                        .defaultBaseUrl("https://api.deepseek.com/v1")
                        .defaultEmbeddingModel("")
                        .build(),
                LlmProviderItem.builder()
                        .provider("groq").label("Groq")
                        .requiresBaseUrl(false)
                        .defaultBaseUrl("https://api.groq.com/openai/v1")
                        .defaultEmbeddingModel("")
                        .build(),
                LlmProviderItem.builder()
                        .provider("other").label("Other")
                        .requiresBaseUrl(true)
                        .defaultBaseUrl(null)
                        .defaultEmbeddingModel("")
                        .build()
        );
        return LlmProviderListResponse.builder().items(items).build();
    }

    @PostMapping("/llm/validate")
    public LlmSessionValidateResponse validateSession(
            @Valid @RequestBody LlmSessionValidateRequest request) {
        Map<String, String> llmProfile = buildLlmProfile(request);
        return LlmSessionValidateResponse.builder()
                .provider(llmProfile.getOrDefault("provider", "other"))
                .model(llmProfile.get("model"))
                .baseUrl(llmProfile.get("base_url"))
                .embeddingModel(llmProfile.get("embedding_model"))
                .chatOk(true)
                .embeddingOk(true)
                .errors(List.of())
                .build();
    }

    @PostMapping("/llm")
    public LlmSessionCreateResponse createSession(
            @Valid @RequestBody LlmSessionCreateRequest request,
            HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String ownerId = authContext.requireSubject();
        Map<String, String> llmProfile = buildLlmProfile(request);

        String sessionId = sessionStore.createSession(llmProfile, ownerId);

        log.info("LLM session created | owner_id={} | session_id={} | provider={} | model={}",
                ownerId, sessionId,
                llmProfile.getOrDefault("provider", "other"),
                llmProfile.get("model"));

        return LlmSessionCreateResponse.builder()
                .sessionId(sessionId)
                .provider(llmProfile.getOrDefault("provider", "other"))
                .expiresIn(properties.getSession().getTtlSeconds())
                .model(llmProfile.get("model"))
                .baseUrl(llmProfile.get("base_url"))
                .embeddingModel(llmProfile.get("embedding_model"))
                .build();
    }

    @DeleteMapping("/llm/{sessionId}")
    public LlmSessionDeleteResponse deleteSession(
            @PathVariable String sessionId,
            HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String ownerId = authContext.requireSubject();

        boolean deleted = sessionStore.deleteSession(sessionId.trim(), ownerId);
        if (!deleted) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "会话不存在、已过期或不属于当前用户");
        }

        log.info("LLM session deleted | owner_id={} | session_id={}", ownerId, sessionId);
        return LlmSessionDeleteResponse.builder()
                .sessionId(sessionId.trim())
                .status("success")
                .message("LLM 会话已删除。")
                .build();
    }

    private Map<String, String> buildLlmProfile(LlmSessionCreateRequest request) {
        Map<String, String> profile = new LinkedHashMap<>();
        profile.put("api_key", request.getApiKey().trim());
        profile.put("model", request.getModel().trim());

        String provider = request.getProvider() != null ? request.getProvider().trim().toLowerCase() : "other";
        profile.put("provider", provider);

        String baseUrl = request.getBaseUrl();
        if (baseUrl != null && !baseUrl.isBlank()) {
            profile.put("base_url", baseUrl.trim());
        } else {
            profile.put("base_url", resolveDefaultBaseUrl(provider));
        }

        String embeddingModel = request.getEmbeddingModel();
        profile.put("embedding_model", embeddingModel != null ? embeddingModel.trim() : "");

        return profile;
    }

    private Map<String, String> buildLlmProfile(LlmSessionValidateRequest request) {
        Map<String, String> profile = new LinkedHashMap<>();
        profile.put("api_key", request.getApiKey().trim());
        profile.put("model", request.getModel().trim());

        String provider = request.getProvider() != null ? request.getProvider().trim().toLowerCase() : "other";
        profile.put("provider", provider);

        String baseUrl = request.getBaseUrl();
        if (baseUrl != null && !baseUrl.isBlank()) {
            profile.put("base_url", baseUrl.trim());
        } else {
            profile.put("base_url", resolveDefaultBaseUrl(provider));
        }

        String embeddingModel = request.getEmbeddingModel();
        profile.put("embedding_model", embeddingModel != null ? embeddingModel.trim() : "");

        return profile;
    }

    private String resolveDefaultBaseUrl(String provider) {
        return switch (provider) {
            case "qwen" -> "https://dashscope.aliyuncs.com/compatible-mode/v1";
            case "openai" -> "https://api.openai.com/v1";
            case "deepseek" -> "https://api.deepseek.com/v1";
            case "groq" -> "https://api.groq.com/openai/v1";
            default -> "";
        };
    }
}