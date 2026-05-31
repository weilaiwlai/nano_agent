package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.memory.UserMemoryManager;
import com.nanoagent.service.model.*;
import com.nanoagent.service.session.LlmSessionStore;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/memory")
public class MemoryController {

    private static final Logger log = LoggerFactory.getLogger(MemoryController.class);

    private final UserMemoryManager memoryManager;
    private final AuthService authService;
    private final LlmSessionStore sessionStore;
    private final NanoAgentProperties properties;

    public MemoryController(UserMemoryManager memoryManager, AuthService authService,
                             LlmSessionStore sessionStore, NanoAgentProperties properties) {
        this.memoryManager = memoryManager;
        this.authService = authService;
        this.sessionStore = sessionStore;
        this.properties = properties;
    }

    @PostMapping(consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public MemoryResponse saveMemory(@Valid @RequestBody MemoryRequest request,
                                      HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String tokenSubject = authContext.requireSubject();
        String userId = authService.resolveEffectiveUserId(tokenSubject, request.getUserId());

        String sessionId = request.getSessionId();
        Map<String, String> llmProfile;
        if (sessionId != null && !sessionId.isBlank()) {
            llmProfile = sessionStore.getProfile(sessionId.trim());
            if (llmProfile == null || llmProfile.isEmpty()) {
                throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "LLM 会话不存在或已过期");
            }
        } else {
            llmProfile = Map.of();
        }

        String memoryId = memoryManager.savePreference(userId, request.getPreferenceText(), llmProfile);
        if (memoryId == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "长期记忆写入失败，请检查输入内容后重试");
        }

        log.info("记忆写入成功 | user_id={} | memory_id={}", userId, memoryId);

        return MemoryResponse.builder()
                .userId(userId)
                .status("success")
                .message("偏好已成功写入长期记忆。")
                .memoryId(memoryId)
                .build();
    }

    @GetMapping("/{userId}")
    public MemoryListResponse listMemories(@PathVariable String userId,
                                            @RequestParam(defaultValue = "50") int limit,
                                            HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String tokenSubject = authContext.requireSubject();
        String resolvedUserId = authService.resolveEffectiveUserId(tokenSubject, userId);

        List<Map<String, Object>> items = memoryManager.listMemories(resolvedUserId, Math.min(limit, 200));

        List<MemoryItem> memoryItems = items.stream()
                .map(item -> MemoryItem.builder()
                        .memoryId(String.valueOf(item.getOrDefault("memory_id", "")))
                        .preferenceText(String.valueOf(item.getOrDefault("preference_text", "")))
                        .timestamp(String.valueOf(item.getOrDefault("timestamp", "")))
                        .build())
                .collect(Collectors.toList());

        return MemoryListResponse.builder()
                .userId(resolvedUserId)
                .items(memoryItems)
                .build();
    }

    @DeleteMapping("/{userId}/{memoryId}")
    public MemoryDeleteResponse deleteMemory(@PathVariable String userId,
                                              @PathVariable String memoryId,
                                              HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String tokenSubject = authContext.requireSubject();
        String resolvedUserId = authService.resolveEffectiveUserId(tokenSubject, userId);

        boolean deleted = memoryManager.deleteMemory(resolvedUserId, memoryId.trim());
        if (!deleted) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "未找到可删除的记忆记录");
        }

        log.info("记忆删除成功 | user_id={} | memory_id={}", resolvedUserId, memoryId);

        return MemoryDeleteResponse.builder()
                .userId(resolvedUserId)
                .memoryId(memoryId)
                .status("success")
                .message("记忆已删除。")
                .build();
    }
}