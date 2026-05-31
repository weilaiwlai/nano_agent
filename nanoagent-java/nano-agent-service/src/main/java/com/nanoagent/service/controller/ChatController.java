package com.nanoagent.service.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.graph.AgentState;
import com.nanoagent.service.graph.AgentWorkflowEngine;
import com.nanoagent.service.graph.GraphConfig;
import com.nanoagent.service.model.ChatRequest;
import com.nanoagent.service.model.ChatResumeRequest;
import com.nanoagent.service.session.LlmSessionStore;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/v1")
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final AgentWorkflowEngine workflowEngine;
    private final LlmSessionStore sessionStore;
    private final AuthService authService;
    private final NanoAgentProperties properties;
    private final ObjectMapper objectMapper;

    public ChatController(AgentWorkflowEngine workflowEngine, LlmSessionStore sessionStore,
                           AuthService authService, NanoAgentProperties properties) {
        this.workflowEngine = workflowEngine;
        this.sessionStore = sessionStore;
        this.authService = authService;
        this.properties = properties;
        this.objectMapper = new ObjectMapper();
    }

    @PostMapping(path = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter chat(@Valid @RequestBody ChatRequest request, HttpServletRequest httpRequest) {
        SseEmitter emitter = new SseEmitter(600_000L);

        CompletableFuture.runAsync(() -> {
            try {
                AuthContext authContext = authService.authenticate(httpRequest);
                String tokenSubject = authContext.requireSubject();
                String userId = authService.resolveEffectiveUserId(tokenSubject, request.getUserId());

                String sessionId = request.getSessionId();
                if (properties.getSession().isRequireLlmSession() && (sessionId == null || sessionId.isBlank())) {
                    sendEvent(emitter, "error", Map.of("message", "缺少 llm_session_id"));
                    emitter.complete();
                    return;
                }

                Map<String, String> llmProfile;
                if (sessionId != null && !sessionId.isBlank()) {
                    llmProfile = sessionStore.getProfile(sessionId.trim());
                    if (llmProfile == null || llmProfile.isEmpty()) {
                        sendEvent(emitter, "error", Map.of("message", "LLM 会话无效或已过期"));
                        emitter.complete();
                        return;
                    }
                    String ownerId = sessionStore.getOwnerId(sessionId.trim());
                    if (!tokenSubject.equals(ownerId)) {
                        sendEvent(emitter, "error", Map.of("message", "LLM 会话不属于当前用户"));
                        emitter.complete();
                        return;
                    }
                } else {
                    llmProfile = Map.of();
                }

                String threadId = request.getThreadId();
                if (threadId == null || threadId.isBlank()) {
                    threadId = UUID.randomUUID().toString();
                }

                String query = request.getQuery();
                if (query == null || query.isBlank()) {
                    sendEvent(emitter, "error", Map.of("message", "query 不能为空"));
                    emitter.complete();
                    return;
                }

                GraphConfig config = GraphConfig.builder()
                        .userId(userId)
                        .threadId(threadId)
                        .llmProfile(llmProfile)
                        .query(query)
                        .metadata(Map.of())
                        .build();

                var finalState = workflowEngine.runWorkflow(userId, threadId, llmProfile, query);

                sendResult(emitter, finalState, threadId);

            } catch (Exception e) {
                log.error("Chat error", e);
                try {
                    sendEvent(emitter, "error", Map.of("message", e.getMessage()));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            } finally {
                try {
                    emitter.complete();
                } catch (Exception ignored) {}
            }
        });

        emitter.onTimeout(() -> {
            log.warn("SSE emitter timed out");
            try { emitter.complete(); } catch (Exception ignored) {}
        });

        emitter.onError(ex -> {
            log.error("SSE emitter error: {}", ex.getMessage());
        });

        return emitter;
    }

    @PostMapping(path = "/chat/resume", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter resumeChat(@Valid @RequestBody ChatResumeRequest request, HttpServletRequest httpRequest) {
        SseEmitter emitter = new SseEmitter(600_000L);

        CompletableFuture.runAsync(() -> {
            try {
                AuthContext authContext = authService.authenticate(httpRequest);
                String tokenSubject = authContext.requireSubject();
                String userId = authService.resolveEffectiveUserId(tokenSubject, request.getUserId());

                String sessionId = request.getSessionId();
                Map<String, String> llmProfile;
                if (sessionId != null && !sessionId.isBlank()) {
                    llmProfile = sessionStore.getProfile(sessionId.trim());
                    if (llmProfile == null || llmProfile.isEmpty()) {
                        sendEvent(emitter, "error", Map.of("message", "LLM 会话无效或已过期"));
                        emitter.complete();
                        return;
                    }
                } else {
                    llmProfile = Map.of();
                }

                String threadId = request.getThreadId();
                if (threadId == null || threadId.isBlank()) {
                    sendEvent(emitter, "error", Map.of("message", "thread_id 不能为空"));
                    emitter.complete();
                    return;
                }

                String action = request.getAction();
                if (action == null || action.isBlank()) {
                    sendEvent(emitter, "error", Map.of("message", "action 不能为空（approve 或 reject）"));
                    emitter.complete();
                    return;
                }

                GraphConfig config = GraphConfig.builder()
                        .userId(userId)
                        .threadId(threadId)
                        .llmProfile(llmProfile)
                        .metadata(Map.of())
                        .build();

                var finalState = workflowEngine.resumeWorkflow(userId, threadId, llmProfile, action);

                sendResult(emitter, finalState, threadId);

            } catch (Exception e) {
                log.error("Resume chat error", e);
                try {
                    sendEvent(emitter, "error", Map.of("message", e.getMessage()));
                } catch (Exception ignored) {}
                emitter.completeWithError(e);
            } finally {
                try {
                    emitter.complete();
                } catch (Exception ignored) {}
            }
        });

        emitter.onTimeout(() -> {
            try { emitter.complete(); } catch (Exception ignored) {}
        });

        emitter.onError(ex -> {
            log.error("SSE resume emitter error: {}", ex.getMessage());
        });

        return emitter;
    }

    @SuppressWarnings("unchecked")
    private void sendResult(SseEmitter emitter, Object finalState, String threadId) {
        try {
            List<AgentState.Message> messages;
            if (finalState instanceof Map<?, ?> stateMap) {
                Object msgsObj = stateMap.get("messages");
                if (msgsObj instanceof List<?> list) {
                    messages = (List<AgentState.Message>) list;
                } else {
                    messages = Collections.emptyList();
                }
            } else {
                return;
            }

            if (messages == null || messages.isEmpty()) {
                sendEvent(emitter, "done", Map.of("thread_id", threadId));
                return;
            }

            String currentNode = "";
            for (int i = 0; i < messages.size(); i++) {
                AgentState.Message msg = messages.get(i);

                String sender = inferSender(msg);
                if (!sender.isBlank() && !sender.equals(currentNode)) {
                    currentNode = sender;
                    sendEvent(emitter, "agent_switch", Map.of("agent", sender));
                }

                if (msg.getToolCalls() != null && !msg.getToolCalls().isEmpty()) {
                    for (AgentState.ToolCall tc : msg.getToolCalls()) {
                        Map<String, Object> toolStart = new LinkedHashMap<>();
                        toolStart.put("type", "tool_start");
                        toolStart.put("tool", tc.getName());
                        sendEvent(emitter, "tool_start", toolStart);

                        if ("tool_send_report".equals(tc.getName())) {
                            Map<String, Object> approval = new LinkedHashMap<>();
                            approval.put("type", "approval");
                            approval.put("thread_id", threadId);
                            approval.put("tool", "send_report");
                            approval.put("args", tc.getArgs());
                            approval.put("content", msg.getContent());
                            approval.put("message", "请确认是否发送邮件？回复 approve 或 reject");
                            sendEvent(emitter, "approval_required", approval);
                            sendEvent(emitter, "done", Map.of("thread_id", threadId));
                            return;
                        } else {
                            Map<String, Object> toolEnd = new LinkedHashMap<>();
                            toolEnd.put("type", "tool_end");
                            toolEnd.put("tool", tc.getName());
                            sendEvent(emitter, "tool_end", toolEnd);
                        }
                    }
                }

                if (msg.getType() == AgentState.Message.MessageType.TOOL) {
                    Map<String, Object> payload = new LinkedHashMap<>();
                    payload.put("type", "tool_end");
                    payload.put("tool", msg.getName());
                    sendEvent(emitter, "tool_end", payload);
                }

                if (msg.getType() == AgentState.Message.MessageType.AI && (msg.getToolCalls() == null || msg.getToolCalls().isEmpty())) {
                    String content = msg.getContent() != null ? msg.getContent() : "";
                    if (!content.isBlank()) {
                        sendEvent(emitter, "token", Map.of("content", content));
                    }
                }
            }

            sendEvent(emitter, "done", Map.of("thread_id", threadId));

        } catch (Exception e) {
            log.error("Error sending result", e);
            try {
                sendEvent(emitter, "done", Map.of("thread_id", threadId));
            } catch (Exception ignored) {}
        }
    }

    private String inferSender(AgentState.Message msg) {
        if (msg.getType() == AgentState.Message.MessageType.SYSTEM) return "";
        if (msg.getType() == AgentState.Message.MessageType.TOOL) return toolsNodeName(msg);

        String content = msg.getContent() != null ? msg.getContent().toUpperCase().trim() : "";
        if (content.contains("KNOWLEDGE_WORKER")) return "knowledge_worker_node";
        if (content.contains("REPORTER")) return "reporter_node";
        if (content.contains("ASSISTANT")) return "assistant_node";
        if (content.contains("FINISH")) return "";
        if (msg.getType() == AgentState.Message.MessageType.AI) return "assistant_node";
        return "";
    }

    private String toolsNodeName(AgentState.Message msg) {
        String name = msg.getName() != null ? msg.getName() : "";
        if (name.contains("send_report")) return "permission_tools_node";
        return "tools_node";
    }

    private void sendEvent(SseEmitter emitter, String eventName, Object data) {
        try {
            emitter.send(SseEmitter.event()
                    .name(eventName)
                    .data(objectMapper.writeValueAsString(data)));
        } catch (Exception e) {
            log.error("Failed to send SSE event: {}", e.getMessage());
        }
    }
}