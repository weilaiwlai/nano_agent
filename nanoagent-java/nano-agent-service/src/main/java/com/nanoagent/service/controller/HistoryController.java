package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.graph.ConversationHistoryViewer;
import com.nanoagent.service.model.ChatHistoryItem;
import com.nanoagent.service.model.ChatHistoryResponse;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import javax.sql.DataSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/history")
public class HistoryController {

    private static final Logger log = LoggerFactory.getLogger(HistoryController.class);

    private final ConversationHistoryViewer historyViewer;
    private final AuthService authService;

    public HistoryController(DataSource dataSource, AuthService authService) {
        this.historyViewer = new ConversationHistoryViewer(dataSource, "graph_checkpoints");
        this.authService = authService;
    }

    @GetMapping("/threads")
    public List<String> listThreadIds(@RequestParam(defaultValue = "50") int limit,
                                       HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        authContext.requireSubject();
        return historyViewer.listThreadIds(limit);
    }

    @GetMapping("/threads/{threadId}")
    public ChatHistoryResponse getThreadHistory(@PathVariable String threadId,
                                                  HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String subject = authContext.requireSubject();

        List<Map<String, Object>> rawHistory = historyViewer.getConversationHistory(threadId);
        List<ChatHistoryItem> items = new ArrayList<>();
        for (Map<String, Object> item : rawHistory) {
            items.add(ChatHistoryItem.builder()
                    .role(String.valueOf(item.getOrDefault("type", "unknown")))
                    .content(String.valueOf(item.getOrDefault("content", "")))
                    .timestamp(String.valueOf(item.getOrDefault("updated_at", "")))
                    .build());
        }

        return ChatHistoryResponse.builder()
                .threadId(threadId)
                .userId(subject)
                .messages(items)
                .build();
    }
}