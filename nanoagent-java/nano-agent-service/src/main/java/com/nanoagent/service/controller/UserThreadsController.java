package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import com.nanoagent.service.graph.ConversationHistoryViewer;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.*;

import javax.sql.DataSource;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1")
public class UserThreadsController {

    private final ConversationHistoryViewer historyViewer;
    private final AuthService authService;

    public UserThreadsController(DataSource dataSource, AuthService authService) {
        this.historyViewer = new ConversationHistoryViewer(dataSource, "graph_checkpoints");
        this.authService = authService;
    }

    @GetMapping("/user_threads/{userId}")
    public Map<String, Object> getUserThreads(@PathVariable String userId,
                                               @RequestParam(defaultValue = "100") int limit,
                                               HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String tokenSubject = authContext.requireSubject();
        String resolvedUserId = authService.resolveEffectiveUserId(tokenSubject, userId);

        List<String> allThreadIds = historyViewer.listThreadIds(limit);
        List<String> userThreadIds = allThreadIds.stream()
                .filter(tid -> tid.equals(resolvedUserId) || tid.startsWith(resolvedUserId + ":"))
                .collect(Collectors.toList());

        return Map.of("user_id", resolvedUserId, "thread_ids", userThreadIds);
    }
}