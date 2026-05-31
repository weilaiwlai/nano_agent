package com.nanoagent.service.graph;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ConversationHistoryViewer {

    private static final Logger log = LoggerFactory.getLogger(ConversationHistoryViewer.class);

    private final DataSource dataSource;
    private final String tableName;
    private final ObjectMapper objectMapper;

    public ConversationHistoryViewer(DataSource dataSource, String tableName) {
        this.dataSource = dataSource;
        this.tableName = tableName != null && !tableName.isBlank() ? tableName : "graph_checkpoints";
        this.objectMapper = new ObjectMapper();
    }

    public List<String> listThreadIds(int limit) {
        List<String> threadIds = new ArrayList<>();
        try (Connection conn = dataSource.getConnection()) {
            String sql = "SELECT thread_id FROM " + tableName + " ORDER BY updated_at DESC LIMIT ?";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setInt(1, limit);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        threadIds.add(rs.getString("thread_id"));
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to list thread IDs: {}", e.getMessage());
        }
        return threadIds;
    }

    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> getConversationHistory(String threadId) {
        List<Map<String, Object>> history = new ArrayList<>();
        try (Connection conn = dataSource.getConnection()) {
            String sql = "SELECT state_json, created_at, updated_at FROM " + tableName + " WHERE thread_id = ?";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (rs.next()) {
                        String stateJson = rs.getString("state_json");
                        Timestamp createdAt = rs.getTimestamp("created_at");
                        Timestamp updatedAt = rs.getTimestamp("updated_at");

                        Map<String, Object> state = objectMapper.readValue(stateJson,
                                new TypeReference<Map<String, Object>>() {});

                        if (state.containsKey("messages")) {
                            List<Map<String, Object>> messages = (List<Map<String, Object>>) state.get("messages");
                            if (messages != null) {
                                for (Map<String, Object> msg : messages) {
                                    Map<String, Object> item = new LinkedHashMap<>();
                                    item.put("type", msg.getOrDefault("type", "unknown"));
                                    item.put("content", msg.getOrDefault("content", ""));
                                    item.put("name", msg.getOrDefault("name", ""));
                                    item.put("created_at", createdAt != null ? createdAt.toString() : "");
                                    item.put("updated_at", updatedAt != null ? updatedAt.toString() : "");
                                    history.add(item);
                                }
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to get conversation history | threadId={} | error={}", threadId, e.getMessage());
        }
        return history;
    }
}