package com.nanoagent.service.graph.checkpoint;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Map;

public class PostgresCheckpointer implements GraphCheckpointer {

    private static final Logger log = LoggerFactory.getLogger(PostgresCheckpointer.class);

    private final DataSource dataSource;
    private final ObjectMapper objectMapper;
    private final String tableName;

    public PostgresCheckpointer(DataSource dataSource, String tableName) {
        this.dataSource = dataSource;
        this.objectMapper = new ObjectMapper();
        this.tableName = tableName != null && !tableName.isBlank() ? tableName : "graph_checkpoints";
        initTable();
    }

    private void initTable() {
        try (Connection conn = dataSource.getConnection();
             Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS " + tableName + " (" +
                    "thread_id VARCHAR(255) PRIMARY KEY," +
                    "state_json TEXT NOT NULL," +
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP," +
                    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" +
                    ")");
            log.info("Postgres checkpointer table initialized: {}", tableName);
        } catch (Exception e) {
            log.error("Failed to initialize checkpointer table: {}", e.getMessage());
        }
    }

    @Override
    public void put(String threadId, Map<String, Object> state) {
        try (Connection conn = dataSource.getConnection()) {
            String json = objectMapper.writeValueAsString(state);
            String sql = "INSERT INTO " + tableName + " (thread_id, state_json, updated_at) " +
                    "VALUES (?, ?, CURRENT_TIMESTAMP) " +
                    "ON CONFLICT (thread_id) DO UPDATE SET state_json = ?, updated_at = CURRENT_TIMESTAMP";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, threadId);
                ps.setString(2, json);
                ps.setString(3, json);
                ps.executeUpdate();
            }
        } catch (Exception e) {
            log.error("Postgres checkpointer put error | threadId={} | error={}", threadId, e.getMessage());
        }
    }

    @Override
    public Map<String, Object> get(String threadId) {
        try (Connection conn = dataSource.getConnection()) {
            String sql = "SELECT state_json FROM " + tableName + " WHERE thread_id = ?";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    if (rs.next()) {
                        String json = rs.getString("state_json");
                        return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
                    }
                }
            }
        } catch (Exception e) {
            log.error("Postgres checkpointer get error | threadId={} | error={}", threadId, e.getMessage());
        }
        return null;
    }

    @Override
    public void delete(String threadId) {
        try (Connection conn = dataSource.getConnection()) {
            String sql = "DELETE FROM " + tableName + " WHERE thread_id = ?";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, threadId);
                ps.executeUpdate();
            }
        } catch (Exception e) {
            log.error("Postgres checkpointer delete error | threadId={} | error={}", threadId, e.getMessage());
        }
    }

    @Override
    public boolean exists(String threadId) {
        try (Connection conn = dataSource.getConnection()) {
            String sql = "SELECT 1 FROM " + tableName + " WHERE thread_id = ?";
            try (PreparedStatement ps = conn.prepareStatement(sql)) {
                ps.setString(1, threadId);
                try (ResultSet rs = ps.executeQuery()) {
                    return rs.next();
                }
            }
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public String getBackendName() {
        return "postgres";
    }
}