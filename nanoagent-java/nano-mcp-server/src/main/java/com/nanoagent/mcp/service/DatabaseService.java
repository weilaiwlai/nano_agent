package com.nanoagent.mcp.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanoagent.mcp.config.McpServerProperties;
import com.nanoagent.mcp.security.SqlSecurityValidator;
import io.r2dbc.spi.ConnectionFactory;
import org.springframework.r2dbc.core.DatabaseClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class DatabaseService {

    private static final Logger log = LoggerFactory.getLogger(DatabaseService.class);

    private final ConnectionFactory connectionFactory;
    private final McpServerProperties properties;
    private final SqlSecurityValidator sqlValidator;
    private final ObjectMapper objectMapper;
    public DatabaseService(ConnectionFactory connectionFactory, McpServerProperties properties, SqlSecurityValidator sqlValidator, ObjectMapper objectMapper) {
        this.connectionFactory = connectionFactory;
        this.properties = properties;
        this.sqlValidator = sqlValidator;
        this.objectMapper = objectMapper;
    }

    public Mono<String> queryDatabase(String sql) {
        String safetyError = sqlValidator.validate(sql);
        if (safetyError != null) {
            return jsonResponse(Map.of("status", "error", "message", safetyError));
        }

        String limitedSql = sqlValidator.buildLimitedSelectSql(sql, properties.getQueryRowLimit());
        DatabaseClient client = DatabaseClient.create(connectionFactory);

        int rawTimeoutMs = properties.getQueryTimeoutMs();
        final int timeoutMs = rawTimeoutMs > 0 ? rawTimeoutMs : 3000;

        String timeoutSql = limitedSql;
        if (isPostgres()) {
            timeoutSql = "SET LOCAL statement_timeout = " + timeoutMs + "; " + limitedSql;
        }

        return client.sql(timeoutSql)
                .fetch()
                .all()
                .collectList()
                .timeout(java.time.Duration.ofMillis(timeoutMs + 1000))
                .map(rows -> {
                    boolean truncated = rows.size() >= properties.getQueryRowLimit();
                    Map<String, Object> result = new LinkedHashMap<>();
                    result.put("status", "success");
                    result.put("row_count", rows.size());
                    result.put("row_limit", properties.getQueryRowLimit());
                    result.put("truncated", truncated);
                    result.put("rows", rows);
                    try {
                        return objectMapper.writeValueAsString(result);
                    } catch (JsonProcessingException e) {
                        return "{\"status\":\"error\",\"message\":\"Failed to serialize result\"}";
                    }
                })
                .onErrorResume(e -> {
                    String errorMsg = e.getMessage() != null ? e.getMessage().toLowerCase() : "";
                    if (errorMsg.contains("timeout") || errorMsg.contains("statement timeout")
                            || errorMsg.contains("canceling statement")) {
                        log.warn("Database query timeout | sql_len={} | timeout_ms={}", sql.length(), timeoutMs);
                        return jsonResponse(Map.of(
                                "status", "error",
                                "message", "数据库查询超时（>" + timeoutMs + "ms），请缩小查询范围后重试。"
                        ));
                    }
                    log.error("Database query failed: {}", e.getMessage());
                    return jsonResponse(Map.of(
                            "status", "error",
                            "message", "数据库查询失败，请检查 SQL 语法和表结构。"
                    ));
                });
    }

    private boolean isPostgres() {
        try {
            String metadata = connectionFactory.getMetadata().getName();
            return metadata != null && metadata.toLowerCase().contains("postgres");
        } catch (Exception e) {
            return false;
        }
    }

    public Mono<String> upsertUserSetting(String userId, String settingKey, String settingValue) {
        if (userId == null || userId.isBlank()) {
            return jsonResponse(Map.of("status", "error", "message", "user_id 不能为空。"));
        }
        if (settingKey == null || settingKey.isBlank()) {
            return jsonResponse(Map.of("status", "error", "message", "setting_key 不能为空。"));
        }
        if (!McpServerProperties.ALLOWED_SETTING_KEYS.contains(settingKey.trim())) {
            return jsonResponse(Map.of(
                    "status", "error",
                    "message", "setting_key '" + settingKey + "' 不在允许的键列表中。"
            ));
        }

        DatabaseClient client = DatabaseClient.create(connectionFactory);
        String upsertSql = """
                INSERT INTO agent_user_settings (user_id, setting_key, setting_value)
                VALUES (:user_id, :setting_key, :setting_value)
                ON CONFLICT (user_id, setting_key)
                DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = NOW()
                """;

        return client.sql(upsertSql)
                .bind("user_id", userId.trim())
                .bind("setting_key", settingKey.trim())
                .bind("setting_value", settingValue.trim())
                .fetch()
                .rowsUpdated()
                .then(jsonResponse(Map.of("status", "success", "message", "设置已更新")))
                .onErrorResume(e -> {
                    log.error("Failed to upsert user setting: {}", e.getMessage());
                    return jsonResponse(Map.of("status", "error", "message", "写入设置失败"));
                });
    }

    private Mono<String> jsonResponse(Map<String, Object> payload) {
        try {
            return Mono.just(objectMapper.writeValueAsString(payload));
        } catch (JsonProcessingException e) {
            return Mono.just("{\"status\":\"error\",\"message\":\"Serialization error\"}");
        }
    }
}