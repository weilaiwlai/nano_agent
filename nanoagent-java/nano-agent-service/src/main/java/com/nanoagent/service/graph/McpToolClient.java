package com.nanoagent.service.graph;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanoagent.service.config.NanoAgentProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class McpToolClient {

    private static final Logger log = LoggerFactory.getLogger(McpToolClient.class);

    private final String baseUrl;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public McpToolClient(NanoAgentProperties properties) {
        String configuredUrl = properties.getMcp().getBaseUrl();
        this.baseUrl = configuredUrl != null && !configuredUrl.isBlank() ? configuredUrl : "http://localhost:8000";
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    public String callTool(String toolName, Map<String, Object> arguments) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("name", toolName);
            body.put("arguments", arguments != null ? arguments : Map.of());

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(
                    baseUrl + "/tools/call", request, Map.class);

            if (response.getBody() != null && response.getBody().containsKey("content")) {
                Object content = response.getBody().get("content");
                return content instanceof String ? (String) content : objectMapper.writeValueAsString(content);
            }
            return objectMapper.writeValueAsString(response.getBody());
        } catch (Exception e) {
            log.error("MCP tool call error | tool={} | error={}", toolName, e.getMessage());
            return "{\"status\":\"error\",\"tool\":\"" + toolName + "\",\"message\":\"" + e.getMessage() + "\"}";
        }
    }

    public String searchTool(String query) {
        return callTool("search", Map.of("query", query));
    }

    public String getCurrentTimeTool() {
        return callTool("get_current_time", Map.of());
    }

    public String queryDatabaseTool(String sql) {
        return callTool("query_database", Map.of("sql", sql));
    }

    public String readFileTool(String path) {
        return callTool("read_file", Map.of("path", path));
    }

    public String writeFileTool(String path, String content) {
        return callTool("write_file", Map.of("path", path, "content", content));
    }

    public String listAllowedDirectoriesTool() {
        return callTool("list_allowed_directories", Map.of());
    }

    public String sendReportTool(String email, String content) {
        return callTool("send_report", Map.of("email", email, "content", content));
    }

    public String upsertUserSettingTool(String userId, String settingKey, String settingValue) {
        return callTool("upsert_user_setting", Map.of("user_id", userId, "setting_key", settingKey, "setting_value", settingValue));
    }
}