package com.nanoagent.mcp.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class ToolService {

    private static final Logger log = LoggerFactory.getLogger(ToolService.class);

    private final DatabaseService databaseService;
    private final EmailService emailService;
    private final FilesystemService filesystemService;
    private final ObjectMapper objectMapper;
    public ToolService(DatabaseService databaseService, EmailService emailService, FilesystemService filesystemService, ObjectMapper objectMapper) {
        this.databaseService = databaseService;
        this.emailService = emailService;
        this.filesystemService = filesystemService;
        this.objectMapper = objectMapper;
    }

    public Mono<String> search(String query) {
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");
        if (tavilyApiKey == null || tavilyApiKey.isBlank()) {
            return jsonResponse(Map.of(
                    "status", "error",
                    "message", "未配置 TAVILY_API_KEY 环境变量，无法进行网络搜索。"
            ));
        }

        WebClient client = WebClient.builder()
                .baseUrl("https://api.tavily.com")
                .defaultHeader("Content-Type", "application/json")
                .build();

        return client.post()
                .uri("/search")
                .bodyValue(Map.of(
                        "api_key", tavilyApiKey,
                        "query", query,
                        "search_depth", "basic",
                        "max_results", 5
                ))
                .retrieve()
                .bodyToMono(Map.class)
                .map(result -> {
                    try {
                        return objectMapper.writeValueAsString(Map.of(
                                "status", "success",
                                "results", result.getOrDefault("results", "")
                        ));
                    } catch (JsonProcessingException e) {
                        return "{\"status\":\"error\",\"message\":\"Failed to serialize search results\"}";
                    }
                })
                .onErrorResume(e -> {
                    log.error("Search failed: {}", e.getMessage());
                    return jsonResponse(Map.of("status", "error", "message", "搜索失败：" + e.getMessage()));
                });
    }

    public Mono<String> getCurrentTime(String timezone) {
        String tz = timezone != null ? timezone : "Asia/Shanghai";
        Map<String, String> timezoneMapping = Map.of(
                "中国", "Asia/Shanghai",
                "北京", "Asia/Shanghai",
                "上海", "Asia/Shanghai",
                "China", "Asia/Shanghai",
                "Beijing", "Asia/Shanghai",
                "CST", "Asia/Shanghai"
        );
        String actualTimezone = timezoneMapping.getOrDefault(tz, tz);

        try {
            ZoneId zoneId = ZoneId.of(actualTimezone);
            ZonedDateTime now = ZonedDateTime.now(zoneId);
            String formatted = now.format(DateTimeFormatter.ofPattern("yyyy年MM月dd日 HH:mm:ss z"));
            log.info("Get time | timezone={} | time={}", tz, formatted);
            return Mono.just(tz + " 的当前时间是: " + formatted);
        } catch (Exception e) {
            return Mono.just("错误: 未知的时区 '" + tz + "'。请提供有效的时区名称。");
        }
    }

    public Mono<String> queryDatabase(String sql) {
        return databaseService.queryDatabase(sql);
    }

    public Mono<String> sendReport(String email, String content) {
        return emailService.sendReport(email, content);
    }

    public Mono<String> upsertUserSetting(String userId, String settingKey, String settingValue) {
        return databaseService.upsertUserSetting(userId, settingKey, settingValue);
    }

    public Mono<String> isPathAllowed(String path) {
        return filesystemService.isPathAllowed(path);
    }

    public Mono<String> readFile(String path) {
        return filesystemService.readFile(path);
    }

    public Mono<String> writeFile(String path, String content) {
        return filesystemService.writeFile(path, content);
    }

    public Mono<String> createDirectory(String path) {
        return filesystemService.createDirectory(path);
    }

    public Mono<String> moveFile(String path, String newPath) {
        return filesystemService.moveFile(path, newPath);
    }

    public Mono<String> editFile(String path, String content) {
        return filesystemService.editFile(path, content);
    }

    public Mono<String> listAllowedDirectories() {
        return filesystemService.listAllowedDirectories();
    }

    private Mono<String> jsonResponse(Map<String, Object> payload) {
        try {
            return Mono.just(objectMapper.writeValueAsString(payload));
        } catch (JsonProcessingException e) {
            return Mono.just("{\"status\":\"error\",\"message\":\"Serialization error\"}");
        }
    }
}