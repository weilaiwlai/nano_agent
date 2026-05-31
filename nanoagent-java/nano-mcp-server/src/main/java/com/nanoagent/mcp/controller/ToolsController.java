package com.nanoagent.mcp.controller;

import com.nanoagent.mcp.service.ToolService;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RestController
@RequestMapping("/tools")
public class ToolsController {

    private static final Logger log = LoggerFactory.getLogger(ToolsController.class);

    private final ToolService toolService;
    public ToolsController(ToolService toolService) {
        this.toolService = toolService;
    }

    @GetMapping("/health")
    public Mono<Map<String, String>> health() {
        return Mono.just(Map.of("status", "ok", "service", "mcp_server"));
    }

    @PostMapping("/search")
    public Mono<String> search(@RequestBody Map<String, String> request) {
        String query = request.getOrDefault("query", "");
        log.info("Tool call | search | query_len={}", query.length());
        return toolService.search(query);
    }

    @PostMapping("/get_current_time")
    public Mono<String> getCurrentTime(@RequestBody(required = false) Map<String, String> request) {
        String timezone = request != null ? request.getOrDefault("timezone", "Asia/Shanghai") : "Asia/Shanghai";
        log.info("Tool call | get_current_time | timezone={}", timezone);
        return toolService.getCurrentTime(timezone);
    }

    @PostMapping("/query_database")
    public Mono<String> queryDatabase(@RequestBody Map<String, String> request) {
        String sql = request.getOrDefault("sql", "");
        log.info("Tool call | query_database | sql_len={}", sql.length());
        return toolService.queryDatabase(sql);
    }

    @PostMapping("/send_report")
    public Mono<String> sendReport(@RequestBody Map<String, String> request) {
        String email = request.getOrDefault("email", "");
        String content = request.getOrDefault("content", "");
        log.info("Tool call | send_report | email={} | content_len={}",
                maskEmail(email), content.length());
        return toolService.sendReport(email, content);
    }

    @PostMapping("/upsert_user_setting")
    public Mono<String> upsertUserSetting(@RequestBody Map<String, String> request) {
        String userId = request.getOrDefault("user_id", "");
        String settingKey = request.getOrDefault("setting_key", "");
        String settingValue = request.getOrDefault("setting_value", "");
        log.info("Tool call | upsert_user_setting | user_id={} | key={}", userId, settingKey);
        return toolService.upsertUserSetting(userId, settingKey, settingValue);
    }

    @PostMapping("/is_path_allowed")
    public Mono<String> isPathAllowed(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        log.info("Tool call | is_path_allowed | path={}", path);
        return toolService.isPathAllowed(path);
    }

    @PostMapping("/read_file")
    public Mono<String> readFile(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        log.info("Tool call | read_file | path={}", path);
        return toolService.readFile(path);
    }

    @PostMapping("/write_file")
    public Mono<String> writeFile(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        String content = request.getOrDefault("content", "");
        log.info("Tool call | write_file | path={} | content_len={}", path, content.length());
        return toolService.writeFile(path, content);
    }

    @PostMapping("/create_directory")
    public Mono<String> createDirectory(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        log.info("Tool call | create_directory | path={}", path);
        return toolService.createDirectory(path);
    }

    @PostMapping("/move_file")
    public Mono<String> moveFile(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        String newPath = request.getOrDefault("new_path", "");
        log.info("Tool call | move_file | path={} | new_path={}", path, newPath);
        return toolService.moveFile(path, newPath);
    }

    @PostMapping("/edit_file")
    public Mono<String> editFile(@RequestBody Map<String, String> request) {
        String path = request.getOrDefault("path", "");
        String content = request.getOrDefault("content", "");
        log.info("Tool call | edit_file | path={} | content_len={}", path, content.length());
        return toolService.editFile(path, content);
    }

    @PostMapping("/list_allowed_directories")
    public Mono<String> listAllowedDirectories() {
        log.info("Tool call | list_allowed_directories");
        return toolService.listAllowedDirectories();
    }

    @PostMapping("/call")
    public Mono<String> callTool(@RequestBody Map<String, Object> request) {
        String toolName = String.valueOf(request.getOrDefault("name", ""));
        @SuppressWarnings("unchecked")
        Map<String, Object> arguments = (Map<String, Object>) request.getOrDefault("arguments", Map.of());

        log.info("Tool call | unified | name={} | args_keys={}", toolName, arguments.keySet());

        return switch (toolName) {
            case "search" -> {
                String query = String.valueOf(arguments.getOrDefault("query", ""));
                yield toolService.search(query);
            }
            case "get_current_time", "tool_get_current_time" -> {
                String tz = String.valueOf(arguments.getOrDefault("timezone", "Asia/Shanghai"));
                yield toolService.getCurrentTime(tz);
            }
            case "query_database", "tool_query_database" -> {
                String sql = String.valueOf(arguments.getOrDefault("sql", ""));
                yield toolService.queryDatabase(sql);
            }
            case "send_report", "tool_send_report" -> {
                String email = String.valueOf(arguments.getOrDefault("email", ""));
                String content = String.valueOf(arguments.getOrDefault("content", ""));
                yield toolService.sendReport(email, content);
            }
            case "upsert_user_setting", "tool_upsert_user_setting" -> {
                String userId = String.valueOf(arguments.getOrDefault("user_id", ""));
                String key = String.valueOf(arguments.getOrDefault("setting_key", ""));
                String value = String.valueOf(arguments.getOrDefault("setting_value", ""));
                yield toolService.upsertUserSetting(userId, key, value);
            }
            case "is_path_allowed", "tool_is_path_allowed" -> {
                String path = String.valueOf(arguments.getOrDefault("path", ""));
                yield toolService.isPathAllowed(path);
            }
            case "read_file", "tool_read_file" -> {
                String path = String.valueOf(arguments.getOrDefault("path", ""));
                yield toolService.readFile(path);
            }
            case "write_file", "tool_write_file" -> {
                String path = String.valueOf(arguments.getOrDefault("path", ""));
                String content = String.valueOf(arguments.getOrDefault("content", ""));
                yield toolService.writeFile(path, content);
            }
            case "create_directory", "tool_create_directory" -> {
                String path = String.valueOf(arguments.getOrDefault("path", ""));
                yield toolService.createDirectory(path);
            }
            case "move_file", "tool_move_file" -> {
                String src = String.valueOf(arguments.getOrDefault("src", ""));
                String dst = String.valueOf(arguments.getOrDefault("dst", ""));
                yield toolService.moveFile(src, dst);
            }
            case "edit_file", "tool_edit_file" -> {
                String path = String.valueOf(arguments.getOrDefault("path", ""));
                String content = String.valueOf(arguments.getOrDefault("content", ""));
                yield toolService.editFile(path, content);
            }
            case "list_allowed_directories", "tool_list_allowed_directories" ->
                toolService.listAllowedDirectories();
            default -> Mono.just("{\"status\":\"error\",\"message\":\"Unknown tool: " + toolName + "\"}");
        };
    }

    private String maskEmail(String email) {
        if (email == null || email.isBlank()) return "***";
        int atIndex = email.indexOf('@');
        if (atIndex <= 0) return "***";
        return email.substring(0, Math.min(2, atIndex)) + "***" + email.substring(atIndex);
    }
}