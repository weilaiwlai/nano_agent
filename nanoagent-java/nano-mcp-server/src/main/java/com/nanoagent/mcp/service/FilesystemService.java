package com.nanoagent.mcp.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class FilesystemService {

    private static final Logger log = LoggerFactory.getLogger(FilesystemService.class);

    private final List<Path> allowedDirs;
    private final ObjectMapper objectMapper;

    public FilesystemService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
        String agentDataPath = System.getenv().getOrDefault("AGENT_DATA_PATH",
                Path.of("agentdata").toAbsolutePath().toString());
        this.allowedDirs = List.of(Path.of(agentDataPath).toAbsolutePath().normalize());
        log.info("FilesystemService initialized | allowed_dirs={}", allowedDirs);
    }

    public Mono<String> isPathAllowed(String path) {
        Path absPath = Path.of(path).toAbsolutePath().normalize();
        boolean allowed = allowedDirs.stream().anyMatch(dir -> absPath.startsWith(dir));
        return jsonResponse(Map.of("allowed", allowed));
    }

    public Mono<String> readFile(String path) {
        return ensurePathAllowed(path)
                .flatMap(allowed -> {
                    if (!allowed) {
                        return jsonResponse(Map.of("status", "error", "message", "路径不被允许"));
                    }
                    return Mono.fromCallable(() -> {
                        String content = Files.readString(Path.of(path));
                        return jsonResponseSync(Map.of("status", "success", "content", content));
                    }).subscribeOn(Schedulers.boundedElastic());
                });
    }

    public Mono<String> writeFile(String path, String content) {
        Path resolvedPath = resolvePath(path);
        return ensurePathAllowed(resolvedPath.toString())
                .flatMap(allowed -> {
                    if (!allowed) {
                        return jsonResponse(Map.of("status", "error", "message", "路径不被允许"));
                    }
                    return Mono.fromCallable(() -> {
                        Files.createDirectories(resolvedPath.getParent());
                        Files.writeString(resolvedPath, content);
                        return jsonResponseSync(Map.of("status", "success"));
                    }).subscribeOn(Schedulers.boundedElastic());
                });
    }

    public Mono<String> createDirectory(String path) {
        return ensurePathAllowed(path)
                .flatMap(allowed -> {
                    if (!allowed) {
                        return jsonResponse(Map.of("status", "error", "message", "路径不被允许"));
                    }
                    return Mono.fromCallable(() -> {
                        Files.createDirectories(Path.of(path));
                        return jsonResponseSync(Map.of("status", "success"));
                    }).subscribeOn(Schedulers.boundedElastic());
                });
    }

    public Mono<String> moveFile(String path, String newPath) {
        return ensurePathAllowed(path)
                .flatMap(allowed -> {
                    if (!allowed) {
                        return jsonResponse(Map.of("status", "error", "message", "路径不被允许"));
                    }
                    return ensurePathAllowed(newPath)
                            .flatMap(newAllowed -> {
                                if (!newAllowed) {
                                    return jsonResponse(Map.of("status", "error", "message", "目标路径不被允许"));
                                }
                                return Mono.fromCallable(() -> {
                                    Files.move(Path.of(path), Path.of(newPath));
                                    return jsonResponseSync(Map.of("status", "success"));
                                }).subscribeOn(Schedulers.boundedElastic());
                            });
                });
    }

    public Mono<String> editFile(String path, String content) {
        return writeFile(path, content);
    }

    public Mono<String> listAllowedDirectories() {
        List<String> dirs = allowedDirs.stream()
                .map(Path::toString)
                .toList();
        return jsonResponse(Map.of("status", "success", "directories", dirs));
    }

    private Mono<Boolean> ensurePathAllowed(String path) {
        Path absPath = Path.of(path).toAbsolutePath().normalize();
        boolean allowed = allowedDirs.stream().anyMatch(dir -> absPath.startsWith(dir));
        if (!allowed) {
            log.warn("Path not allowed: {}", path);
        }
        return Mono.just(allowed);
    }

    private Path resolvePath(String path) {
        Path p = Path.of(path);
        if (!p.isAbsolute()) {
            return allowedDirs.get(0).resolve(p).toAbsolutePath().normalize();
        }
        return p.toAbsolutePath().normalize();
    }

    private Mono<String> jsonResponse(Map<String, Object> payload) {
        try {
            return Mono.just(objectMapper.writeValueAsString(payload));
        } catch (JsonProcessingException e) {
            return Mono.just("{\"status\":\"error\",\"message\":\"Serialization error\"}");
        }
    }

    private String jsonResponseSync(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException e) {
            return "{\"status\":\"error\",\"message\":\"Serialization error\"}";
        }
    }
}