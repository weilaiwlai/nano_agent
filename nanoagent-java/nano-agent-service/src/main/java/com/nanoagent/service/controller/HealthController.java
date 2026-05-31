package com.nanoagent.service.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class HealthController {

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "nanoagent");
    }

    @GetMapping("/health/mcp")
    public Map<String, String> healthMcp() {
        return Map.of("status", "ok", "service", "mcp_proxy");
    }
}