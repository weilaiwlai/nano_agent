package com.nanoagent.mcp.security;

import com.nanoagent.mcp.config.McpServerProperties;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import org.springframework.web.server.WebFilter;
import org.springframework.web.server.WebFilterChain;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

@Component
public class ServiceAuthFilter implements WebFilter {

    private final McpServerProperties properties;
    public ServiceAuthFilter(McpServerProperties properties) {
        this.properties = properties;
    }

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, WebFilterChain chain) {
        if (!properties.isRequireAuth()) {
            return chain.filter(exchange);
        }

        ServerHttpRequest request = exchange.getRequest();
        String path = request.getURI().getPath();

        if (!isProtectedPath(path)) {
            return chain.filter(exchange);
        }

        if (!isAuthorizedServiceRequest(request)) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            String body = "{\"status\":\"error\",\"message\":\"服务间鉴权失败\"}";
            return exchange.getResponse()
                    .writeWith(Mono.just(exchange.getResponse()
                            .bufferFactory()
                            .wrap(body.getBytes())));
        }

        return chain.filter(exchange);
    }

    private boolean isProtectedPath(String path) {
        String normalized = path.replaceAll("/+$", "");
        if (normalized.isEmpty()) normalized = "/";
        return normalized.startsWith("/tools/") || normalized.startsWith("/mcp");
    }

    private boolean isAuthorizedServiceRequest(ServerHttpRequest request) {
        String token = extractServiceToken(request);
        if (token == null || token.isBlank()) {
            return false;
        }
        return MessageDigest.isEqual(
                token.getBytes(StandardCharsets.UTF_8),
                properties.getServiceToken().getBytes(StandardCharsets.UTF_8));
    }

    private String extractServiceToken(ServerHttpRequest request) {
        String headerToken = request.getHeaders().getFirst("X-Service-Token");
        if (headerToken != null && !headerToken.isBlank()) {
            return headerToken.trim();
        }

        String authorization = request.getHeaders().getFirst("Authorization");
        if (authorization == null || authorization.isBlank()) {
            return "";
        }
        String[] parts = authorization.trim().split(" ", 2);
        if (parts.length != 2 || !"bearer".equalsIgnoreCase(parts[0])) {
            return "";
        }
        return parts[1].trim();
    }
}