package com.nanoagent.service.auth;

import com.nanoagent.service.config.NanoAgentProperties;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.Key;
import java.security.MessageDigest;
import java.security.PublicKey;

@Component
public class AuthService {

    private static final Logger log = LoggerFactory.getLogger(AuthService.class);

    private final NanoAgentProperties properties;

    public AuthService(NanoAgentProperties properties) {
        this.properties = properties;
    }

    public AuthContext authenticate(HttpServletRequest request) {
        NanoAgentProperties.AuthConfig auth = properties.getAuth();

        if (!auth.isRequireApiAuth()) {
            return AuthContext.builder()
                    .authType(AuthContext.AuthType.DISABLED)
                    .subject(null)
                    .build();
        }

        String bearerToken = extractBearerToken(request);

        if (bearerToken == null || bearerToken.isBlank()) {
            if (auth.isAllowApiKeyFallback() && !auth.getAllowedApiKeys().isEmpty()) {
                String apiKey = request.getHeader("x-api-key");
                if (isAllowedApiKey(apiKey, auth)) {
                    return AuthContext.builder()
                            .authType(AuthContext.AuthType.API_KEY)
                            .subject(null)
                            .build();
                }
            }
            throw new AuthException(401, "缺少认证信息");
        }

        if (looksLikeJwt(bearerToken)) {
            try {
                Claims claims = decodeJwtClaims(bearerToken, auth);
                String subject = claims.getSubject();
                if (subject == null || subject.isBlank()) {
                    throw new AuthException(401, "JWT claims 缺少有效的 sub 字段");
                }
                return AuthContext.builder()
                        .authType(AuthContext.AuthType.JWT)
                        .subject(subject.trim())
                        .build();
            } catch (AuthException e) {
                throw e;
            } catch (Exception e) {
                throw new AuthException(401, "JWT 校验失败，请重新登录");
            }
        }

        if (auth.isAllowApiKeyFallback() && isAllowedApiKey(bearerToken, auth)) {
            return AuthContext.builder()
                    .authType(AuthContext.AuthType.API_KEY)
                    .subject(null)
                    .build();
        }

        throw new AuthException(401, "认证信息格式错误");
    }

    public String resolveEffectiveUserId(String tokenSubject, String clientUserId) {
        NanoAgentProperties.AuthConfig auth = properties.getAuth();
        if (!auth.isRequireUserSub()) {
            return clientUserId.trim();
        }
        if (tokenSubject.equals(clientUserId.trim())) {
            return clientUserId.trim();
        }
        throw new AuthException(403,
                "用户身份不匹配：JWT subject (" + tokenSubject + ") 与请求 user_id (" + clientUserId + ") 不一致");
    }

    private String extractBearerToken(HttpServletRequest request) {
        String authorization = request.getHeader(HttpHeaders.AUTHORIZATION);
        if (authorization == null || authorization.isBlank()) {
            return null;
        }
        String[] parts = authorization.trim().split(" ", 2);
        if (parts.length != 2 || !"bearer".equalsIgnoreCase(parts[0])) {
            return null;
        }
        return parts[1].trim();
    }

    private boolean looksLikeJwt(String token) {
        return token.chars().filter(c -> c == '.').count() == 2;
    }

    private boolean isAllowedApiKey(String candidate, NanoAgentProperties.AuthConfig auth) {
        if (candidate == null || candidate.isBlank()) {
            return false;
        }
        return auth.getAllowedApiKeys().stream()
                .anyMatch(allowed -> MessageDigest.isEqual(
                        candidate.getBytes(StandardCharsets.UTF_8),
                        allowed.getBytes(StandardCharsets.UTF_8)));
    }

    private Claims decodeJwtClaims(String token, NanoAgentProperties.AuthConfig auth) {
        String jwksUrl = auth.getJwtJwksUrl();
        if (jwksUrl != null && !jwksUrl.isBlank()) {
            JwksKeyResolver keyResolver = new JwksKeyResolver(jwksUrl);
            Key signingKey = keyResolver.getSigningKeyFromJwt(token);
            if (signingKey != null) {
                return Jwts.parser()
                        .verifyWith((PublicKey) signingKey)
                        .build()
                        .parseSignedClaims(token)
                        .getPayload();
            }
        }

        String secret = auth.getJwtHs256Secret();
        if (secret != null && !secret.isBlank()) {
            SecretKey key = new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
            return Jwts.parser()
                    .verifyWith(key)
                    .build()
                    .parseSignedClaims(token)
                    .getPayload();
        }

        throw new AuthException(503, "JWT 校验未完成配置（缺少 JWT_JWKS_URL 或 JWT_HS256_SECRET）");
    }
}