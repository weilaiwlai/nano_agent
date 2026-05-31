package com.nanoagent.service.auth;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.jsonwebtoken.Jwts;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.client.RestTemplate;

import java.math.BigInteger;
import java.security.Key;
import java.security.KeyFactory;
import java.security.PublicKey;
import java.security.spec.RSAPublicKeySpec;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class JwksKeyResolver {

    private static final Logger log = LoggerFactory.getLogger(JwksKeyResolver.class);

    private final String jwksUrl;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    private final Map<String, Key> keyCache = new ConcurrentHashMap<>();
    private volatile long lastFetchTime = 0;
    private static final long CACHE_TTL_MS = TimeUnit.HOURS.toMillis(1);

    public JwksKeyResolver(String jwksUrl) {
        this.jwksUrl = jwksUrl;
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    public Key getSigningKey(String kid) {
        if (shouldRefreshCache()) {
            refreshKeys();
        }
        return keyCache.get(kid);
    }

    public Key getSigningKeyFromJwt(String token) {
        try {
            String[] parts = token.split("\\.");
            if (parts.length < 2) return null;

            String headerJson = new String(Base64.getUrlDecoder().decode(parts[0]));
            Map<String, Object> header = objectMapper.readValue(headerJson, Map.class);
            String kid = (String) header.get("kid");

            if (kid != null) {
                Key key = getSigningKey(kid);
                if (key != null) return key;
            }

            refreshKeys();
            if (kid != null) {
                return getSigningKey(kid);
            }

            if (!keyCache.isEmpty()) {
                return keyCache.values().iterator().next();
            }
        } catch (Exception e) {
            log.warn("Failed to extract signing key from JWT: {}", e.getMessage());
        }
        return null;
    }

    private boolean shouldRefreshCache() {
        return keyCache.isEmpty() || System.currentTimeMillis() - lastFetchTime > CACHE_TTL_MS;
    }

    private void refreshKeys() {
        try {
            Map<String, Object> response = restTemplate.getForObject(jwksUrl, Map.class);
            if (response == null || !response.containsKey("keys")) {
                log.warn("JWKS response missing 'keys' field");
                return;
            }

            List<Map<String, Object>> keys = (List<Map<String, Object>>) response.get("keys");
            for (Map<String, Object> jwk : keys) {
                String kid = (String) jwk.get("kid");
                String kty = (String) jwk.get("kty");

                if (kid == null || kty == null) continue;

                try {
                    Key key = parseJwk(jwk);
                    if (key != null) {
                        keyCache.put(kid, key);
                        log.debug("Cached JWK key: kid={}, kty={}", kid, kty);
                    }
                } catch (Exception e) {
                    log.warn("Failed to parse JWK key kid={}: {}", kid, e.getMessage());
                }
            }

            lastFetchTime = System.currentTimeMillis();
            log.info("JWKS keys refreshed | count={} | url={}", keyCache.size(), jwksUrl);
        } catch (Exception e) {
            log.error("Failed to fetch JWKS from {}: {}", jwksUrl, e.getMessage());
        }
    }

    private Key parseJwk(Map<String, Object> jwk) throws Exception {
        String kty = (String) jwk.get("kty");

        if ("RSA".equals(kty)) {
            String n = (String) jwk.get("n");
            String e = (String) jwk.get("e");

            if (n == null || e == null) return null;

            BigInteger modulus = new BigInteger(1, Base64.getUrlDecoder().decode(n));
            BigInteger exponent = new BigInteger(1, Base64.getUrlDecoder().decode(e));

            RSAPublicKeySpec spec = new RSAPublicKeySpec(modulus, exponent);
            KeyFactory keyFactory = KeyFactory.getInstance("RSA");
            return keyFactory.generatePublic(spec);
        }

        if ("EC".equals(kty)) {
            String crv = (String) jwk.get("crv");
            String x = (String) jwk.get("x");
            String y = (String) jwk.get("y");

            if (crv == null || x == null || y == null) return null;

            java.security.spec.ECPoint ecPoint = new java.security.spec.ECPoint(
                    new BigInteger(1, Base64.getUrlDecoder().decode(x)),
                    new BigInteger(1, Base64.getUrlDecoder().decode(y)));

            java.security.spec.ECParameterSpec ecSpec = getEcParameterSpec(crv);
            java.security.spec.ECPublicKeySpec ecKeySpec = new java.security.spec.ECPublicKeySpec(ecPoint, ecSpec);
            KeyFactory keyFactory = KeyFactory.getInstance("EC");
            return keyFactory.generatePublic(ecKeySpec);
        }

        log.warn("Unsupported JWK key type: {}", kty);
        return null;
    }

    private java.security.spec.ECParameterSpec getEcParameterSpec(String crv) throws Exception {
        return switch (crv) {
            case "P-256" -> {
                java.security.AlgorithmParameters params = java.security.AlgorithmParameters.getInstance("EC");
                params.init(new java.security.spec.ECGenParameterSpec("secp256r1"));
                yield params.getParameterSpec(java.security.spec.ECParameterSpec.class);
            }
            case "P-384" -> {
                java.security.AlgorithmParameters params = java.security.AlgorithmParameters.getInstance("EC");
                params.init(new java.security.spec.ECGenParameterSpec("secp384r1"));
                yield params.getParameterSpec(java.security.spec.ECParameterSpec.class);
            }
            case "P-521" -> {
                java.security.AlgorithmParameters params = java.security.AlgorithmParameters.getInstance("EC");
                params.init(new java.security.spec.ECGenParameterSpec("secp521r1"));
                yield params.getParameterSpec(java.security.spec.ECParameterSpec.class);
            }
            default -> throw new IllegalArgumentException("Unsupported EC curve: " + crv);
        };
    }
}