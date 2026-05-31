package com.nanoagent.service.session;

import com.nanoagent.service.config.NanoAgentProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class LlmSessionStore {

    private static final Logger log = LoggerFactory.getLogger(LlmSessionStore.class);

    private final StringRedisTemplate redisTemplate;
    private final NanoAgentProperties properties;
    private final Map<String, Map<String, String>> localStore;
    private final SecretKeySpec encryptionKey;

    public LlmSessionStore(StringRedisTemplate redisTemplate, NanoAgentProperties properties) {
        this.redisTemplate = redisTemplate;
        this.properties = properties;
        this.localStore = new ConcurrentHashMap<>();
        this.encryptionKey = initEncryptionKey();
    }

    private SecretKeySpec initEncryptionKey() {
        String masterKey = properties.getSession().getMasterKey();
        if (masterKey == null || masterKey.isBlank()) {
            return null;
        }
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            byte[] keyBytes = sha.digest(masterKey.getBytes(StandardCharsets.UTF_8));
            byte[] key16 = new byte[16];
            System.arraycopy(keyBytes, 0, key16, 0, 16);
            return new SecretKeySpec(key16, "AES");
        } catch (Exception e) {
            log.warn("Failed to initialize encryption key, API keys will be stored in plaintext");
            return null;
        }
    }

    private String encrypt(String plaintext) {
        if (encryptionKey == null || plaintext == null) return plaintext;
        try {
            Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, encryptionKey);
            byte[] encrypted = cipher.doFinal(plaintext.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(encrypted);
        } catch (Exception e) {
            log.warn("Encryption failed, storing plaintext: {}", e.getMessage());
            return plaintext;
        }
    }

    private String decrypt(String ciphertext) {
        if (encryptionKey == null || ciphertext == null) return ciphertext;
        try {
            Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, encryptionKey);
            byte[] decoded = Base64.getDecoder().decode(ciphertext);
            return new String(cipher.doFinal(decoded), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return ciphertext;
        }
    }

    public String createSession(Map<String, String> llmProfile, String ownerId) {
        String sessionId = UUID.randomUUID().toString();
        Map<String, String> profile = new LinkedHashMap<>(llmProfile);

        if (profile.containsKey("api_key") && encryptionKey != null) {
            profile.put("api_key", encrypt(profile.get("api_key")));
        }
        profile.put("owner_id", ownerId);

        int ttl = properties.getSession().getTtlSeconds();
        String key = sessionKey(sessionId);

        try {
            redisTemplate.opsForHash().putAll(key, profile);
            redisTemplate.expire(key, Duration.ofSeconds(ttl));
        } catch (Exception e) {
            log.warn("Redis store failed, using local fallback: {}", e.getMessage());
            localStore.put(sessionId, profile);
        }

        return sessionId;
    }

    public Map<String, String> getProfile(String sessionId) {
        Map<Object, Object> raw;
        try {
            raw = redisTemplate.opsForHash().entries(sessionKey(sessionId));
        } catch (Exception e) {
            raw = new LinkedHashMap<>();
            Map<String, String> local = localStore.get(sessionId);
            if (local != null) {
                raw.putAll(local);
            }
        }

        if (raw == null || raw.isEmpty()) return null;

        Map<String, String> profile = new LinkedHashMap<>();
        for (Map.Entry<Object, Object> entry : raw.entrySet()) {
            profile.put(String.valueOf(entry.getKey()), String.valueOf(entry.getValue()));
        }

        if (profile.containsKey("api_key") && encryptionKey != null) {
            String decrypted = decrypt(profile.get("api_key"));
            if (decrypted != null) {
                profile.put("api_key", decrypted);
            }
        }

        return profile;
    }

    public String getOwnerId(String sessionId) {
        try {
            Object raw = redisTemplate.opsForHash().get(sessionKey(sessionId), "owner_id");
            if (raw != null) return String.valueOf(raw);
        } catch (Exception e) {
            Map<String, String> local = localStore.get(sessionId);
            if (local != null) return local.get("owner_id");
        }
        return "";
    }

    public boolean deleteSession(String sessionId, String ownerId) {
        try {
            String key = sessionKey(sessionId);
            Object storedOwner = redisTemplate.opsForHash().get(key, "owner_id");
            if (storedOwner != null && !ownerId.equals(String.valueOf(storedOwner))) {
                return false;
            }
            return Boolean.TRUE.equals(redisTemplate.delete(key));
        } catch (Exception e) {
            boolean deleted = localStore.remove(sessionId) != null;
            return deleted;
        }
    }

    public boolean validateSession(String sessionId) {
        try {
            return Boolean.TRUE.equals(redisTemplate.hasKey(sessionKey(sessionId)));
        } catch (Exception e) {
            return localStore.containsKey(sessionId);
        }
    }

    private String sessionKey(String sessionId) {
        return "nanoagent:session:" + sessionId;
    }
}