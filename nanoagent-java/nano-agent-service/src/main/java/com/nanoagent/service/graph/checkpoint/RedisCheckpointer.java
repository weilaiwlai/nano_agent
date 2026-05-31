package com.nanoagent.service.graph.checkpoint;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Map;

public class RedisCheckpointer implements GraphCheckpointer {

    private static final Logger log = LoggerFactory.getLogger(RedisCheckpointer.class);

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final String prefix;

    public RedisCheckpointer(RedisConnectionFactory connectionFactory, String prefix) {
        this.objectMapper = new ObjectMapper();
        this.prefix = prefix != null && !prefix.isBlank() ? prefix : "nanoagent:checkpoint";
        this.redisTemplate = new StringRedisTemplate(connectionFactory);
        log.info("Redis checkpointer initialized | prefix={}", this.prefix);
    }

    private String key(String threadId) {
        return prefix + ":" + threadId;
    }

    @Override
    public void put(String threadId, Map<String, Object> state) {
        try {
            String json = objectMapper.writeValueAsString(state);
            redisTemplate.opsForValue().set(key(threadId), json);
        } catch (Exception e) {
            log.error("Redis checkpointer put error | threadId={} | error={}", threadId, e.getMessage());
        }
    }

    @Override
    public Map<String, Object> get(String threadId) {
        try {
            String json = redisTemplate.opsForValue().get(key(threadId));
            if (json == null) return null;
            return objectMapper.readValue(json, new TypeReference<Map<String, Object>>() {});
        } catch (Exception e) {
            log.error("Redis checkpointer get error | threadId={} | error={}", threadId, e.getMessage());
            return null;
        }
    }

    @Override
    public void delete(String threadId) {
        try {
            redisTemplate.delete(key(threadId));
        } catch (Exception e) {
            log.error("Redis checkpointer delete error | threadId={} | error={}", threadId, e.getMessage());
        }
    }

    @Override
    public boolean exists(String threadId) {
        try {
            Boolean exists = redisTemplate.hasKey(key(threadId));
            return Boolean.TRUE.equals(exists);
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public String getBackendName() {
        return "redis";
    }
}