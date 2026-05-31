package com.nanoagent.service.graph.checkpoint;

import com.nanoagent.service.config.NanoAgentProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;

@Component
public class CheckpointerFactory {

    private static final Logger log = LoggerFactory.getLogger(CheckpointerFactory.class);

    private final NanoAgentProperties properties;

    @Autowired(required = false)
    private DataSource dataSource;

    @Autowired(required = false)
    private RedisConnectionFactory redisConnectionFactory;

    public CheckpointerFactory(NanoAgentProperties properties) {
        this.properties = properties;
    }

    public GraphCheckpointer createCheckpointer() {
        String backend = properties.getGraphCheckpointerBackend();
        if (backend == null || backend.isBlank()) {
            backend = "memory";
        }

        log.info("Creating graph checkpointer | backend={}", backend);

        return switch (backend.toLowerCase()) {
            case "redis" -> {
                if (redisConnectionFactory == null) {
                    log.warn("RedisConnectionFactory not available, falling back to memory checkpointer");
                    yield new MemoryCheckpointer();
                }
                String prefix = properties.getGraphCheckpointerPrefix();
                yield new RedisCheckpointer(redisConnectionFactory, prefix);
            }
            case "postgres", "postgresql" -> {
                if (dataSource == null) {
                    log.warn("DataSource not available, falling back to memory checkpointer");
                    yield new MemoryCheckpointer();
                }
                String tableName = properties.getGraphCheckpointerTableName();
                yield new PostgresCheckpointer(dataSource, tableName);
            }
            default -> new MemoryCheckpointer();
        };
    }
}
