package com.nanoagent.mcp.config;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.PropertySource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DotenvEnvironmentPostProcessor implements EnvironmentPostProcessor {

    private static final String DOTENV_SOURCE_NAME = "dotenv";
    private static final Pattern DB_URL_PATTERN = Pattern.compile(
            "(?:postgresql\\+asyncpg|postgresql)://([^:]*):([^@]*)@([^:/]+):(\\d+)/([^?]+)");

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> dotenv = loadDotenv();
        if (!dotenv.isEmpty()) {
            expandDatabaseConfig(dotenv);
            PropertySource<?> source = new MapPropertySource(DOTENV_SOURCE_NAME, dotenv);
            environment.getPropertySources().addFirst(source);
        }
    }

    private void expandDatabaseConfig(Map<String, Object> properties) {
        String dbUrl = (String) properties.get("DB_URL");
        if (dbUrl != null && !dbUrl.isBlank()) {
            Matcher m = DB_URL_PATTERN.matcher(dbUrl.trim());
            if (m.matches()) {
                if (!properties.containsKey("DATABASE_USER")) {
                    properties.put("DATABASE_USER", m.group(1));
                }
                if (!properties.containsKey("DATABASE_PASSWORD")) {
                    properties.put("DATABASE_PASSWORD", m.group(2));
                }
                if (!properties.containsKey("DATABASE_HOST")) {
                    properties.put("DATABASE_HOST", m.group(3));
                }
                if (!properties.containsKey("DATABASE_PORT")) {
                    properties.put("DATABASE_PORT", m.group(4));
                }
                if (!properties.containsKey("DATABASE_NAME")) {
                    properties.put("DATABASE_NAME", m.group(5));
                }
                String r2dbcUrl = "r2dbc:postgresql://" + m.group(3) + ":" + m.group(4) + "/" + m.group(5);
                if (!properties.containsKey("DATABASE_URL")) {
                    properties.put("DATABASE_URL", r2dbcUrl);
                }
            }
        } else {
            mapIfMissing(properties, "POSTGRES_HOST", "DATABASE_HOST");
            mapIfMissing(properties, "POSTGRES_PORT", "DATABASE_PORT");
            mapIfMissing(properties, "POSTGRES_USER", "DATABASE_USER");
            mapIfMissing(properties, "POSTGRES_PASSWORD", "DATABASE_PASSWORD");
            mapIfMissing(properties, "POSTGRES_DB", "DATABASE_NAME");
        }
    }

    private void mapIfMissing(Map<String, Object> props, String fromKey, String toKey) {
        String value = (String) props.get(fromKey);
        if (value != null && !value.isBlank() && !props.containsKey(toKey)) {
            props.put(toKey, value);
        }
    }

    private Map<String, Object> loadDotenv() {
        Map<String, Object> properties = new HashMap<>();

        String[] searchPaths = {
                System.getProperty("DOTENV_PATH"),
                System.getenv("DOTENV_PATH"),
                System.getProperty("user.dir") + "/.env",
                System.getProperty("user.dir") + "/../.env",
        };

        for (String path : searchPaths) {
            if (path == null) continue;
            try {
                Resource resource = new FileSystemResource(path);
                if (resource.exists() && resource.isReadable()) {
                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            line = line.trim();
                            if (line.isEmpty() || line.startsWith("#")) continue;
                            int commentIdx = line.indexOf(" #");
                            if (commentIdx > 0) {
                                line = line.substring(0, commentIdx).trim();
                            }
                            int eqIdx = line.indexOf('=');
                            if (eqIdx > 0) {
                                String key = line.substring(0, eqIdx).trim();
                                String value = line.substring(eqIdx + 1).trim();
                                if (value.startsWith("\"") && value.endsWith("\"")) {
                                    value = value.substring(1, value.length() - 1);
                                } else if (value.startsWith("'") && value.endsWith("'")) {
                                    value = value.substring(1, value.length() - 1);
                                }
                                properties.put(key, value);
                            }
                        }
                    }
                    if (!properties.isEmpty()) {
                        break;
                    }
                }
            } catch (Exception ignored) {
            }
        }
        return properties;
    }
}