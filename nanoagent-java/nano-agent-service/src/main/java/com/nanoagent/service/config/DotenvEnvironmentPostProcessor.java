package com.nanoagent.service.config;

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
    private static final Pattern REDIS_URL_PATTERN = Pattern.compile(
            "redis://(?::([^@]*)@)?([^:/]+):(\\d+)(?:/(\\d+))?");

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> dotenv = loadDotenv();
        if (!dotenv.isEmpty()) {
            expandRedisUrl(dotenv);
            PropertySource<?> source = new MapPropertySource(DOTENV_SOURCE_NAME, dotenv);
            environment.getPropertySources().addFirst(source);
        }
    }

    private void expandRedisUrl(Map<String, Object> properties) {
        String redisUrl = (String) properties.get("REDIS_URL");
        if (redisUrl == null || redisUrl.isBlank()) {
            return;
        }

        Matcher m = REDIS_URL_PATTERN.matcher(redisUrl.trim());
        if (m.matches()) {
            String password = m.group(1);
            String host = m.group(2);
            String port = m.group(3);
            String db = m.group(4);

            if (host != null && !properties.containsKey("REDIS_HOST")) {
                properties.put("REDIS_HOST", host);
            }
            if (port != null && !properties.containsKey("REDIS_PORT")) {
                properties.put("REDIS_PORT", port);
            }
            if (password != null && !password.isEmpty() && !properties.containsKey("REDIS_PASSWORD")) {
                properties.put("REDIS_PASSWORD", password);
            }
            if (db != null && !properties.containsKey("REDIS_DB")) {
                properties.put("REDIS_DB", db);
            }
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