package com.nanoagent.frontend.config;

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

public class DotenvEnvironmentPostProcessor implements EnvironmentPostProcessor {

    private static final String DOTENV_SOURCE_NAME = "dotenv";

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> dotenv = loadDotenv();
        if (!dotenv.isEmpty()) {
            PropertySource<?> source = new MapPropertySource(DOTENV_SOURCE_NAME, dotenv);
            environment.getPropertySources().addFirst(source);
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